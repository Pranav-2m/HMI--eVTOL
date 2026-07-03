import cv2
import numpy as np
import json
import shutil
import argparse
import random
import time
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DATA as BASE, setup_path

RGB_SOURCE_FOLDERS = [
    BASE / "raw" / "rgb" / 'real',
    BASE / "raw" / "rgb" / 'custom',
    BASE / "raw" / "rgb" / 'online',
    BASE / "raw" / "rgb" / 'placeholder',
    BASE / "raw" / "rgb" / 'synthetic_pairs',
]

THERMAL_SOURCE_FOLDERS = [
    BASE / "raw" / "thermal",
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".pnb", ".bmp", ".tiff", ".tif"}

OUT_RGB_PROC = BASE / "raw" / "rgb" / "processed"
OUT_THERMAL_PROC = BASE / "synthetic" / "thermal"
OUT_ANNOTS = BASE / "annotations" / "all"

OUT_DATASET = BASE / "dataset"
OUT_MODELS = BASE / "models" / "rgb_detector"
LOGS_DIR = BASE / "logs"

CLASSES = {0: "human", 1: "fire", 2: "landing_zone"}

TRAIN_CONFIG = {
    "model": "yolov8s.pt",
    "epochs": 20,
    "patience": 5,
    "imgsz": 320,
    "batch": 16,
    "workers": 4,
    "device": "cpu",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "flipud": 0.3,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.1,
}

def setup_directories():
    dirs = [
        OUT_RGB_PROC, OUT_THERMAL_PROC, OUT_ANNOTS, LOGS_DIR,
        OUT_DATASET / "train" / "images",
        OUT_DATASET / "train" / "labels",
        OUT_DATASET / "val" / "images",
        OUT_DATASET / "val" / "labels",
        OUT_DATASET / "test" / "images",
        OUT_DATASET / "test" / "labels",
        OUT_MODELS,
    ] + RGB_SOURCE_FOLDERS + THERMAL_SOURCE_FOLDERS

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(" Directories Ready")

def discover_all_images():
    print("\n Scanning all image sources...")

    already_done = {p.stem for p in OUT_ANNOTS.glob("*.json")}
    print(f"Already annotated: {len(already_done)} images")

    all_images, source_counts = [], {}

    for folder in RGB_SOURCE_FOLDERS:
        if not folder.exists():
            continue
        imgs = [
            p for p in folder.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTS
            and not p.stem.endswith("_thermal")
            and not p.stem.endswith("_thermal_gray")
        ]
        if imgs:
            source_counts[str(folder.name)] = len(imgs)
            all_images.extend(imgs)
            print(f'{folder.name:<30}: {len(imgs):>4} images')

    def clean_stem(path):
        s = path.stem
        for suffix in ["_rgb", "_thermal", "_thermal_gray", "_fused", "_annotated"]:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
            return s
            
    unprocessed = [
        img for img in all_images
        if clean_stem(img) not in already_done
    ]

    print(f"\n Total found: {len(all_images)}")
    print(f" Already done: {len(already_done)}")
    print(f" To process: {len(unprocessed)}")
 
    return unprocessed, already_done, source_counts
    
def generate_thermal_from_rgb(rgb_img):
    H, W = rgb_img.shape[:2]
    thermal = np.zeros((H, W), dtype=np.float32)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV).astype(np.float32)

    thermal += gray / 255.0 * 80 + 40

    sky = np.zeros((H, W), dtype=bool)
    sky[:H//5, :] = True
    sky |= ((hsv[:, :, 0] > 100) & (hsv[:,:,0] < 130) & (hsv[:,:,1] > 40))
    thermal[sky] = np.random.uniform(10, 35, thermal[sky].shape)

    water = ((hsv[:,:,0]> 95) & (hsv[:,:,0] < 135) & (gray < 100))
    thermal[water] = np.random.uniform(20, 45, thermal[water].shape)

    veg = ((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 40))
    thermal[veg] = np.random.uniform(45, 70, thermal[veg].shape)

    concrete = ((hsv[:,:,1] < 30) & (gray > 80) & (gray < 200))
    thermal[concrete] = np.random.uniform(70, 110, thermal[concrete].shape)

    thermal = cv2.GaussianBlur(thermal, (21, 21), 0)
    noise = np.random.normal(0, 5, (H, W)).astype(np.float32)
    thermal = np.clip(thermal + noise, 0, 200).astype(np.uint)

    return thermal

def add_survivor_signatures(thermal_gray, n_humans=None):
    H, W = thermal_gray.shape[:2] if len(thermal_gray.shape) == 3 else thermal_gray.shape
    annotations = []
    n = n_humans or random.randint(1,3)

    for _ in range(n):
        cx = random.randint(40, W-40)
        cy = random.randint(40, H-40)
        rx = random.randint(8, 18)
        ry = random.randint(15, 28)
        intensity = random.randint(175, 225)

        thermal_gray = np.ascontiguousarray(thermal_gray, dtype=np.uint8)
        cv2.ellipse(thermal_gray, (cx, cy), (rx, ry), random.randint(-20, 20), 0, 360, int(intensity), -1)
        thermal_gray = cv2.GaussianBlur(thermal_gray, (7, 7), 0)

        x1 = max(0, cx-rx); y1 = max(0, cy-ry)
        x2 = min(W, cx+rx); y2 = min(H, cy+ry)

        annotations.append({
            "label": "human",
            "class_id": 0,
            "bbox_abs": [x1, y1, x2, y2],
            "bbox_yolo": [cx/W, cy/H, rx*2/W, ry*2/H],
            "intensity": intensity,
        })

    if random.random() < 0.3:
        cx = random.randint(10, W-10)
        cy = random.randint(10, H-10)
        r = random.randint(3, 7)
        cv2.circle(thermal_gray, (cx, cy), r, 255, -1)
        x1 = max(0, cx-r*2); y1 = max(0, cy-r*2)
        x2 = min(W, cx+r*2); y2 = min(H, cy+r*2)
        annotations.append({
            "label": "fire",
            "class_id": 1,
            "bbox_abs": [x1, y1, x2, y2],
            "bbox_yolo": [cx/W, cy/H, r*4/W, r*4/H],
            "intensity": 255,
        })
    return thermal_gray, annotations

def process_images(unprocessed_imgs, start_id=0):
    if not unprocessed_imgs:
        print("No new images to add")
        return 0
    print(f"\n processing {len(unprocessed_imgs)} images...")
    Target_W, Target_H = 640, 640
    processed, skipped = 0, 0

    for i, img_path in enumerate(unprocessed_imgs):
        img_id = start_id + i

        rgb = cv2.imread(str(img_path))
        if rgb is None:
            skipped += 1
            continue

        h, w = rgb.shape[:2]
        if h < 32 or w < 32:
            skipped += 1
            continue

        scale = min(Target_W/w, Target_H/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))
        canvas = np.zeros((Target_H, Target_W, 3), dtype=np.uint8)
        y_off = (Target_H - new_h) // 2
        x_off = (Target_W - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        rgb_norm = canvas

        thermal_gray = generate_thermal_from_rgb(rgb_norm)
        thermal_gray, annotations = add_survivor_signatures(thermal_gray)
        thermal_colored = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)
        fused = cv2.addWeighted(rgb_norm, 0.6, thermal_colored, 0.4, 0)

        sorce_tag = img_path.parent.name[:8]
        name = f"{sorce_tag}_{img_id:05d}"
        cv2.imwrite(str(OUT_RGB_PROC / f"{name}_rgb.jpg"), rgb_norm)
        cv2.imwrite(str(OUT_THERMAL_PROC / f"{name}_thermal.jpg"), thermal_colored)
        cv2.imwrite(str(OUT_THERMAL_PROC / f"{name}_thermal_gray.jpg"), thermal_gray)

        annotation = {
            "image_id": name,
            "source_file": str(img_path),
            "source_folder": img_path.parent.name,
            "image_size": {"height": Target_H, "width": Target_W},
            "objects": annotations,
            "yolo_labels": [
                f"{obj['class_id']}"
                f"{obj['bbox_yolo'][0]:.6f}"
                f"{obj['bbox_yolo'][1]:.6f}"
                f"{obj['bbox_yolo'][2]:.0f}"
                f"{obj['bbox_yolo'][3]:.0f}"
                for obj in annotations
            ]
        }

        with open(OUT_ANNOTS / f"{name}.json", "w") as f:
            json.dump(annotation, f, indent=2)

        with open(OUT_ANNOTS / f"{name}.txt", "w") as f:
            f.write("\n".join(annotation["yolo_labels"]))

        processed += 1
        if (i+1) % 20 == 0 or i == len(unprocessed_imgs)-1:
            print(f"[{i+1:>4}/{len(unprocessed_imgs)}] processed"
                  f"({skipped} skipped)")
        
    print(f"PRocessed: {processed} skipped: {skipped}")
    return processed

def build_fused_dataset(train_r=0.7, val_r=0.2, test_r=0.1):
    print(" Building fused dataset from all annotations...")

    for split in ["train", "val", "test"]:
        for subfolder in ["images", "labels"]:
            d = OUT_DATASET / split / subfolder
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents = True, exist_ok= True)
    annot_files = sorted(OUT_ANNOTS.glob("*.json"))
    if not annot_files:
        print(" No annotations found - run processing")
        return None
    print(f"found {len(annot_files)} total annotations")

    random.seed(42)
    ids = [f.stem for f in annot_files]
    random.shuffle(ids)
    MAX_IMAGES = 4000
    if len(ids) > MAX_IMAGES:
        print(f"  Capping dataset: {len(ids)} → {MAX_IMAGES} images")
        ids = ids[:MAX_IMAGES]
    n = len(ids)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    splits = {
        "train": ids[:t1],
        "val": ids[t1:t2],
        "test": ids[t2:],
    }
    
    stats = {s: {"images": 0, "objects": 0} for s in splits}
    source_stats = {}

    for split_name, split_ids in splits.items():
        for img_id in split_ids:
            annot_path = OUT_ANNOTS / f"{img_id}.json"
            if not annot_path.exists():
                continue
            with open(annot_path) as f:
                annotation = json.load(f)

            src = annotation.get("source_folder", "unknown")
            source_stats[src] = source_stats.get(src, 0) + 1

            img_candidates = [
                OUT_RGB_PROC / f"{img_id}_rgb.jpg",
                OUT_THERMAL_PROC / f"{img_id}_thermal.jpg",
                BASE / "raw" / "rgb" / "synthetic_pairs" / f"{img_id}_rgb.jpg",
            ]

            img = None
            for candidate in img_candidates:
                if candidate.exists():
                    img = cv2.imread(str(candidate))
                    if img is not None:
                        break

            if img is None:
                img = random.randint(30, 150, (480, 640, 3), dtype=np.uint8)

            thermal_path = OUT_THERMAL_PROC / f"{img_id}_thermal.jpg"
            if thermal_path.exists():
                thermal = cv2.imread(str(thermal_path))
                if thermal is not None:
                    t_resized = cv2.resize(thermal, (640, 640))
                    img = cv2.addWeighted(cv2.resize(img, (640, 640)), 0.6, t_resized, 0.4, 0)

            dst_img = OUT_DATASET / split_name / "images" / f"{img_id}.jpg"
            dst_lbl = OUT_DATASET / split_name / "labels" / f"{img_id}.txt"
            cv2.imwrite(str(dst_img), cv2.resize(img, (640, 480)))

            src_lbl = OUT_ANNOTS / f"{img_id}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(annotation.get("yolo_labels",[])))
            stats[split_name]["images"] += 1
            stats[split_name]["objects"] += len(annotation.get("objects", []))

    yaml_path = OUT_DATASET / "dataset.yaml"
    yaml_content = f"""# SAR eVTOL Detection Dataset — Full Pipeline
# Generated: {datetime.now().isoformat()}
# Total images: {sum(s['images'] for s in stats.values())}
path: {str(OUT_DATASET.resolve())}
train: train/images
val:   val/images
test:  test/images
nc: {len(CLASSES)}
names:
  0: human
  1: fire
  2: landing_zone
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("Dataset Splits:")
    for splits, s in stats.items():
        print(f" {split:<8}: {s['images']:>4} images |"
              f"{s['objects']:>4} objects")
        
    print("Images by source folder:")
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        bar = min(30, count // max(1, len(annot_files)//30))
        print(f"{src:<30}: {count:>4} {bar}")
    return yaml_path

def train_model(yaml_path, config, quick=False):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed")
        return None
    
    if quick:
        config = config.copy()
        config["epochs"] = 3
        config["batch"] = 4
        config["workers"] = 0
        print("Quick mode 3 epochs")

    run_name = f"full_ {datetime.now().strftime('%Y%m%d_%H%M%S')}"

    n_train = len(list((OUT_DATASET / "train" / "images").glob("*.jpg")))
    n_val = len(list((OUT_DATASET / "val" / "images").glob("*.jpg")))
    print(f"\n { '='*55}")
    print("Training YOLOv8s on full dataset")
    print(f" Train images: {n_train}")
    print(f"Val images: {n_val}")
    print(f"Epochs: {config['epochs']}")
    print(f"  Batch       : {config['batch']}")
    est = config['epochs'] * n_train * 0.3 / 60
    print(f"  Est. time   : ~{est:.0f} min (CPU)")
    model = YOLO(config["model"])
    t_start = time.time()

    results = model.train(
        data         = str(yaml_path),
        epochs       = config["epochs"],
        imgsz        = config["imgsz"],
        batch        = config["batch"],
        patience     = config["patience"],
        lr0          = config["lr0"],
        lrf          = config["lrf"],
        momentum     = config["momentum"],
        weight_decay = config["weight_decay"],
        warmup_epochs= config["warmup_epochs"],
        box          = config["box"],
        cls          = config["cls"],
        dfl          = config["dfl"],
        device       = config["device"],
        workers      = config["workers"],
        project      = str(OUT_MODELS),
        name         = run_name,
        exist_ok     = True,
        verbose      = True,
        save         = True,
        plots        = True,
        hsv_h        = config["hsv_h"],
        hsv_s        = config["hsv_s"],
        hsv_v        = config["hsv_v"],
        degrees      = config["degrees"],
        translate    = config["translate"],
        scale        = config["scale"],
        shear        = config["shear"],
        flipud       = config["flipud"],
        fliplr       = config["fliplr"],
        mosaic       = config["mosaic"],
        mixup        = config["mixup"],
        copy_paste   = config["copy_paste"],
    )
    t_elapsed = time.time() - t_start
    best_path = OUT_MODELS / run_name / "weights" / "best.pt"

    summary = {
        "run_name":    run_name,
        "timestamp":   datetime.now().isoformat(),
        "duration_min":round(t_elapsed/60, 1),
        "n_train":     n_train,
        "n_val":       n_val,
        "best_weights":str(best_path),
        "classes":     CLASSES,
    }
    try:
        summary['metrics'] = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)",0)),
            "mAP50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
        }
    except Exception:
        summary["metrics"] = {}
    
    log_path = LOGS_DIR / f"{run_name}_summary.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f" TRAINING COMPLETE ")
    print(f"{'='*55}")
    print(f"Duration: {t_elapsed/60:.1f} minutes")
    m = summary.get("metrics", {})
    if m:
        print(f"  mAP@50     : {m.get('mAP50', 0):.3f}")
        print(f"  Precision  : {m.get('precision', 0):.3f}")
        print(f"  Recall     : {m.get('recall', 0):.3f}")
    print(f"\n  Best weights: {best_path}")
    print(f" Plots       : {OUT_MODELS / run_name}/results.png")
 
    return best_path

def run(args):
    print("\n [1/5] setting up directories")
    setup_directories()

    if args.retrain_only:
        print(" --retrain-only: skipping data processing")
        yaml_path = OUT_DATASET / "dataset.yaml"
        if not yaml_path.exists():
            print(" No dataset found. Run without --retrain-only first")
            return
        if not args.skip_train:
            train_model(yaml_path, TRAIN_CONFIG, quick = args.quick)
            return
        
    print("\n[2/5] Discovering images across all sources...")
    unprocessed, already_done, source_counts = discover_all_images()
    if args.source:
        before = len(unprocessed)
        unprocessed = [
            p for p in unprocessed
            if args.source.lower() in p.parent.name.lower()
        ]
        print(f" --source filter '{args.source}"
              f"{before} -> {len(unprocessed)} images")
        
    print("\n[3/5] Process images -> thermal -> annotations...")
    start_id = len(list(OUT_ANNOTS.glob("*.json")))
    n_processed = process_images(unprocessed, start_id=start_id)
    total_annots = len(list(OUT_ANNOTS.glob("*.json")))
    print(f"\n total annotations now: {total_annots}")

    print("\n[4/5] Building Fused Training Dataset...")
    yaml_path = build_fused_dataset()
    if yaml_path is None:
        print(" Dataset build failed")
        return
    
    if not args.skip_train:
        print("\n[5/5] Training YOLOv8s on full dataset...")
        train_model(yaml_path, TRAIN_CONFIG, quick=args.quick)
    else:
        print("\n[5/5] --skip-train: skipping model training...")
        print(f"Dataset ready at: {yaml_path}")
        print(" Run without --skip-train to train the model")

    print("PIPELINE COMPLETE")
    print(f"  New images processed : {n_processed}")
    print(f"  Total dataset size   : {total_annots}")
    for split in ["train", "val", "test"]:
        n = len(list((OUT_DATASET/split/"images").glob("*.jpg")))
        print(f"{split:<8}: {n} images")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR eVTOL - Unified Data & Training Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Process data only, skip training")
    parser.add_argument("--retrain-only", action="store_true", help="Skip data processing, retrain on existing dataset")
    parser.add_argument("--quick", action="store_true", help="Quick test: 3 training epochs only")
    parser.add_argument("--source", type=str, default=None, help="Process only source folder like real, custom or online")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")

    args = parser.parse_args()
    if args.epochs: TRAIN_CONFIG["epochs"] = args.epochs
    if args.batch: TRAIN_CONFIG["batch"] = args.batch
    run(args)
    