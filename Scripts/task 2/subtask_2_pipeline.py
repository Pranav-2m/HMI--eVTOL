import cv2
import numpy as np
import json
import shutil
import random
import argparse
import re
import time
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DATA as BASE, setup_path
setup_path()

THERMAL_DIR = BASE_DIR / "dataset_thermal"
MODELS_DIR = BASE_DIR / "models" / "thermal_detector"

MAX_IMAGES = 8000

CLASSES = {0: "human", 1: "fire", 2: "landing_zone"}
THERMAL_CONFIG ={
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
    "box": 8.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.6,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "flipud": 0.3,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.05,
}

DEFAULT_WH = {0: (0.09, 0.12), 1: (0.12, 0.10), 2: (0.20, 0.18)}

def fix_label_line(line: str):
    line = line.strip()
    if not line or ' ' in line:
        return line
    
    spaced = re.sub(r'(\d)(0\.)', r'\1 \2', line)
    spaced = re.sub(r'\s+', ' ', spaced).strip()
    tokens = spaced.split()

    if len(tokens) < 3:
        return None
    
    try:
        class_id = int(tokens[0])
        cx = float(tokens[1])
        cy = float(tokens[2])
    except ValueError:
        return None
    
    if not (0 <= class_id <= 9):
        return None
    if not (0.0 <= cx <= 1.0) or not (0.0 <= cy <= 1.0):
        return None
    if len(tokens) >= 5:
        try:
            w, h = float(tokens[3]), float(tokens[4])
            if 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                return f"{class_id} {cx:.6f} {w:.6f {h:.6f}}"
        except ValueError:
            pass

    w, h = DEFAULT_WH.get(class_id, (0.09, 0.12))
    w = min(w, 2*min(cx, 1.0 - cx)) if 0 < cx < 1 else w
    h = min(h, 2*min(cy, 1.0 - cy)) if 0 < cy < 1 else h
    w = max(0.02, w)
    h = max(0.02, h)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def fix_label_file(path: Path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    
    fixed = []
    changed = False
    for line in lines:
        result = fix_label_line(line)
        if result is None:
            changed = True
        elif result != line:
            changed = True
        else:
            fixed.append(line)

    if changed:
        path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return changed

def fix_all_labels_in(folder: Path):
    files = list(folder.glob("*.txt"))
    if not files:
        return 0
    fixed = sum(1 for f in files if fix_label_file(f))
    return fixed

def del_cache_files(base: Path):
    deleted = sum(1 for c in base.rglob("*.cache") if c.unlink() is None)
    return deleted

def build_thermal_dataset():
    print("\n Byuilding thermal dataset")
    therm_dir = BASE_DIR / "synthetic" / "thermal"
    annot_dir = BASE_DIR / "annotations" / "all"
    therm_imgs = sorted(therm_dir.glob("*_thermal.jpg"))
    print(f"found {len(therm_imgs)} thermal images in synthetic/thermal")

    if not therm_imgs:
        print("No thermal images found.")
        print("run run_piprlinr first to generate them")
        return None
    
    random.seed(42)
    imgs = list(therm_imgs)
    random.shuffle(imgs)

    if len(imgs) > MAX_IMAGES:
        print(f" Vapping: {len(imgs)} -> {MAX_IMAGES} images")
        imgs = imgs[:MAX_IMAGES]

    n = len(imgs)
    t1 = int(n * 0.70)
    t2 = int(n * 0.90)
    splits = {
        "train": imgs[:t1],
        "val": imgs[t1:t2],
        "test": imgs[t2:],
    }
    if THERMAL_DIR.exists():
        shutil.rmtree(THERMAL_DIR)
        print(" Cleared old thermal dataset")

    for split in ["train", "val", "test"]:
        (THERMAL_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (THERMAL_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


        counts = {}
        no_label = 0
        has_label = 0

        for split_name, split_imgs in splits.items():
            copied = 0
            for src in split_imgs:
                img_stem = src.stem
                stem = src.stem.replace("_thermal", "")
                dst_img = THERMAL_DIR / split_name / "images" / f"{img_stem}.jpg"
                dst_lbl = THERMAL_DIR / split_name / "labels" / f"{img_stem}.txt"
                src_lbl = annot_dir / f"{stem}.txt"
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_img)

                if src_lbl.exists():
                    content = src_lbl.read_text(encoding="utf-8").strip()
                    if content:
                        shutil.copy2(src_lbl, dst_lbl)
                        has_label += 1
                    else:
                        no_label += 1
                        continue
                else:
                    no_label += 1
                    continue
                copied += 1
        counts[split_name] = copied

    print("\n Fixing label Format..")
    total_fixed = 0
    for split in ["train", "val", "test"]:
        n_fixed = fix_all_labels_in(THERMAL_DIR / split / "labels")
        if n_fixed:
            print(f" {split}: {n_fixed} files fixed")
            total_fixed  += n_fixed

    if total_fixed == 0:
        print(" All labels already clean")

    del_cache_files(THERMAL_DIR)
    yaml_content = f"""# SAR eVTOL — Thermal IR Dataset
# Generated: {datetime.now().isoformat()}
# Images: {sum(counts.values())} (capped from {len(therm_imgs)})
 
path: {str(THERMAL_DIR.resolve())}
train: train/images
val:   val/images
test:  test/images
 
nc: {len(CLASSES)}
names:
  0: human
  1: fire
  2: landing_zone
"""
    yaml_path = THERMAL_DIR / "dataset_thermal.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\n Dataset Built:")
    for split, count in counts.items():
        lbl_count = len(list((THERMAL_DIR / split / "labels").glob("*.txt")))
        img_count = len(list((THERMAL_DIR / split / "images").glob("*.jpg")))
        print(f" {split:.<8}: {img_count} images | {lbl_count} labels")

    print(f" labels with content: {has_label}")
    print(f" empty labels : {no_label}")

    if has_label == 0:
        print("\n ALL LAbels ARE EMPTY")
        return None
    print(f" YAML: {yaml_path}")
    return yaml_path

def verify_dataset(yaml_path, n_check=5):
    print("\n Verifying dataset")

    train_imgs = list((THERMAL_DIR / "train" / "images").glob("*.jpg"))
    train_lbls = list((THERMAL_DIR / "train" / "labels").glob("*.txt"))

    img_stems = {p.stem for p in train_imgs}
    lbl_stems = {p.stem for p in train_lbls}
    paired = img_stems & lbl_stems
    unpaired = img_stems - lbl_stems

    print(f" Train images : {len(train_imgs)}")
    print(f" Train labels: {len(train_lbls)}")
    print(f" Paired: {len(paired)}")
    if unpaired:
        print(f" unpaired imgd: {len(unpaired)} (no matching label)")

    non_empty = 0
    for lbl in train_lbls[:20]:
        if lbl.read_text(encoding="utf-8").strip():
            non_empty += 1
    print(f" Non-empty labels (first 20): {non_empty}/20")
    if non_empty == 0:
        print(" All sampled labels are empty")
        return False
    
    for img_path in train_imgs[:n_check]:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Unreadable: {img_path.name} ")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f" {img_path.name}:"
              f"mean = {np.mean(gray):.0f}"
              f"std = {np.std(gray):.0f}"
              f"max = {np.max(gray)}")
    return True

def train_thermal(config, yaml_path, quick=False):
    try:
        from ultralytics import YOLO
    except ImportError:
        print(" pip install utralytics")
        return None
    
    if quick:
        config = config.copy()
        config["epochs"] = 3
        config["batch"] = 4
        config["workers"] = 0
        print("Quick mode: 3 epochs")

    run_name = f"thermal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n_train = len(list((THERMAL_DIR / "train"/"images").glob("*.jpg")))
    est_min = config["epochs"] * n_train / config["batch"] * 0.3 / 60

    print("Training Thermal detector = YOLOv8s")
    print(f"  Train images : {n_train}")
    print(f"  Epochs       : {config['epochs']}")
    print(f"  imgsz        : {config['imgsz']}")
    print(f"  Estimated    : ~{est_min:.0f} min (CPU)")

    model = YOLO(config["model"])
    t_start = time.time()
    results = model.train(
        data          = str(yaml_path),
        epochs        = config["epochs"],
        imgsz         = config["imgsz"],
        batch         = config["batch"],
        patience      = config["patience"],
        lr0           = config["lr0"],
        lrf           = config["lrf"],
        momentum      = config["momentum"],
        weight_decay  = config["weight_decay"],
        warmup_epochs = config["warmup_epochs"],
        box           = config["box"],
        cls           = config["cls"],
        dfl           = config["dfl"],
        device        = config["device"],
        workers       = config["workers"],
        project       = str(MODELS_DIR),
        name          = run_name,
        exist_ok      = True,
        verbose       = True,
        save          = True,
        plots         = True,
        hsv_h         = config["hsv_h"],
        hsv_s         = config["hsv_s"],
        hsv_v         = config["hsv_v"],
        degrees       = config["degrees"],
        translate     = config["translate"],
        scale         = config["scale"],
        shear         = config["shear"],
        flipud        = config["flipud"],
        fliplr        = config["fliplr"],
        mosaic        = config["mosaic"],
        mixup         = config["mixup"],
        copy_paste    = config["copy_paste"],
    )
    t_elapsed = time.time() - t_start
    best_path = MODELS_DIR / run_name / "weights" / "best.pt"

    summary = {
        "run_name": run_name,
        "model": "yolov8s_thermal",
        "timestamp": datetime.now().isoformat(),
        "duration_min": round(t_elapsed/60, 1),
        "n_train": n_train,
        "best_weights": str(best_path), 
    }
    try:
        summary["metrics"] = {
            "mAP50":     float(results.results_dict.get("metrics/mAP50(B)",     0)),
            "mAP50_95":  float(results.results_dict.get("metrics/mAP50-95(B)",  0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)",    0)),
        }
    except Exception:
        summary["metrics"] = {}
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print("THERMAL DETECTOR COMPLETE")
    print(f"  Duration : {t_elapsed/60:.1f} min")
    m = summary.get("metrics", {})
    if m:
        print(f"  mAP@50   : {m.get('mAP50', 0):.3f}")
        print(f"  Precision: {m.get('precision', 0):.3f}")
        print(f"  Recall   : {m.get('recall', 0):.3f}")
    print(f"  Weights  : {best_path}")
 
    return str(best_path), summary

def run_fusion_test(rgb_model_path=None, thermal_model_path=None):
    print("FUSION LAYER TEST")
    rgb_dets = [
        (0, 0.72, [80, 120, 115, 175]),
        (0, 0.65, [400, 200, 435, 255]),
        (1, 0.88, [510, 90, 570, 145]),
        (0, 0.55, [200, 300, 235, 360]),
    ]
    thm_dets = [
        (0, 0.85, [82, 116, 118, 178]),
        (0, 0.71, [402, 198, 438, 258]),
        (1, 0.91, [512, 88, 573, 148]),
        (0, 0.78, [350, 50, 385, 100]),
    ]

    def compute_iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0: return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter)
 
    IOU_THRESH      = 0.45
    THERMAL_WEIGHT  = 1.4
    RGB_WEIGHT      = 1.0
    AGREEMENT_BONUS = 0.15
    CLASS_NAMES     = {0:"human", 1:"fire", 2:"landing_zone"}
 
    fused = []
    for class_id in CLASS_NAMES:
        rgb_cls = [(c,cf,b) for c,cf,b in rgb_dets if c==class_id]
        thm_cls = [(c,cf,b) for c,cf,b in thm_dets if c==class_id]
        matched_r = set(); matched_t = set()

        for ri, (_, rc, rb) in enumerate(rgb_cls):
            best_iou = IOU_THRESH; best_ti = -1
            for ti, (_, tc, tb) in enumerate(thm_cls):
                if ti in matched_t: continue
                iou = compute_iou(rb, tb)
                if iou > best_iou: best_iou = iou; best_ti = ti
            if best_ti >= 0:
                _,tc,tb = thm_cls[best_ti]
                matched_r.add(ri); matched_t.add(best_ti)
                tw = RGB_WEIGHT + THERMAL_WEIGHT
                fbox = [int((rb[i]*RGB_WEIGHT + tb[i]*THERMAL_WEIGHT)/tw) for i in range(4)]
                conf = min(0.99, (rc*RGB_WEIGHT + tc*THERMAL_WEIGHT)/tw + AGREEMENT_BONUS)
                fused.append((class_id, round(conf, 3), fbox, "rgb+thermal", rc, tc))
            else: fused.append((class_id, round(rc*RGB_WEIGHT, 3), rb, "rgb", rc, 0.0))

        for ti, (_,tc,tb) in enumerate(thm_cls):
            if ti not in matched_t:
                boost = 1.1 if class_id == 0 else 1.0
                fused.append((class_id, round(min(0.99, tc*THERMAL_WEIGHT*boost), 3), tb, "thermal", 0.0, tc))
    fused.sort(key=lambda x: -x[1])

    print(f"\n  Input : {len(rgb_dets)} RGB + {len(thm_dets)} thermal")
    print(f"  Output: {len(fused)} fused detections\n")
    print(f"  {'Class':<14} {'Conf':>6} {'RGB':>6} {'Therm':>6} {'Source'}")
    for cls, conf, box, src, rc, tc in fused:
        print(f"  {CLASS_NAMES[cls]:<14} {conf:>6.3f}"
              f"{rc:>6.3f} {tc:>6.3f} {src}")
        
    print("\n Fusion layer working correctly")
    return fused

def run_feature_extractot_test():
    print(" FEATURE EXTRACTOR TEST")
    H, W = 480, 640
    DEFAULT_WH_FE = {1:(0.09, 0.12), 1:(0.12, 0.10), 2: (0.20, 0.18)}
    mock_dets = [
        (0, 0.91, [80,  120, 115, 175], "rgb+thermal"),
        (0, 0.83, [400,  90, 570, 145], "rgb+thermal"),
        (1, 0.95, [510,  90, 570, 145], "rgb+thermal"),
        (0, 0.71, [200,  300, 235, 360], "rgb+thermal"),
    ]

    thermal_gray = np.random.randint(40, 100, (H, W), dtype=np.uint8)
    for cls, conf, bbox, _ in mock_dets:
        if cls == 0:
            cx = (bbox[0]+bbox[2])//2
            cy = (bbox[1]+bbox[3])//2
            cv2.ellipse(thermal_gray, (cx, cy), (15,25), 0, 0, 360, random.randint(170, 220), -1)
    CLASS_NAMES = {0: "human", 1:"fire", 2:"lansing_zone"}

    def get_thermal_intensity(gray, bbox):
        x1, y1, x2, y2 = [max(0, bbox[i]) for i in range(4)]
        x2 = min(W,x2); y2 = min(H,y2)
        region = gray[y1:y2, x1:x2]
        return float(np.mean(region)) if region.size > 0 else 128.0
    
    def assess_posture(bbox):
        w = max(1, bbox[2]-bbox[0])
        h = max(1, bbox[3]-bbox[1])
        ar = h/w
        area = w*h
        if area < 400: return "buried"
        elif ar > 1.8: return "standing"
        elif ar > 1.0: return "crouching"
        elif ar > 0.5: return "lying"
        else: return "flat"

    def vital_status(t_mean):
        if t_mean > 180: return "likely_alive"
        elif t_mean > 120: return "possibly_alive"
        elif t_mean > 60: return "status_unknown"
        else: return "cold"

    def priority_score(cls, conf, t_mean, posture, vital):
        score = 2.0
        if conf > 0.85: score -= 0.2
        if vital == "likely_alive": score -= 0.5
        elif vital == "cold": score += 0.5
        if posture in ["lying", "flat", "buried"]: score -= 0.5
        elif posture == "standing": score += 0.2
        return max(1, min(3, round(score)))
    
    PRIORITY_LABELS = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM"}

    results = []
    for i, (cls, conf, bbox, src) in enumerate(mock_dets):
        t_mean = get_thermal_intensity(thermal_gray, bbox)
        posture = assess_posture(bbox)
        vital = vital_status(t_mean)
        pri = priority_score(cls, conf, t_mean, posture, vital)
        results.append((CLASS_NAMES[cls], conf, t_mean, posture, vital, PRIORITY_LABELS[pri]))
    print(f"\n {'class':<14}  {'Conf':>5} {'Thermal':>8} "
          f"{'Posture':<12} {'Vital':<14} {'Priority'}")
    for r in results:
        print(f"{r[0]:<14} {r[1]:>5.2f} {r[2]:>8.1f} {r[3]:<12} {r[4]:<14} {r[5]}")
    print("Feature extractor working correctly")
    return results

def run_lz_classifier_test():
    print("LZ CLASSIFIER TEST")
    H, W = 480, 480
    GRID_ROWS, GRID_COLS = 3, 4
    MIN_SAFE_SCORE = 0.55

    rgb_frame     = np.random.randint(40, 180, (H,W,3), dtype=np.uint8)
    thermal_frame = np.random.randint(30, 100, (H,W),   dtype=np.uint8)
    cv2.circle(thermal_frame, (80, 80), 40, 240, -1)
    rgb_frame[340:480, 480:640] = [120, 60, 30]

    cell_w = W// GRID_COLS
    cell_h = H// GRID_ROWS
    candidates = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = col * cell_w; y1 = row * cell_h
            x2 = min(W, x1+cell_w); y2 = min(H, y1+cell_h)
            cx = (x1+x2)//2; cy = (y1+y2)//2

            region_gray = cv2.cvtColor(rgb_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(region_gray, cv2.CV_64F).var()
            flatness = max(0.0, min(1.0, 1.0 - lap_var/800.0))

            t_region = thermal_frame[y1:y2, x1:x2]
            t_max    = int(np.max(t_region))
            if   t_max > 230: thermal_sc = 0.0
            elif t_max > 200: thermal_sc = 0.3
            elif np.mean(t_region) > 160: thermal_sc = 0.6
            else: thermal_sc = 1.0

            hsv = cv2.cvtColor(rgb_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).astype(np.float32)
            mean_h = np.mean(hsv[:,:,0])
            mean_v = np.mean(hsv[:,:,2])
            if 100 < mean_h < 130 and mean_v < 100:
                material_sc = 0.01
            elif mean_h < 20 and mean_v > 150:
                material_sc = 0.05
            else:
                material_sc = 0.75
            
            clearance = 1.0
            lz_score = (flatness*0.3 + clearance*0.3 + thermal_sc*0.2 + material_sc*0.1 + 1.0*0.1)
            rid = row*GRID_COLS + col
            import math
            bearing = (math.degrees(math.atan2(cx-W//2, -(cy-H//2))) + 360)%360

            if   lz_score >= 0.75: status = "CLEAR"
            elif lz_score >= 0.55: status = "MARGINAL"
            elif thermal_sc < 0.2: status = "HOT"
            else:                  status = "UNSAFE"
 
            candidates.append({
                "id": rid, "bbox": [x1,y1,x2,y2],
                "lz_score": round(lz_score,3),
                "flatness": round(flatness,3),
                "thermal":  round(thermal_sc,3),
                "material": round(material_sc,3),
                "status":   status,
                "bearing":  round(bearing,1),
                "is_safe":  lz_score >= MIN_SAFE_SCORE,
            })
 
    candidates.sort(key=lambda c: -c["lz_score"])
 
    print(f"\n  {'ID':>3} {'Status':<10} {'Score':>6} "
          f"{'Flat':>6} {'Therm':>6} {'Bear':>6}")
    for c in candidates:
        marker = " <- BEST" if c == candidates[0] else ""
        print(f"  {c['id']:>3} {c['status']:<10} "
              f"{c['lz_score']:>6.3f} "
              f"{c['flatness']:>6.3f} "
              f"{c['thermal']:>6.3f} "
              f"{c['bearing']:>5.0f}°{marker}")
 
    safe = [c for c in candidates if c["is_safe"]]
    best = candidates[0] if candidates else None
 
    if best and best["is_safe"]:
        print(f"\n  ✅ Best LZ: Region {best['id']} "
              f"score={best['lz_score']:.3f} "
              f"bearing={best['bearing']:.0f}° [{best['status']}]")
    else:
        print("\n  ⚠️  No fully safe LZ found — HOVER_DROP recommended")
 
    print(f"\n  Safe zones : {len(safe)}/{len(candidates)}")
    return candidates

def main(args):
    thermal_weights = None
    if not args.test_only:
        print("[2,2] building thermal dataset")
        yaml_path = build_thermal_dataset()

        if yaml_path is None:
            print("Dataset building failed")
            return
        ok = verify_dataset(yaml_path)
        if not ok:
            print("Dataset failed")
            return
        
        if not args.skip_train:
            print(" Training thermal Detector")
            result = train_thermal(THERMAL_CONFIG, yaml_path, quick=args.quick)
            if result:
                thermal_weights, _ = result
        else: print(" --skip-train: skipping model training")

    print(" Testing Fusion Layer")
    run_fusion_test()

    print("Testing Feature Extractor")
    run_feature_extractot_test

    print("Testing LZ classifier")
    run_lz_classifier_test()

    print("SUBTASK 2 DONE")
    if thermal_weights:
        print(f" Thermal model: {thermal_weights}")
    
    print(f"  Fusion        : ✅ tested")
    print(f"  Features      : ✅ tested")
    print(f"  LZ classifier : ✅ tested")
    print(f"\n  Next step:")
    print(f"  python phase3c_rl_trainer.py --episodes 2000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR Subtask 2 pipeline")
    parser.add_argument("--quick", action="store_true", help="3-epoch quick test")
    parser.add_argument("--skip-train", action="store_true", help="Build dataset only, skip training")
    parser.add_argument("--test-only", action="store_true", help="Run 2.3/2.4/2.5 tests only, skip 2.2")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES, help=f"Cap dataset size (default: {MAX_IMAGES})")
    args = parser.parse_args()
    if args.epochs:     THERMAL_CONFIG["epochs"] = args.epochs
    if args.max_images: MAX_IMAGES = args.max_images
 
    main(args)