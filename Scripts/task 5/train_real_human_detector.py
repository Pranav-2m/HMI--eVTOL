import argparse
import json
import time
from pathlib import Path
from datetime import datetime


BASE       = Path(r"c:/Users/User/Documents/internship/IISC/Project 1/sar_data")
DATASET    = BASE / "dataset_real" / "dataset_real.yaml"
MODELS_DIR = BASE / "models" / "human_detector_real"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG = {
    "model":         "yolov8s.pt",
    "epochs":        40,
    "patience":      8,
    "imgsz":         640,
    "batch":         16,
    "workers":       4,
    "device":        "cpu",
    "lr0":           0.01,
    "lrf":           0.01,
    "momentum":      0.937,
    "weight_decay":  0.0005,
    "warmup_epochs": 3,
    "box":           7.5,
    "cls":           0.5,
    "dfl":           1.5,
    "hsv_h":         0.015,
    "hsv_s":         0.7,
    "hsv_v":         0.4,
    "degrees":       15.0,
    "translate":     0.1,
    "scale":         0.5,
    "shear":         2.0,
    "flipud":        0.3,
    "fliplr":        0.5,
    "mosaic":        1.0,
    "mixup":         0.1,
    "copy_paste":    0.1,
}

def find_latest_checkpoint(models_dir):
    checkpoints = sorted(
        models_dir.glob("*/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return checkpoints[0] if checkpoints else None

def check_dataset():
    print("\nDataset check:")
    if not DATASET.exists():
        print(f"  NOT FOUND: {DATASET}")
        print("  Run merge_real_human_datasets.py first.")
        return False

    for split in ["train", "val", "test"]:
        img_dir = BASE / "dataset_real" / split / "images"
        n = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        print(f"  {split:<8}: {n} images")
    return True


def train(quick=False):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return None

    config = TRAIN_CONFIG.copy()
    if quick:
        config["epochs"]  = 3
        config["batch"]   = 4
        config["workers"] = 0
        print("\nQUICK MODE: 3 epochs")

    run_name = "human_real_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    n_train = len(list((BASE/"dataset_real"/"train"/"images").glob("*.*")))
    est_min = config["epochs"] * n_train / config["batch"] * 0.3 / 60

    print("\n" + "="*55)
    print("  Training REAL human detector - YOLOv8s")
    print("="*55)
    print(f"  Train images : {n_train}")
    print(f"  Epochs       : {config['epochs']}")
    print(f"  Estimated    : ~{est_min:.0f} min (CPU)")
    print()

    checkpoint = find_latest_checkpoint(MODELS_DIR)

    if checkpoint:
        print(f"\nFound checkpoint:")
        print(f"  {checkpoint}")
        print("Resuming training...\n")

        model = YOLO(str(checkpoint))
        resume_training = True

    else:
        print("\nNo previous checkpoint found.")
        print("Starting new training...\n")

        model = YOLO(config["model"])
        resume_training = False
    #model   = YOLO(config["model"])
    t_start = time.time()

    results = model.train(
        data          = str(DATASET),
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
        resume = resume_training,
    )

    t_elapsed = time.time() - t_start
    best_path = MODELS_DIR / run_name / "weights" / "best.pt"

    summary = {
        "run_name":     run_name,
        "model":        "yolov8s_human_real",
        "data_sources": "C2A + SARD (real, human-verified labels)",
        "timestamp":    datetime.now().isoformat(),
        "duration_min": round(t_elapsed/60, 1),
        "n_train":      n_train,
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

    (MODELS_DIR / (run_name + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )

    print("\n" + "="*55)
    print("  REAL HUMAN DETECTOR - TRAINING COMPLETE")
    print("="*55)
    print(f"  Duration : {t_elapsed/60:.1f} min")
    m = summary.get("metrics", {})
    if m:
        print(f"  mAP@50   : {m.get('mAP50', 0):.3f}")
        print(f"  Precision: {m.get('precision', 0):.3f}")
        print(f"  Recall   : {m.get('recall', 0):.3f}")
    print("\n  This mAP reflects performance on REAL disaster/SAR photos,")
    print("  not synthetic ellipses. Compare honestly against the old")
    print("  0.994 number - that figure was measuring a different,")
    print("  easier (fake) task.")
    print(f"\n  Weights: {best_path}")

    return str(best_path), summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if args.epochs:
        TRAIN_CONFIG["epochs"] = args.epochs

    if check_dataset():
        train(quick=args.quick)