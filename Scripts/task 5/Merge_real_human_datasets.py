"""
merge_real_human_datasets.py — Real Human Dataset Merger
─────────────────────────────────────────────────────────
Replaces synthetic human annotations with REAL, human-verified
bounding boxes from C2A and SARD.

WHY THIS REPLACES THE OLD PIPELINE OUTPUT:
  Your previous 19,845 "online" images had FAKE annotations —
  ellipses painted at random coordinates by add_survivor_signatures().
  C2A and SARD are already YOLO format with REAL bounding boxes
  drawn by humans around actual people in actual disaster/SAR photos.

BOTH DATASETS ARE SINGLE-CLASS "human" (class 0) — no remapping
needed, they already match your existing class scheme.

WHAT THIS SCRIPT DOES:
  1. Scans C2A and SARD train/val/test splits
  2. Copies images + labels directly (already valid YOLO format)
  3. Filters out any malformed/empty label files
  4. Merges into sar_data/dataset_real/ — a SEPARATE clean dataset
     (kept apart from your old synthetic dataset so you can compare
      model performance trained on real vs synthetic data directly)
  5. Builds dataset_real.yaml ready for YOLOv8 training

Usage:
  python merge_real_human_datasets.py
  python merge_real_human_datasets.py --max-images 6000
"""

import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime


# ── Paths — update if your folder names differ ────────────
BASE = Path(r"c:/Users/User/Documents/internship/IISC/Project 1/sar_data")

SOURCES = {
    "c2a": {
        "root": BASE / "raw/rgb/real/c2a/C2A_Dataset/new_dataset3",
    },
    "sard": {
        "root": BASE / "raw/rgb/real/sard/search-and-rescue",
    },
}

OUT_DATASET = BASE / "dataset_real"
CLASSES     = {0: "human"}


def find_split_dir(root, split_name):
    """
    Finds the images/labels folders for a split, handling
    naming differences (val vs valid) between datasets.
    """
    candidates = [split_name]
    if split_name == "val":
        candidates.append("valid")

    for name in candidates:
        img_dir = root / name / "images"
        lbl_dir = root / name / "labels"
        if img_dir.exists() and lbl_dir.exists():
            return img_dir, lbl_dir
    return None, None


def validate_label(lbl_path):
    """
    Confirms a label file has at least one valid 5-value YOLO line.
    Rejects empty or malformed files so we don't poison the merge
    with bad data the way the old pipeline did.
    """
    try:
        content = lbl_path.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    if not content:
        return False
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        try:
            cls = int(parts[0])
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            return False
        if cls != 0:
            return False
        if not all(0.0 <= v <= 1.0 for v in vals):
            return False
    return True


def collect_pairs(source_name, root):
    """
    Collects all valid (image, label) pairs from a dataset's
    train/val/test splits, tagging each with its source.
    """
    pairs = []
    for split in ["train", "val", "test"]:
        img_dir, lbl_dir = find_split_dir(root, split)
        if img_dir is None:
            continue

        images = (list(img_dir.glob("*.jpg")) +
                  list(img_dir.glob("*.jpeg")) +
                  list(img_dir.glob("*.png")))

        kept, skipped = 0, 0
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists() and validate_label(lbl):
                pairs.append((img, lbl, source_name))
                kept += 1
            else:
                skipped += 1

        print(f"  {source_name}/{split:<6}: {kept} valid, "
              f"{skipped} skipped (missing/invalid label)")

    return pairs


def build_dataset(max_images=None, split_ratios=(0.70, 0.20, 0.10)):
    print("=" * 55)
    print("  Real Human Dataset Merger - C2A + SARD")
    print("=" * 55)

    all_pairs = []

    print("\nScanning C2A...")
    if SOURCES["c2a"]["root"].exists():
        all_pairs += collect_pairs("c2a", SOURCES["c2a"]["root"])
    else:
        print(f"  NOT FOUND: {SOURCES['c2a']['root']}")

    print("\nScanning SARD...")
    if SOURCES["sard"]["root"].exists():
        all_pairs += collect_pairs("sard", SOURCES["sard"]["root"])
    else:
        print(f"  NOT FOUND: {SOURCES['sard']['root']}")

    print(f"\nTotal valid real human pairs: {len(all_pairs)}")

    if not all_pairs:
        print("\nNo valid pairs found - check SOURCES paths above.")
        return None

    # Shuffle and cap
    random.seed(42)
    random.shuffle(all_pairs)

    if max_images and len(all_pairs) > max_images:
        print(f"Capping: {len(all_pairs)} -> {max_images} images")
        all_pairs = all_pairs[:max_images]

    # Split
    n = len(all_pairs)
    t1 = int(n * split_ratios[0])
    t2 = int(n * (split_ratios[0] + split_ratios[1]))
    splits = {
        "train": all_pairs[:t1],
        "val":   all_pairs[t1:t2],
        "test":  all_pairs[t2:],
    }

    # Clear and rebuild output dirs
    if OUT_DATASET.exists():
        shutil.rmtree(OUT_DATASET)
    for split in ["train", "val", "test"]:
        (OUT_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)

    print()
    source_counts = {"c2a": 0, "sard": 0}
    for split_name, pairs in splits.items():
        for i, (img, lbl, source) in enumerate(pairs):
            name = f"{source}_{i:05d}{img.suffix}"
            shutil.copy2(img, OUT_DATASET / split_name / "images" / name)
            shutil.copy2(lbl, OUT_DATASET / split_name / "labels" /
                        f"{source}_{i:05d}.txt")
            source_counts[source] += 1
        print(f"  {split_name:<8}: {len(pairs)} images copied")

    yaml_content = """# SAR - Real Human Detection Dataset
# Sources: C2A (disaster scenarios) + SARD (aerial search & rescue)
# Generated: """ + datetime.now().isoformat() + """
# Total images: """ + str(n) + """

path: """ + str(OUT_DATASET.resolve()) + """
train: train/images
val:   val/images
test:  test/images

nc: 1
names:
  0: human
"""
    yaml_path = OUT_DATASET / "dataset_real.yaml"
    yaml_path.write_text(yaml_content)

    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)
    print(f"  C2A images : {source_counts['c2a']}")
    print(f"  SARD images: {source_counts['sard']}")
    print(f"  Total      : {n}")
    print(f"  YAML       : {yaml_path}")
    print("\nNext: train YOLOv8s on this real dataset:")
    print("  python train_real_human_detector.py")

    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=8000,
        help="Cap total images (default 8000, use 0 for no cap)")
    args = parser.parse_args()
    cap = args.max_images if args.max_images > 0 else None
    build_dataset(max_images=cap)