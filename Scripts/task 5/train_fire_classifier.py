"""
train_fire_classifier.py — Real Fire Classifier (binary CNN)
─────────────────────────────────────────────────────────────
The fire_dataset on Kaggle is classification-only:
  fire_images/      — whole photos containing fire, no boxes
  non_fire_images/  — whole photos without fire

Rather than force this into YOLO (which needs bounding boxes
we don't have and can't honestly fabricate), this trains a
SEPARATE lightweight binary classifier that answers one question
per frame: "is fire present in this scene, yes or no?"

This is what your decision engine actually needs — MissionState
has fire_detected: bool. A classifier supplies this more
reliably than forcing weak/fake boxes through a detector.

ARCHITECTURE:
  MobileNetV2 backbone (pretrained on ImageNet, ~3.4M params)
  -> small classifier head -> sigmoid -> P(fire present)

  MobileNetV2 chosen over a custom CNN because:
    - Pretrained weights already understand textures/colour
      patterns (orange/red/flicker-like patterns transfer well
      from ImageNet's general object recognition)
    - Small enough to run fast on CPU alongside YOLO
    - Far less prone to overfitting on ~1000 images than
      training a CNN from scratch

OUTPUT INTEGRATION:
  Saves fire_classifier.pt — loaded separately from the YOLO
  model. At inference time, run BOTH:
    yolo_detections = yolo_model.predict(frame)       # humans
    fire_present     = fire_classifier.predict(frame)  # fire bool
  Then merge fire_present into MissionState.fire_detected.

Usage:
  python train_fire_classifier.py
  python train_fire_classifier.py --quick   (2 epochs, fast test)
"""

import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image


BASE        = Path(r"c:/Users/User/Documents/internship/IISC/Project 1/sar_data")
FIRE_DIR    = BASE / "raw/rgb/real/fire/fire_dataset/fire_images"
NONFIRE_DIR = BASE / "raw/rgb/real/fire/fire_dataset/non_fire_images"
OUT_DIR     = BASE / "models" / "fire_classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class FireDataset(Dataset):
    """
    Loads fire/non-fire images with labels.
      label 1 = fire present
      label 0 = no fire
    """
    def __init__(self, fire_paths, nonfire_paths, transform):
        self.samples = (
            [(p, 1) for p in fire_paths] +
            [(p, 0) for p in nonfire_paths]
        )
        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model():
    """
    MobileNetV2 with a custom binary classification head.
    """
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )
    return model


def train(quick=False):
    print("=" * 55)
    print("  Fire Classifier Training (MobileNetV2)")
    print("=" * 55)

    if not FIRE_DIR.exists() or not NONFIRE_DIR.exists():
        print("\nDataset not found.")
        print(f"  Expected: {FIRE_DIR}")
        print(f"  Expected: {NONFIRE_DIR}")
        return

    fire_paths    = list(FIRE_DIR.glob("*.png")) + list(FIRE_DIR.glob("*.jpg"))
    nonfire_paths = list(NONFIRE_DIR.glob("*.png")) + list(NONFIRE_DIR.glob("*.jpg"))

    print(f"\nFire images    : {len(fire_paths)}")
    print(f"Non-fire images: {len(nonfire_paths)}")

    if not fire_paths or not nonfire_paths:
        print("Missing images in one or both classes.")
        return

    train_tf, val_tf = build_transforms()
    full_dataset = FireDataset(fire_paths, nonfire_paths, train_tf)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"\nTrain: {n_train}  Val: {n_val}  Device: {DEVICE}")

    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs = 2 if quick else 15
    print(f"Epochs: {epochs}\n")

    best_val_acc = 0.0
    t_start = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct  = 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()

                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        val_loss /= n_val
        val_acc   = correct / n_val
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  "
              f"prec={precision:.3f}  "
              f"recall={recall:.3f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
            "precision": precision, "recall": recall,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "fire_classifier_best.pt")

    t_elapsed = time.time() - t_start

    summary = {
        "model":          "mobilenet_v2_fire_classifier",
        "timestamp":      datetime.now().isoformat(),
        "duration_min":   round(t_elapsed / 60, 1),
        "n_fire":         len(fire_paths),
        "n_nonfire":      len(nonfire_paths),
        "best_val_acc":   round(best_val_acc, 4),
        "final_precision":round(history[-1]["precision"], 4),
        "final_recall":   round(history[-1]["recall"], 4),
        "weights_path":   str(OUT_DIR / "fire_classifier_best.pt"),
        "history":        history,
    }
    (OUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n" + "=" * 55)
    print("  FIRE CLASSIFIER TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Duration       : {t_elapsed/60:.1f} min")
    print(f"  Best val acc   : {best_val_acc:.3f}")
    print(f"  Final precision: {history[-1]['precision']:.3f}")
    print(f"  Final recall   : {history[-1]['recall']:.3f}")
    print(f"\n  Weights: {OUT_DIR / 'fire_classifier_best.pt'}")
    print("\nNext: wire into inference engine - see fire_inference.py")


def load_fire_classifier(weights_path=None):
    """Loads the trained fire classifier for inference."""
    path = weights_path or (OUT_DIR / "fire_classifier_best.pt")
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    _, val_tf = build_transforms()
    return model, val_tf


def predict_fire(model, transform, pil_image):
    """Runs inference on a single PIL image. Returns (bool, confidence)."""
    with torch.no_grad():
        img_t = transform(pil_image).unsqueeze(0).to(DEVICE)
        logit = model(img_t).squeeze()
        prob  = torch.sigmoid(logit).item()
    return prob > 0.5, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    train(quick=args.quick)