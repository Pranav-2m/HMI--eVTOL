import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
import torch
from PIL import Image
import numpy as np
import cv2

from train_fire_classifier import build_model, build_transforms, DEVICE

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAR_DATA as BASE, TASK_FOLDERS, find_best_model

try:
    from step5b_inference_engine import Detection, CLASS_INFO, \
        PRIORITY_LABELS, get_location_hint, compute_priority
except ImportError:
    @dataclass
    class Detection:
        class_id: int
        class_name: str
        confidence: float
        bbox: List[int]
        priority: int
        priority_label: str
        color: tuple
        location_hint: str
        thermal_intensity: float

    def get_location_hint(bbox, fw=640, fh=480):
        return "Centre"

    def compute_priority(class_id, confidence, thermal, bbox, fw=640, fh=480):
        return 2


class CombinedInferenceEngine:
    def __init__(self, human_model_path=None, fire_model_path=None):
        self.human_model    = None
        self.fire_model     = None
        self.fire_transform = None
        self.mock_mode      = False

        self._load_human_model(human_model_path)
        self._load_fire_model(fire_model_path)

        if self.human_model is None and self.fire_model is None:
            print("[CombinedInferenceEngine] No models loaded - MOCK mode")
            self.mock_mode = True

    def _load_human_model(self, path=None):
        search = [path] if path else []
        search += list((BASE / "models" / "human_detector_real").rglob("best.pt"))
        search += list((BASE / "models" / "rgb_detector").rglob("best.pt"))

        for p in search:
            if p and Path(p).exists():
                try:
                    from ultralytics import YOLO
                    self.human_model = YOLO(str(p))
                    print(f"[CombinedInferenceEngine] Human model: {p}")
                    return
                except Exception as e:
                    print(f"[CombinedInferenceEngine] Human model load failed: {e}")

        print("[CombinedInferenceEngine] No human model found")

    def _load_fire_model(self, path=None):
        default_path = BASE / "models" / "fire_classifier" / "fire_classifier_best.pt"
        p = Path(path) if path else default_path

        if not p.exists():
            print(f"[CombinedInferenceEngine] No fire classifier found at {p}")
            return

        try:
            self.fire_model = build_model()
            self.fire_model.load_state_dict(
                torch.load(str(p), map_location=DEVICE)
            )
            self.fire_model.to(DEVICE)
            self.fire_model.eval()
            _, self.fire_transform = build_transforms()
            print(f"[CombinedInferenceEngine] Fire classifier: {p}")
        except Exception as e:
            print(f"[CombinedInferenceEngine] Fire classifier load failed: {e}")
            self.fire_model = None

    def run(self, frame, thermal_frame=None):
        """Same signature as the old InferenceEngine."""
        if self.mock_mode:
            return self._mock_detections(frame)

        detections = []
        H, W = frame.shape[:2]

        # Human detection (YOLO, localised boxes)
        if self.human_model is not None:
            try:
                results = self.human_model.predict(
                    source=frame, conf=0.25, iou=0.45, verbose=False
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = [x1, y1, x2, y2]

                        thermal_intensity = self._get_thermal_intensity(
                            thermal_frame, bbox
                        )
                        priority = compute_priority(
                            0, confidence, thermal_intensity, bbox, W, H
                        )

                        detections.append(Detection(
                            class_id=0,
                            class_name="Human",
                            confidence=confidence,
                            bbox=bbox,
                            priority=priority,
                            priority_label=["", "CRITICAL", "HIGH", "MEDIUM"][priority],
                            color=(0, 255, 100),
                            location_hint=get_location_hint(bbox, W, H),
                            thermal_intensity=thermal_intensity,
                        ))
            except Exception as e:
                print(f"[CombinedInferenceEngine] Human inference error: {e}")

        # Fire detection (classifier, whole-frame flag)
        if self.fire_model is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img   = Image.fromarray(rgb_frame)
                img_t     = self.fire_transform(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logit = self.fire_model(img_t).squeeze()
                    fire_prob = torch.sigmoid(logit).item()

                if fire_prob > 0.5:
                    detections.append(Detection(
                        class_id=2,
                        class_name="Fire (whole-frame)",
                        confidence=round(fire_prob, 3),
                        bbox=[0, 0, W, H],
                        priority=2,
                        priority_label="HIGH",
                        color=(0, 50, 255),
                        location_hint="Scene-wide",
                        thermal_intensity=200.0,
                    ))
            except Exception as e:
                print(f"[CombinedInferenceEngine] Fire inference error: {e}")

        detections.sort(key=lambda d: (d.priority, -d.confidence))
        return detections

    def _get_thermal_intensity(self, thermal_frame, bbox):
        if thermal_frame is None:
            return 128.0
        try:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            H, W = thermal_frame.shape[:2]
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return 128.0
            region = thermal_frame[y1:y2, x1:x2]
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            return float(np.mean(region))
        except Exception:
            return 128.0

    def _mock_detections(self, frame):
        return []


if __name__ == "__main__":
    print("Testing CombinedInferenceEngine...\n")
    engine = CombinedInferenceEngine()

    test_dir = BASE / "dataset_real" / "test" / "images"
    test_imgs = list(test_dir.glob("*.*"))[:5] if test_dir.exists() else []

    if not test_imgs:
        print("No test images found at", test_dir)
    else:
        for img_path in test_imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            dets = engine.run(frame)
            print(f"\n{img_path.name}: {len(dets)} detections")
            for d in dets:
                print(f"  [{d.priority_label}] {d.class_name} "
                      f"conf={d.confidence:.2f} loc={d.location_hint}")