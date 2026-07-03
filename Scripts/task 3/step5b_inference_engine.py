import cv2
import numpy as np
import random
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import find_best_model, MODELS_DIR

@dataclass
class Detection:
    class_id:   int          # 0=human, 1=distress_signal, 2=fire
    class_name: str          # readable label
    confidence: float        # 0.0–1.0 model confidence
    bbox:       List[int]    # [x1, y1, x2, y2] pixel coords
    priority:   int          # 1=critical, 2=high, 3=medium
    priority_label: str      # "CRITICAL" / "HIGH" / "MEDIUM"
    color:      tuple        # BGR color for drawing
    location_hint: str       # "NW quadrant", "Centre", etc.
    thermal_intensity: float # avg thermal value in bbox (0–255)


# ── Class Definitions ─────────────────────────────────────
CLASS_INFO = {
    0: {
        "name":  "Human",
        "color": (0, 255, 100),    # green
        "base_priority": 1,
    },
    1: {
        "name":  "Distress Signal",
        "color": (0, 140, 255),    # orange
        "base_priority": 1,
    },
    2: {
        "name":  "Fire",
        "color": (0, 50, 255),     # red
        "base_priority": 2,
    },
}

PRIORITY_LABELS = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM"}
PRIORITY_COLORS = {
    "CRITICAL": (0, 50,  255),   # red
    "HIGH":     (0, 165, 255),   # orange
    "MEDIUM":   (0, 255, 255),   # yellow
}


# ── Location Helper ───────────────────────────────────────

def get_location_hint(bbox, frame_w=640, frame_h=480):
    """
    Converts pixel coordinates to a human-readable
    location hint like 'NW quadrant' or 'Centre'.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    h_pos = "W" if cx < frame_w * 0.33 else \
            "E" if cx > frame_w * 0.67 else "Centre"
    v_pos = "N" if cy < frame_h * 0.33 else \
            "S" if cy > frame_h * 0.67 else ""

    if h_pos == "Centre" and v_pos == "":
        return "Centre"
    return f"{v_pos}{h_pos}".strip()


def compute_priority(class_id, confidence, thermal_intensity,
                     bbox, frame_w=640, frame_h=480):
    """
    Assigns a priority score (1=most urgent) based on:
    - Class type (human/distress = highest)
    - Detection confidence
    - Thermal intensity (hotter = more urgent for humans)
    - Isolation (further from centre = harder to reach)
    """
    base = CLASS_INFO.get(class_id, {}).get("base_priority", 3)

    # Upgrade priority if high confidence
    if confidence > 0.85:
        base = max(1, base - 1)

    # For humans: high thermal intensity = alive & warm = priority 1
    if class_id == 0:
        if thermal_intensity > 180:
            base = 1
        elif thermal_intensity < 80:
            base = min(3, base + 1)  # possibly deceased, lower priority

    return base


# ── Inference Engine ───────────────────────────────────────

class InferenceEngine:
    def __init__(self, model_path=None):
        self.model      = None
        self.mock_mode  = False
        self.model_path = model_path
        self._mock_tick = 0
        self._load_model()

    def _load_model(self):
        """
        Tries to load the trained YOLOv8 model.
        Falls back to MOCK mode if not found yet.
        """
        # Search common locations for best.pt
        search_paths = []
        if self.model_path:
            search_paths.append(Path(self.model_path))

        # Auto-search in models directory using config
        if MODELS_DIR.exists():
            # Prefer real human detector, fall back to rgb_detector
            best_real = find_best_model("human_detector_real")
            best_rgb  = find_best_model("rgb_detector")
            for pt in [best_real, best_rgb]:
                if pt and pt not in search_paths:
                    search_paths.append(pt)

        for path in search_paths:
            if path.exists():
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(str(path))
                    print(f"[InferenceEngine] Model loaded: {path}")
                    return
                except Exception as e:
                    print(f"[InferenceEngine] Failed to load {path}: {e}")

        # No model found — use mock mode
        print("[InferenceEngine] No trained model found — running in MOCK mode")
        print("  (Train with step4 to enable real detections)")
        self.mock_mode = True

    def run(self, frame, thermal_frame=None):
        if self.mock_mode:
            return self._mock_detections(frame, thermal_frame)
        else:
            return self._real_detections(frame, thermal_frame)

    # ── Real YOLOv8 Inference ─────────────────────────────

    def _real_detections(self, frame, thermal_frame=None):
        """Runs actual YOLOv8 inference on the frame."""
        H, W = frame.shape[:2]
        detections = []

        try:
            results = self.model.predict(
                source  = frame,
                conf    = 0.25,
                iou     = 0.45,
                verbose = False,
            )

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    class_id   = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = [x1, y1, x2, y2]

                    # Get thermal intensity in this bbox region
                    thermal_intensity = self._get_thermal_intensity(
                        thermal_frame, bbox
                    )

                    priority = compute_priority(
                        class_id, confidence,
                        thermal_intensity, bbox, W, H
                    )

                    info = CLASS_INFO.get(class_id, {
                        "name": f"Class{class_id}",
                        "color": (255, 255, 255),
                        "base_priority": 3
                    })

                    detections.append(Detection(
                        class_id          = class_id,
                        class_name        = info["name"],
                        confidence        = confidence,
                        bbox              = bbox,
                        priority          = priority,
                        priority_label    = PRIORITY_LABELS[priority],
                        color             = info["color"],
                        location_hint     = get_location_hint(bbox, W, H),
                        thermal_intensity = thermal_intensity,
                    ))

        except Exception as e:
            print(f"[InferenceEngine] Inference error: {e}")

        # Sort by priority (1 first), then confidence
        detections.sort(key=lambda d: (d.priority, -d.confidence))
        return detections

    def _get_thermal_intensity(self, thermal_frame, bbox):
        """Gets average thermal intensity in the detection bounding box."""
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

    # ── Mock Detections (no model needed) ─────────────────
    # Generates believable fake detections that change over
    # time so the HMI looks alive during development.

    def _mock_detections(self, frame, thermal_frame=None):
        """
        Generates realistic fake detections.
        Uses sine waves so detections appear/disappear
        naturally rather than flickering randomly.
        """
        H, W = frame.shape[:2]
        detections = []
        self._mock_tick += 1
        t = self._mock_tick

        # ── Scenario bank ─────────────────────────────────
        # Each scenario is active for ~30 ticks then cycles
        scenario = (t // 30) % 4

        scenarios = [
            # Scenario 0: 2 humans, 1 distress signal
            [
                (0, 0.91, [80,  120, 115, 175],  185.0),
                (0, 0.78, [400, 200, 435, 260],  172.0),
                (1, 0.88, [300, 80,  318,  98],  240.0),
            ],
            # Scenario 1: 3 humans, 1 fire
            [
                (0, 0.93, [150, 150, 188, 210],  190.0),
                (0, 0.82, [450, 300, 485, 360],  178.0),
                (0, 0.67, [250, 350, 278, 400],  155.0),
                (2, 0.95, [520, 100, 580, 155],  248.0),
            ],
            # Scenario 2: 1 critical human + distress
            [
                (0, 0.96, [320, 240, 365, 310],  220.0),
                (1, 0.90, [100, 80,  120, 100],  250.0),
            ],
            # Scenario 3: complex multi-target
            [
                (0, 0.88, [200, 180, 238, 240],  195.0),
                (0, 0.74, [500, 250, 532, 305],  168.0),
                (1, 0.85, [420, 80,  440, 100],  245.0),
                (2, 0.91, [60,  380, 120, 440],  250.0),
            ],
        ]

        active = scenarios[scenario]

        # Add slight jitter per tick so boxes aren't perfectly static
        for class_id, conf_base, bbox_base, therm in active:
            jitter = 3
            bbox = [
                bbox_base[0] + random.randint(-jitter, jitter),
                bbox_base[1] + random.randint(-jitter, jitter),
                bbox_base[2] + random.randint(-jitter, jitter),
                bbox_base[3] + random.randint(-jitter, jitter),
            ]
            # Clamp to frame
            bbox = [
                max(0, min(W, bbox[0])),
                max(0, min(H, bbox[1])),
                max(0, min(W, bbox[2])),
                max(0, min(H, bbox[3])),
            ]

            # Confidence varies slightly
            confidence = max(0.4, min(0.99,
                conf_base + math.sin(t * 0.3) * 0.03
                          + random.gauss(0, 0.01)
            ))

            priority = compute_priority(
                class_id, confidence, therm, bbox, W, H
            )
            info = CLASS_INFO.get(class_id, {
                "name": "Unknown", "color": (255,255,255),
                "base_priority": 3
            })

            detections.append(Detection(
                class_id          = class_id,
                class_name        = info["name"],
                confidence        = round(confidence, 2),
                bbox              = bbox,
                priority          = priority,
                priority_label    = PRIORITY_LABELS[priority],
                color             = info["color"],
                location_hint     = get_location_hint(bbox, W, H),
                thermal_intensity = therm + random.gauss(0, 2),
            ))

        detections.sort(key=lambda d: (d.priority, -d.confidence))
        return detections


# ── Draw Detections on Frame ───────────────────────────────
# Used by the HMI to render detection overlays on the feed.

def draw_detections(frame, detections):
    """
    Draws bounding boxes, labels, and priority badges
    on the frame. Returns annotated frame.
    """
    annotated = frame.copy()
    H, W = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = PRIORITY_COLORS.get(det.priority_label, (255, 255, 255))

        # ── Bounding box ──────────────────────────────────
        thickness = 3 if det.priority == 1 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # ── Corner accents (tactical HUD style) ──────────
        corner_len = 10
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(annotated, (cx, cy),
                     (cx + dx*corner_len, cy), color, 2)
            cv2.line(annotated, (cx, cy),
                     (cx, cy + dy*corner_len), color, 2)

        # ── Label pill ────────────────────────────────────
        label = (f"[P{det.priority}] {det.class_name} "
                 f"{det.confidence:.0%}")
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        lx = max(0, x1)
        ly = max(th + 8, y1 - 4)
        # Background pill
        cv2.rectangle(annotated,
                      (lx, ly - th - 6),
                      (lx + tw + 8, ly),
                      color, -1)
        # Text
        cv2.putText(annotated, label,
                    (lx + 4, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)

        # ── Location hint ─────────────────────────────────
        cv2.putText(annotated, det.location_hint,
                    (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color, 1, cv2.LINE_AA)

    # ── Detection count overlay ───────────────────────────
    humans    = sum(1 for d in detections if d.class_id == 0)
    distress  = sum(1 for d in detections if d.class_id == 1)
    fires     = sum(1 for d in detections if d.class_id == 2)
    summary   = f"Humans:{humans}  Distress:{distress}  Fire:{fires}"
    cv2.putText(annotated, summary,
                (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    engine = InferenceEngine()

    # Test 1 — blank frame (expect 0 detections)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dets  = engine.run(dummy)
    print(f"Blank frame: {len(dets)} detections (expected 0)")

    # Test 2 — load a real image from your dataset
    from pathlib import Path
    from config import SAR_DATA
    test_dir = SAR_DATA / "dataset_real" / "test" / "images"
    if not test_dir.exists():
        test_dir = SAR_DATA / "dataset" / "test" / "images"
    test_imgs = list(test_dir.glob("*.jpg"))[:3]

    if test_imgs:
        for img_path in test_imgs:
            img  = cv2.imread(str(img_path))
            dets = engine.run(img)
            print(f"\n{img_path.name}: {len(dets)} detections")
            for d in dets:
                print(f"  [{d.priority_label}] {d.class_name} "
                      f"conf={d.confidence:.2f} "
                      f"loc={d.location_hint}")
    else:
        print("No test images found")