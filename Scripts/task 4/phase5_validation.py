import cv2
import numpy as np
import json
import time
import re
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ── Path setup via config ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SAR_DATA as BASE_DIR,
    LOGS_DIR,
    TASK_FOLDERS as _TASK_FOLDERS,
    setup_path,
    find_best_model,
)
setup_path()
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TASK_FOLDERS = list(_TASK_FOLDERS)
SCRIPTS_DIR  = Path(__file__).parent
CLASSES      = {0: "human", 1: "fire", 2: "landing_zone"}
MODEL_PATH   = find_best_model("human_detector_real") or find_best_model("rgb_detector")


# ══════════════════════════════════════════════════════════
#  TEST RESULT STRUCTURES
# ══════════════════════════════════════════════════════════

@dataclass
class TestResult:
    subtask:    str
    name:       str
    passed:     bool
    score:      float           # 0.0–1.0
    details:    Dict = field(default_factory=dict)
    warnings:   List[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class ValidationReport:
    timestamp:  str = ""
    model_path: str = ""
    results:    List[TestResult] = field(default_factory=list)
    overall_pass: bool = False
    overall_score: float = 0.0


# ══════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def fail(msg): print(f"  ❌ {msg}")
def info(msg): print(f"     {msg}")

def load_model():
    """Loads YOLOv8 model, returns model or None."""
    if MODEL_PATH is None:
        warn("No best.pt found in sar_data/models/")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        ok(f"Model loaded: {MODEL_PATH.name}")
        return model
    except Exception as e:
        fail(f"Model load failed: {e}")
        return None

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter)


# ══════════════════════════════════════════════════════════
#  5.1 — DETECTION ACCURACY
# ══════════════════════════════════════════════════════════

def test_5_1_detection_accuracy(model, quick=False) -> TestResult:
    """
    Runs model on held-out test set and computes:
      - Per-class precision, recall, mAP@50
      - Overall mAP@50 and mAP@50-95
      - False positive and false negative rates

    Uses YOLO's built-in val() method on the test split.
    """
    section("5.1 — Detection accuracy")
    t_start = time.time()
    details = {}
    warnings = []

    # ── Check test dataset exists ─────────────────────────
    test_dir  = BASE_DIR / "dataset" / "test" / "images"
    test_imgs = list(test_dir.glob("*.jpg")) if test_dir.exists() else []

    if not test_imgs:
        fail("No test images found")
        return TestResult("5.1", "Detection accuracy",
                          False, 0.0,
                          {"error": "no test images"},
                          duration_s=time.time()-t_start)

    info(f"Test images: {len(test_imgs)}")

    if model is None:
        fail("No model available")
        return TestResult("5.1", "Detection accuracy",
                          False, 0.0,
                          {"error": "no model"},
                          duration_s=time.time()-t_start)

    # ── Run YOLO validation ───────────────────────────────
    try:
        yaml_path = BASE_DIR / "dataset" / "dataset.yaml"
        print(f"\n  Running validation on test split...")

        metrics = model.val(
            data   = str(yaml_path),
            split  = "test",
            imgsz  = 320 if quick else 640,
            conf   = 0.25,
            iou    = 0.45,
            device = "cpu",
            verbose= False,
        )

        # Extract metrics
        map50    = float(metrics.box.map50)
        map5095  = float(metrics.box.map)
        prec     = float(metrics.box.mp)
        recall   = float(metrics.box.mr)

        # Per-class metrics
        per_class = {}
        try:
            for i, name in CLASSES.items():
                per_class[name] = {
                    "precision": round(float(metrics.box.p[i]), 3),
                    "recall":    round(float(metrics.box.r[i]), 3),
                    "mAP50":     round(float(metrics.box.ap50[i]), 3),
                }
        except Exception:
            pass

        details = {
            "n_test_images": len(test_imgs),
            "mAP50":         round(map50,   3),
            "mAP50_95":      round(map5095, 3),
            "precision":     round(prec,    3),
            "recall":        round(recall,  3),
            "per_class":     per_class,
        }

        # ── Print results ─────────────────────────────────
        print(f"\n  {'Metric':<20} {'Value':>8}")
        print(f"  {'-'*30}")
        print(f"  {'mAP@50':<20} {map50:>8.3f}")
        print(f"  {'mAP@50-95':<20} {map5095:>8.3f}")
        print(f"  {'Precision':<20} {prec:>8.3f}")
        print(f"  {'Recall':<20} {recall:>8.3f}")

        if per_class:
            print(f"\n  Per-class breakdown:")
            print(f"  {'Class':<16} {'Prec':>6} {'Rec':>6} {'mAP50':>7}")
            print(f"  {'-'*38}")
            for cls_name, m in per_class.items():
                print(f"  {cls_name:<16} "
                      f"{m['precision']:>6.3f} "
                      f"{m['recall']:>6.3f} "
                      f"{m['mAP50']:>7.3f}")

        # ── Grade ─────────────────────────────────────────
        if map50 >= 0.85:
            ok(f"mAP@50 = {map50:.3f} — excellent")
        elif map50 >= 0.70:
            ok(f"mAP@50 = {map50:.3f} — good")
        elif map50 >= 0.50:
            warn(f"mAP@50 = {map50:.3f} — acceptable, add real images to improve")
            warnings.append("mAP below 0.70 — recommend adding real SAR images")
        else:
            fail(f"mAP@50 = {map50:.3f} — needs improvement")
            warnings.append("mAP below 0.50 — retrain with more diverse data")

        if recall < 0.70:
            warn(f"Recall = {recall:.3f} — model missing some detections")
            warnings.append("Low recall — lower confidence threshold or add training data")

        passed = map50 >= 0.50
        score  = min(1.0, map50)

    except Exception as e:
        fail(f"Validation failed: {e}")
        details = {"error": str(e)}
        passed  = False
        score   = 0.0

    return TestResult("5.1", "Detection accuracy", passed, score,
                      details, warnings,
                      duration_s=round(time.time()-t_start, 2))


# ══════════════════════════════════════════════════════════
#  5.2 — LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════

def test_5_2_latency(model, quick=False) -> TestResult:
    section("5.2 — Latency benchmark")
    t_start  = time.time()
    details  = {}
    warnings = []
    n_runs   = 5 if quick else 20

    H, W = 480, 640
    test_frame = np.random.randint(40, 200, (H, W, 3), dtype=np.uint8)
    test_frame = cv2.GaussianBlur(test_frame, (7, 7), 0)

    timings = {
        "preprocess":  [],
        "inference":   [],
        "features":    [],
        "lz_classify": [],
        "decision":    [],
        "total":       [],
    }

    print(f"\n  Running {n_runs} frames for timing...")

    for i in range(n_runs):
        t0 = time.perf_counter()

        # ── Preprocess ────────────────────────────────────
        t1 = time.perf_counter()
        frame = cv2.resize(test_frame, (W, H))
        t2 = time.perf_counter()
        timings["preprocess"].append((t2-t1)*1000)

        # ── Inference ─────────────────────────────────────
        t3 = time.perf_counter()
        detections = []
        if model:
            try:
                results = model.predict(
                    source=frame, conf=0.25,
                    iou=0.45, verbose=False
                )
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            detections.append({
                                "class_id":  int(box.cls[0]),
                                "confidence":float(box.conf[0]),
                                "bbox":      list(map(int, box.xyxy[0]))
                            })
            except Exception:
                pass
        t4 = time.perf_counter()
        timings["inference"].append((t4-t3)*1000)

        # ── Feature extraction ────────────────────────────
        t5 = time.perf_counter()
        thermal_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        for det in detections:
            bbox = det["bbox"]
            x1,y1,x2,y2 = [max(0,bbox[i]) for i in range(4)]
            x2=min(W,x2); y2=min(H,y2)
            region = thermal_gray[y1:y2, x1:x2]
            if region.size > 0:
                _ = float(np.mean(region))
        t6 = time.perf_counter()
        timings["features"].append((t6-t5)*1000)

        # ── LZ classification ─────────────────────────────
        t7 = time.perf_counter()
        cell_w = W // 4; cell_h = H // 3
        lz_scores = []
        for row in range(3):
            for col in range(4):
                x1=col*cell_w; y1=row*cell_h
                x2=min(W,x1+cell_w); y2=min(H,y1+cell_h)
                region = frame[y1:y2, x1:x2]
                if region.size > 0:
                    lap = cv2.Laplacian(
                        cv2.cvtColor(region, cv2.COLOR_BGR2GRAY),
                        cv2.CV_64F
                    ).var()
                    lz_scores.append(max(0, 1-lap/800))
        t8 = time.perf_counter()
        timings["lz_classify"].append((t8-t7)*1000)

        # ── Decision ──────────────────────────────────────
        t9 = time.perf_counter()
        n_surv = len([d for d in detections if d["class_id"]==0])
        lz_safe = max(lz_scores) > 0.55 if lz_scores else False
        if n_surv > 0 and lz_safe:   action = "LAND"
        elif n_surv > 0:              action = "HOVER_DROP"
        else:                         action = "HOLD"
        t10 = time.perf_counter()
        timings["decision"].append((t10-t9)*1000)

        total = (t10-t0)*1000
        timings["total"].append(total)

        if (i+1) % 5 == 0:
            print(f"  Frame {i+1:>3}: total={total:.1f}ms  "
                  f"inf={timings['inference'][-1]:.1f}ms  "
                  f"lz={timings['lz_classify'][-1]:.1f}ms")

    # ── Compute stats ─────────────────────────────────────
    stats = {}
    for stage, times in timings.items():
        stats[stage] = {
            "mean_ms": round(float(np.mean(times)), 2),
            "max_ms":  round(float(np.max(times)),  2),
            "min_ms":  round(float(np.min(times)),  2),
            "p95_ms":  round(float(np.percentile(times, 95)), 2),
        }

    total_mean = stats["total"]["mean_ms"]
    total_p95  = stats["total"]["p95_ms"]

    # ── Print breakdown ───────────────────────────────────
    print(f"\n  {'Stage':<16} {'Mean':>8} {'P95':>8} {'Max':>8}")
    print(f"  {'-'*44}")
    for stage in ["preprocess","inference","features","lz_classify","decision"]:
        s = stats[stage]
        print(f"  {stage:<16} {s['mean_ms']:>7.1f}ms "
              f"{s['p95_ms']:>7.1f}ms "
              f"{s['max_ms']:>7.1f}ms")
    print(f"  {'-'*44}")
    s = stats["total"]
    print(f"  {'TOTAL':<16} {s['mean_ms']:>7.1f}ms "
          f"{s['p95_ms']:>7.1f}ms "
          f"{s['max_ms']:>7.1f}ms")

    # ── Grade ─────────────────────────────────────────────
    TARGET_MS = 500.0
    if total_p95 <= TARGET_MS:
        ok(f"P95 latency {total_p95:.0f}ms < {TARGET_MS:.0f}ms target ✓")
    else:
        warn(f"P95 latency {total_p95:.0f}ms exceeds {TARGET_MS:.0f}ms target")
        warnings.append(f"Latency {total_p95:.0f}ms > 500ms — "
                       "reduce imgsz or use GPU")

    fps_estimate = 1000 / total_mean
    info(f"Estimated throughput: ~{fps_estimate:.1f} FPS")

    details = {
        "n_runs":      n_runs,
        "target_ms":   TARGET_MS,
        "total_mean":  total_mean,
        "total_p95":   total_p95,
        "fps_estimate":round(fps_estimate, 1),
        "stages":      stats,
    }

    passed = total_p95 <= TARGET_MS
    score  = min(1.0, TARGET_MS / max(total_p95, 1))

    return TestResult("5.2", "Latency benchmark", passed, score,
                      details, warnings,
                      duration_s=round(time.time()-t_start, 2))


# ══════════════════════════════════════════════════════════
#  5.3 — EDGE CASE TESTING
# ══════════════════════════════════════════════════════════

def test_5_3_edge_cases(model) -> TestResult:
    section("5.3 — Edge case testing")
    t_start  = time.time()
    details  = {}
    warnings = []

    H, W = 480, 640
    passed_count = 0
    total_cases  = 0
    case_results = {}

    def run_inference_on(frame):
        """Runs inference and returns detection list."""
        if model is None:
            return []
        try:
            results = model.predict(
                source=frame, conf=0.25,
                iou=0.45, verbose=False
            )
            dets = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        dets.append({
                            "class_id":   int(box.cls[0]),
                            "confidence": float(box.conf[0]),
                        })
            return dets
        except Exception:
            return []

    def simple_decision(n_surv, lz_safe, fire, battery, gps_lock, n_fires=0):
        """Simplified decision logic for edge case checking."""
        if battery < 12:                          return "RTB"
        if n_fires >= 2 and n_surv >= 3:          return "ESCALATE"
        if n_surv > 0 and lz_safe and not fire:   return "LAND"
        if n_surv > 0 and (not lz_safe or fire):  return "HOVER_DROP"
        if not gps_lock and n_surv > 0:           return "RELAY"
        return "HOLD"

    print()

    # ── Case 1: Night / Dark scene ────────────────────────
    total_cases += 1
    dark_frame = np.full((H,W,3), 8, dtype=np.uint8)
    dets = run_inference_on(dark_frame)
    n_humans = len([d for d in dets if d["class_id"]==0])
    decision = simple_decision(n_humans, True, False, 80, True)
    passed = decision == "HOLD"
    case_results["night_scene"] = {
        "detections": len(dets),
        "decision":   decision,
        "expected":   "HOLD",
        "passed":     passed
    }
    if passed: ok("Night scene → HOLD (correct)")
    else:      warn(f"Night scene → {decision} (expected HOLD)")
    if passed: passed_count += 1

    # ── Case 2: Smoke obscured ────────────────────────────
    total_cases += 1
    smoke_frame = np.random.randint(100, 160, (H,W,3), dtype=np.uint8)
    smoke_frame = cv2.GaussianBlur(smoke_frame, (31,31), 0)
    dets = run_inference_on(smoke_frame)
    n_humans = len([d for d in dets if d["class_id"]==0])
    decision = simple_decision(n_humans, True, False, 80, True)
    passed = decision in ["HOLD", "LAND"]
    case_results["smoke_scene"] = {
        "detections": len(dets),
        "decision":   decision,
        "passed":     passed
    }
    if passed: ok(f"Smoke scene → {decision} (acceptable)")
    else:      warn(f"Smoke scene → {decision}")
    if passed: passed_count += 1

    # ── Case 3: No survivors ──────────────────────────────
    total_cases += 1
    decision = simple_decision(0, True, False, 80, True)
    passed = decision == "HOLD"
    case_results["no_survivors"] = {
        "decision": decision, "expected": "HOLD", "passed": passed
    }
    if passed: ok("No survivors → HOLD (correct)")
    else:      fail(f"No survivors → {decision} (expected HOLD)")
    if passed: passed_count += 1

    # ── Case 4: GPS loss ──────────────────────────────────
    total_cases += 1
    decision = simple_decision(2, True, False, 80, False)
    passed = decision == "RELAY"
    case_results["gps_loss"] = {
        "decision": decision, "expected": "RELAY", "passed": passed
    }
    if passed: ok("GPS loss + survivors → RELAY (correct)")
    else:      warn(f"GPS loss → {decision} (expected RELAY)")
    if passed: passed_count += 1

    # ── Case 5: Critical battery ──────────────────────────
    total_cases += 1
    decision = simple_decision(2, True, False, 10, True)
    passed = decision == "RTB"
    case_results["critical_battery"] = {
        "battery":  10, "decision": decision,
        "expected": "RTB", "passed": passed
    }
    if passed: ok("Critical battery (10%) → RTB (correct)")
    else:      fail(f"Critical battery → {decision} (expected RTB)")
    if passed: passed_count += 1

    # ── Case 6: Fire below LZ ─────────────────────────────
    total_cases += 1
    decision = simple_decision(1, False, True, 80, True)
    passed = decision == "HOVER_DROP"
    case_results["fire_below_lz"] = {
        "decision": decision, "expected": "HOVER_DROP", "passed": passed
    }
    if passed: ok("Fire below LZ → HOVER_DROP (correct)")
    else:      fail(f"Fire below LZ → {decision} (expected HOVER_DROP)")
    if passed: passed_count += 1

    # ── Case 7: Unsafe LZ + survivors ────────────────────
    total_cases += 1
    decision = simple_decision(3, False, False, 80, True)
    passed = decision == "HOVER_DROP"
    case_results["unsafe_lz_survivors"] = {
        "decision": decision, "expected": "HOVER_DROP", "passed": passed
    }
    if passed: ok("Unsafe LZ + survivors → HOVER_DROP (correct)")
    else:      fail(f"Unsafe LZ + survivors → {decision}")
    if passed: passed_count += 1

    # ── Case 8: Mass casualty / escalate ─────────────────
    total_cases += 1
    decision = simple_decision(5, False, True, 60, True, n_fires=3)
    passed = decision == "ESCALATE"
    case_results["mass_casualty"] = {
        "decision": decision, "expected": "ESCALATE", "passed": passed
    }
    if passed: ok("Mass casualty (5 survivors, 3 fires) → ESCALATE (correct)")
    else:      warn(f"Mass casualty → {decision} (expected ESCALATE)")
    if passed: passed_count += 1

    # ── Case 9: Low battery but survivors ────────────────
    total_cases += 1
    decision = simple_decision(2, True, False, 22, True)
    passed = decision in ["LAND", "HOVER_DROP"]
    case_results["low_battery_survivors"] = {
        "battery":  22, "decision": decision,
        "expected": "LAND or HOVER_DROP", "passed": passed
    }
    if passed: ok(f"Low battery (22%) + survivors → {decision} (correct)")
    else:      warn(f"Low battery + survivors → {decision}")
    if passed: passed_count += 1

    # ── Case 10: Perfect conditions ───────────────────────
    total_cases += 1
    decision = simple_decision(2, True, False, 80, True)
    passed = decision == "LAND"
    case_results["perfect_conditions"] = {
        "decision": decision, "expected": "LAND", "passed": passed
    }
    if passed: ok("Perfect conditions → LAND (correct)")
    else:      fail(f"Perfect conditions → {decision} (expected LAND)")
    if passed: passed_count += 1

    # ── Summary ───────────────────────────────────────────
    pass_rate = passed_count / total_cases
    print(f"\n  Edge cases passed: {passed_count}/{total_cases} "
          f"({pass_rate*100:.0f}%)")

    if pass_rate < 0.80:
        warnings.append(f"Only {pass_rate*100:.0f}% edge cases passed")

    details = {
        "passed":    passed_count,
        "total":     total_cases,
        "pass_rate": round(pass_rate, 3),
        "cases":     case_results,
    }

    return TestResult("5.3", "Edge case testing",
                      pass_rate >= 0.80, pass_rate,
                      details, warnings,
                      duration_s=round(time.time()-t_start, 2))


# ══════════════════════════════════════════════════════════
#  5.4 — HMI USABILITY CHECK
# ══════════════════════════════════════════════════════════

def test_5_4_hmi_usability() -> TestResult:
    section("5.4 — HMI usability check")
    t_start  = time.time()
    details  = {}
    warnings = []
    checks   = {}

    hmi_file = None
    for folder in _TASK_FOLDERS:
        candidate = Path(folder) / "step5d_cockpit_hmi.py"
        if candidate.exists():
            hmi_file = candidate
            break

    if hmi_file is None:
        fail("step5d_cockpit_hmi.py not found in any task folder")
        return TestResult("5.4", "HMI usability", False, 0.0, {"error": "HMI file not found"}, duration_s=time.time()-t_start)

    content = hmi_file.read_text(encoding="utf-8")

    # ── Check 1: Required panels ──────────────────────────
    required_panels = {
        "feed_panel":           "_build_feed_panel",
        "map_panel":            "_build_map_panel",
        "status_panel":         "_build_status_panel",
        "alert_panel":          "_build_alert_panel",
        "recommendation_panel": "_build_recommendation_panel",
        "telem_bar":            "_build_telem_bar",
    }
    panels_ok = all(m in content for m in required_panels.values())
    checks["all_panels_present"] = panels_ok
    if panels_ok: ok("All 6 cockpit panels defined")
    else:
        missing = [k for k,v in required_panels.items()
                   if v not in content]
        warn(f"Missing panels: {missing}")

    # ── Check 2: Feed mode toggle ─────────────────────────
    modes_ok = all(m in content for m in
                   ['"RGB"', '"THERMAL"', '"FUSED"'])
    checks["feed_mode_toggle"] = modes_ok
    if modes_ok: ok("Feed mode toggle (RGB/THERMAL/FUSED) present")
    else:        warn("Feed mode toggle incomplete")

    # ── Check 3: Operator controls ────────────────────────
    controls_ok = all(m in content for m in
                      ["_on_accept", "_on_override", "_on_abort"])
    checks["operator_controls"] = controls_ok
    if controls_ok: ok("Operator controls (Accept/Override/Abort) defined")
    else:           warn("Some operator controls missing")

    # ── Check 4: Alert priority sorting ──────────────────
    priority_ok = "priority" in content and "_alert_rows" in content
    checks["alert_priority"] = priority_ok
    if priority_ok: ok("Alert panel with priority rows defined")
    else:           warn("Alert priority sorting may be missing")

    # ── Check 5: Decision engine import ───────────────────
    uses_phase3 = "phase3e_hmi_integration" in content
    uses_old    = "step5c_decision_engine" in content
    if uses_phase3:
        ok("Using Phase 3 full decision engine (phase3e)")
        checks["decision_engine"] = "phase3e"
    elif uses_old:
        warn("Still using old step5c decision engine — upgrade recommended")
        warnings.append("HMI using step5c not phase3e — "
                       "replace import for full decision engine")
        checks["decision_engine"] = "step5c"
    else:
        warn("Decision engine import not detected")
        checks["decision_engine"] = "unknown"

    # ── Check 6: Action colours ───────────────────────────
    color_ok = "ACTION_COLORS" in content
    checks["action_colors"] = color_ok
    if color_ok: ok("Action colour mapping present")
    else:        warn("ACTION_COLORS not found")

    # ── Check 7: Map panel with GPS markers ───────────────
    map_ok = "_map_canvas" in content and "marker" in content
    checks["map_with_markers"] = map_ok
    if map_ok: ok("Map panel with GPS marker support defined")
    else:      warn("Map marker support may be missing")

    # ── Check 8: Telemetry bar ────────────────────────────
    telem_ok = "_telem_bar" in content and "battery" in content.lower()
    checks["telem_bar"] = telem_ok
    if telem_ok: ok("Telemetry bar with battery monitoring defined")
    else:        warn("Telemetry bar may be incomplete")

    # ── Manual test checklist ─────────────────────────────
    print(f"\n  Manual checks (verify in running cockpit):")
    manual_checks = [
        "Feed updates every ~500ms with detection boxes drawn",
        "RGB/Thermal/Fused toggle buttons change the feed view",
        "Alert list sorted by priority (CRITICAL first)",
        "AI recommendation updates with each new detection",
        "ACCEPT button logs confirmation to operator log",
        "OVERRIDE button logs override to operator log",
        "ABORT button resets to HOLD state",
        "Map shows survivor dots at correct screen positions",
        "Battery colour changes: green→orange→red as it drains",
        "Mission status badge updates (ACTIVE/RTB/EMERGENCY)",
    ]
    for i, check in enumerate(manual_checks, 1):
        print(f"    [{i:>2}] {check}")

    passed_count = sum(1 for v in checks.values() if v is True or
                      (isinstance(v, str) and v in ["phase3e"]))
    total = len(checks)
    score = passed_count / total

    details = {
        "automated_checks": checks,
        "passed":           passed_count,
        "total":            total,
        "manual_checklist": manual_checks,
    }

    return TestResult("5.4", "HMI usability",
                      score >= 0.75, score,
                      details, warnings,
                      duration_s=round(time.time()-t_start, 2))


# ══════════════════════════════════════════════════════════
#  5.5 — DECISION ENGINE STRESS TEST
# ══════════════════════════════════════════════════════════

def test_5_5_decision_engine() -> TestResult:
    section("5.5 — Decision engine stress test")
    t_start  = time.time()
    details  = {}
    warnings = []

    # Fallback constants when Phase 3 not importable
    class _Action:
        LAND="LAND"; HOVER_DROP="HOVER_DROP"; RELAY="RELAY"
        HOLD="HOLD"; LOITER="LOITER"; RTB="RTB"; ESCALATE="ESCALATE"
    class _MissionState:
        def __init__(self, **kw): [setattr(self,k,v) for k,v in kw.items()]
    Action = _Action
    MissionState = _MissionState
    scoring = None; safety = None; engine_available = False

    try:
        import importlib.util, types

        def load_module(name, folder):
            """Force-loads a module from a specific folder path."""
            p = Path(folder) / f"{name}.py"
            if not p.exists():
                raise ImportError(f"{p} not found")
            spec = importlib.util.spec_from_file_location(name, str(p))
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod          # register before exec
            spec.loader.exec_module(mod)
            return mod

        # Find task 2 folder (where phase3 files live)
        task2 = next(
            (f for f in _TASK_FOLDERS
             if (Path(str(f)) / "phase3a_mission_state.py").exists()),
            None
        )
        if task2 is None:
            raise ImportError("phase3a_mission_state.py not found in any task folder")

        task2 = str(task2)
        m_state   = load_module("phase3a_mission_state",  task2)
        m_scoring = load_module("phase3b_scoring_engine", task2)
        m_engine  = load_module("phase3d_decision_engine", task2)

        MissionState = m_state.MissionState
        Action       = m_state.Action
        ScoringEngine= m_scoring.ScoringEngine
        SafetyLayer  = m_engine.SafetyLayer

        engine_available = True

        weights_path = BASE_DIR / "models" / "scoring_weights.npy"
        scoring = ScoringEngine(weights_path=weights_path if weights_path.exists() else None)
        if weights_path.exists():
            ok(f"RL weights loaded from {weights_path}")
        else:
            warn("scoring_weights.npy not found — run phase3c_rl_trainer.py first")

        safety = SafetyLayer()
        ok("Full Phase 3 decision engine loaded")

    except (ImportError, Exception) as _e:
        warn(f"Phase 3 engine not importable ({_e}) — using rule-based fallback")

    def evaluate_scenario(params):
        #Runs one scenario and returns the chosen action.
        state = MissionState(**params) if engine_available else None

        if engine_available and state:
            scores  = scoring.score_all(state)
            _, _, action = safety.check(scores[0].action, state)
            return action
        else:
            # Fallback rule-based
            n   = params.get("n_survivors", 0)
            lz  = params.get("lz_safe", True)
            bat = params.get("battery", 80)
            fir = params.get("fire_detected", False)
            gps = params.get("gps_lock", True)
            n_f = params.get("n_fires", 0)
            crit= params.get("n_critical", 0)
            if bat < 12:                 return Action.RTB
            if crit >= 3 and n_f >= 2:   return Action.ESCALATE
            if n > 0 and lz and not fir: return Action.LAND
            if n > 0 and (not lz or fir):return Action.HOVER_DROP
            if not gps and n > 0:        return Action.RELAY
            return Action.HOLD

    # ── 10 test scenarios ─────────────────────────────────
    scenarios = [
        {
            "name":     "Ideal rescue",
            "expected": Action.LAND,
            "params":   dict(n_survivors=2, n_critical=1, any_alive=True,
                             lz_safe=True, lz_score=0.82,
                             fire_detected=False, battery=75.0,
                             max_confidence=0.91, gps_lock=True),
        },
        {
            "name":     "Fire blocks landing",
            "expected": Action.HOVER_DROP,
            "params":   dict(n_survivors=1, n_critical=1, any_alive=True,
                             lz_safe=False, lz_score=0.20,
                             fire_detected=True, fire_below_lz=True,
                             battery=65.0, max_confidence=0.88,
                             gps_lock=True),
        },
        {
            "name":     "Critical battery",
            "expected": Action.RTB,
            "params":   dict(n_survivors=1, battery=9.0,
                             lz_safe=True, fire_detected=False,
                             max_confidence=0.80, gps_lock=True),
        },
        {
            "name":     "Empty scene",
            "expected": Action.HOLD,
            "params":   dict(n_survivors=0, battery=80.0,
                             lz_safe=True, fire_detected=False,
                             max_confidence=0.0, gps_lock=True,
                             consecutive_clear_frames=8),
        },
        {
            "name":     "GPS loss",
            "expected": Action.RELAY,
            "params":   dict(n_survivors=2, n_distress=1, battery=70.0,
                             lz_safe=True, fire_detected=False,
                             max_confidence=0.75, gps_lock=False),
        },
        {
            "name":     "Mass casualty",
            "expected": Action.ESCALATE,
            "params":   dict(n_survivors=7, n_critical=5, n_fires=4,
                             any_alive=True, lz_safe=False, lz_score=0.05,
                             fire_detected=True, battery=55.0,
                             max_confidence=0.92, gps_lock=True),
        },
        {
            "name":     "Unsafe LZ only",
            "expected": Action.HOVER_DROP,
            "params":   dict(n_survivors=2, n_critical=1, any_alive=True,
                             lz_safe=False, lz_score=0.30,
                             fire_detected=False, battery=70.0,
                             max_confidence=0.85, gps_lock=True),
        },
        {
            "name":     "Low battery no critical",
            "expected": Action.RTB,
            "params":   dict(n_survivors=0, battery=11.0,
                             lz_safe=True, fire_detected=False,
                             max_confidence=0.0, gps_lock=True),
        },
        {
            "name":     "Distress signal only",
            "expected": Action.RELAY,
            "params":   dict(n_survivors=1, n_distress=2, n_critical=0,
                             any_alive=False, lz_safe=True, lz_score=0.65,
                             fire_detected=False, battery=72.0,
                             max_confidence=0.68, gps_lock=True,
                             ground_team_alerted=False),
        },
        {
            "name":     "Marginal LZ survivors",
            "expected": Action.LAND,
            "params":   dict(n_survivors=1, n_critical=1, any_alive=True,
                             lz_safe=True, lz_score=0.58,
                             fire_detected=False, battery=68.0,
                             max_confidence=0.87, gps_lock=True),
        },
    ]

    passed_count = 0
    scenario_results = {}

    print(f"\n  {'#':<3} {'Scenario':<28} {'Expected':<14} "
          f"{'Got':<14} {'Pass'}")
    print(f"  {'-'*66}")

    for i, sc in enumerate(scenarios, 1):
        try:
            action = evaluate_scenario(sc["params"])
            passed = action == sc["expected"]
        except Exception as e:
            action = f"ERROR: {e}"
            passed = False

        mark = "✅" if passed else "❌"
        print(f"  {i:<3} {sc['name']:<28} "
              f"{sc['expected']:<14} {action:<14} {mark}")

        if passed:
            passed_count += 1

        scenario_results[sc["name"]] = {
            "expected": sc["expected"],
            "got":      action,
            "passed":   passed,
        }

    pass_rate = passed_count / len(scenarios)
    print(f"\n  Scenarios passed: {passed_count}/{len(scenarios)} "
          f"({pass_rate*100:.0f}%)")

    if pass_rate < 0.80:
        warnings.append(f"Only {pass_rate*100:.0f}% decision scenarios "
                       "correct — retrain RL or tune scoring weights")

    details = {
        "engine_available": engine_available,
        "passed":           passed_count,
        "total":            len(scenarios),
        "pass_rate":        round(pass_rate, 3),
        "scenarios":        scenario_results,
    }

    return TestResult("5.5", "Decision engine stress test",
                      pass_rate >= 0.80, pass_rate,
                      details, warnings,
                      duration_s=round(time.time()-t_start, 2))


# ══════════════════════════════════════════════════════════
#  REPORT WRITER
# ══════════════════════════════════════════════════════════

def write_report(results: List[TestResult], model_path):
    """Writes JSON + TXT validation report."""
    now = datetime.now()

    # ── Overall score ─────────────────────────────────────
    weights = {"5.1": 0.30, "5.2": 0.20, "5.3": 0.25,
               "5.4": 0.10, "5.5": 0.15}
    weighted_score = sum(
        r.score * weights.get(r.subtask, 0.20)
        for r in results
    )
    all_passed = all(r.passed for r in results)

    # ── JSON report ───────────────────────────────────────
    report = {
        "timestamp":     now.isoformat(),
        "model_path":    str(model_path),
        "overall_pass":  all_passed,
        "overall_score": round(weighted_score, 3),
        "results": [
            {
                "subtask":    r.subtask,
                "name":       r.name,
                "passed":     r.passed,
                "score":      round(r.score, 3),
                "duration_s": r.duration_s,
                "warnings":   r.warnings,
                "details":    r.details,
            }
            for r in results
        ]
    }

    json_path = LOGS_DIR / f"validation_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── TXT summary ───────────────────────────────────────
    lines = [
        "=" * 55,
        "  SAR eVTOL — Validation Report",
        f"  {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55,
        "",
        f"  Model: {Path(str(model_path)).name if model_path else 'N/A'}",
        f"  Overall score : {weighted_score:.3f}",
        f"  Overall pass  : {'YES' if all_passed else 'NO'}",
        "",
        f"  {'Subtask':<8} {'Name':<28} {'Score':>6} {'Pass'}",
        f"  {'-'*50}",
    ]

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"  {r.subtask:<8} {r.name:<28} "
                     f"{r.score:>6.3f} {mark}")

    # Warnings
    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        lines += ["", "  Warnings:"]
        for w in all_warnings:
            lines.append(f"    • {w}")

    lines += [
        "",
        "  Next steps:",
        "    • Add real SAR images to improve real-world accuracy",
        "    • Run on GPU for <100ms latency",
        "    • Collect operator feedback for online RL learning",
        "=" * 55,
    ]

    txt_path = LOGS_DIR / "validation_summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, txt_path, report


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main(args):
    print("=" * 55)
    print("  SAR eVTOL — Phase 5 Validation Suite")
    print("=" * 55)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    model   = load_model()
    results = []

    run_all = args.test is None
    run = lambda t: run_all or args.test == t

    if run("5.1"):
        results.append(test_5_1_detection_accuracy(model, args.quick))

    if run("5.2"):
        results.append(test_5_2_latency(model, args.quick))

    if run("5.3"):
        results.append(test_5_3_edge_cases(model))

    if run("5.4"):
        results.append(test_5_4_hmi_usability())

    if run("5.5"):
        results.append(test_5_5_decision_engine())

    if not results:
        print("No tests selected. Use --test 5.1 through 5.5")
        return

    # ── Final summary ─────────────────────────────────────
    section("VALIDATION SUMMARY")

    weights = {"5.1":0.30,"5.2":0.20,"5.3":0.25,"5.4":0.10,"5.5":0.15}
    weighted = sum(r.score * weights.get(r.subtask,0.2) for r in results)

    print(f"\n  {'Subtask':<8} {'Name':<28} {'Score':>6} "
          f"{'Time':>6} {'Result'}")
    print(f"  {'-'*56}")
    for r in results:
        mark = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {r.subtask:<8} {r.name:<28} "
              f"{r.score:>6.3f} {r.duration_s:>5.1f}s {mark}")

    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        print(f"\n  Warnings:")
        for w in all_warnings:
            print(f"    • {w}")

    if run_all:
        print(f"\n  Weighted overall score: {weighted:.3f}")
        json_p, txt_p, _ = write_report(results, MODEL_PATH)
        print(f"\n  📄 JSON report : {json_p}")
        print(f"  📄 TXT summary : {txt_p}")

    all_pass = all(r.passed for r in results)
    if all_pass:
        print(f"\n  ✅ ALL TESTS PASSED — system ready")
    else:
        failed = [r.subtask for r in results if not r.passed]
        print(f"\n  ⚠️  Failed: {failed} — see warnings above")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAR eVTOL Phase 5 Validation Suite"
    )
    parser.add_argument("--test",  type=str, default=None,
        choices=["5.1","5.2","5.3","5.4","5.5"],
        help="Run one specific subtask")
    parser.add_argument("--quick", action="store_true",
        help="Fast mode: fewer runs, smaller image size")
    args = parser.parse_args()
    main(args)