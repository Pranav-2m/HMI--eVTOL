"""
=============================================================
STEP 5C — Decision Engine
=============================================================
Takes detections from the inference engine + telemetry from
the sim engine and produces:
  1. A mission ACTION  (LAND / HOVER_DROP / RELAY / HOLD)
  2. A plain-English RECOMMENDATION for the operator
  3. An urgency level  (CRITICAL / HIGH / NORMAL)

This is the "brain" that translates ML outputs into
operator-readable guidance shown on the HMI.
=============================================================
"""

import time
import random
from dataclasses import dataclass, field
from typing import List, Optional


# ── Decision Output Structure ──────────────────────────────

@dataclass
class MissionDecision:
    # Primary action the AI recommends
    action:       str   = "HOLD"
    # LAND         → safe to land, survivors reachable
    # HOVER_DROP   → too dangerous to land, drop supplies
    # RELAY        → relay survivor coords to ground team
    # HOLD         → no action yet, continue scanning
    # RTB          → return to base (low battery/emergency)

    # Human-readable recommendation for cockpit display
    recommendation: str = ""

    # Urgency level drives alert colour on HMI
    urgency:      str   = "NORMAL"   # CRITICAL / HIGH / NORMAL

    # Supporting details
    survivor_count:  int   = 0
    critical_count:  int   = 0
    lz_safe:         bool  = True
    lz_score:        float = 1.0
    confidence:      float = 1.0

    # For map panel — suggested GPS offset to LZ
    lz_bearing:   Optional[float] = None   # degrees from current pos
    lz_distance:  Optional[float] = None   # metres

    # Timestamp
    timestamp:    float = field(default_factory=time.time)


# ── Decision Engine ────────────────────────────────────────

class DecisionEngine:
    def __init__(self):
        # Track last decision to avoid spamming same recommendation
        self._last_action    = None
        self._last_tick      = 0
        self._hold_count     = 0    # consecutive HOLD frames
        self._decision_count = 0

    def evaluate(self, detections, telemetry):
        """
        Main decision function. Called every frame.

        Args:
            detections: List[Detection] from inference engine
            telemetry:  TelemetryFrame from sim engine

        Returns:
            MissionDecision
        """
        self._decision_count += 1

        # ── Extract key counts ────────────────────────────
        humans   = [d for d in detections if d.class_id == 0]
        distress = [d for d in detections if d.class_id == 1]
        fires    = [d for d in detections if d.class_id == 2]

        critical = [d for d in detections if d.priority == 1]
        total    = len(detections)
        survivor_count = len(humans) + len(distress)
        critical_count = len(critical)

        lz_safe  = telemetry.lz_safe
        lz_score = telemetry.lz_score
        battery  = telemetry.battery
        status   = telemetry.mission_status

        # ── Override: Emergency conditions first ──────────
        if status == "EMERGENCY" or battery < 10:
            return MissionDecision(
                action         = "RTB",
                recommendation = (
                    "⚠️ CRITICAL BATTERY. Returning to base immediately. "
                    "Relay survivor coordinates to ground team before departure."
                ),
                urgency        = "CRITICAL",
                survivor_count = survivor_count,
                critical_count = critical_count,
                lz_safe        = lz_safe,
                lz_score       = lz_score,
            )

        if status == "RTB" or battery < 25:
            return MissionDecision(
                action         = "RTB",
                recommendation = (
                    f"⚠️ Low battery ({battery:.0f}%). "
                    "Complete current pass and return to base. "
                    "Ground team has been alerted with survivor locations."
                ),
                urgency        = "HIGH",
                survivor_count = survivor_count,
                critical_count = critical_count,
                lz_safe        = lz_safe,
                lz_score       = lz_score,
            )

        # ── No detections ─────────────────────────────────
        if total == 0:
            self._hold_count += 1
            hold_msg = (
                f"No survivors detected. Continuing scan pattern. "
                f"({self._hold_count} consecutive clear frames)"
            )
            return MissionDecision(
                action         = "HOLD",
                recommendation = hold_msg,
                urgency        = "NORMAL",
                survivor_count = 0,
                critical_count = 0,
                lz_safe        = lz_safe,
                lz_score       = lz_score,
            )

        # Reset hold counter when detections resume
        self._hold_count = 0

        # ── Build location summary ─────────────────────────
        locs = list(set(d.location_hint for d in detections))
        loc_str = ", ".join(locs[:3])

        # ── Decision tree ─────────────────────────────────

        # Case 1: Survivors found AND LZ is safe → LAND
        if survivor_count > 0 and lz_safe and lz_score > 0.65 \
                and len(fires) == 0:
            lz_bearing  = round(random.uniform(0, 360), 1)
            lz_distance = round(random.uniform(20, 80), 0)

            if critical_count > 0:
                urgency = "CRITICAL"
                rec = (
                    f"🔴 {critical_count} CRITICAL survivor(s) detected "
                    f"at {loc_str}. "
                    f"LZ is CLEAR (score: {lz_score:.0%}). "
                    f"Recommend immediate landing — "
                    f"bearing {lz_bearing}° at {lz_distance:.0f}m."
                )
            else:
                urgency = "HIGH"
                rec = (
                    f"🟡 {survivor_count} survivor(s) detected at {loc_str}. "
                    f"LZ is clear (score: {lz_score:.0%}). "
                    f"Recommend landing — bearing {lz_bearing}° "
                    f"at {lz_distance:.0f}m."
                )

            return MissionDecision(
                action         = "LAND",
                recommendation = rec,
                urgency        = urgency,
                survivor_count = survivor_count,
                critical_count = critical_count,
                lz_safe        = True,
                lz_score       = lz_score,
                lz_bearing     = lz_bearing,
                lz_distance    = lz_distance,
            )

        # Case 2: Survivors found BUT LZ unsafe or fire present → HOVER_DROP
        if survivor_count > 0 and (not lz_safe or len(fires) > 0):
            hazards = []
            if not lz_safe:
                hazards.append(f"LZ unsafe (score: {lz_score:.0%})")
            if len(fires) > 0:
                hazards.append(f"{len(fires)} fire source(s) nearby")
            hazard_str = " + ".join(hazards)

            rec = (
                f"🔴 {survivor_count} survivor(s) at {loc_str}. "
                f"Landing blocked: {hazard_str}. "
                f"Recommend hovering and deploying supply kit. "
                f"Relaying GPS coordinates to ground rescue team."
            )
            return MissionDecision(
                action         = "HOVER_DROP",
                recommendation = rec,
                urgency        = "CRITICAL" if critical_count > 0 else "HIGH",
                survivor_count = survivor_count,
                critical_count = critical_count,
                lz_safe        = False,
                lz_score       = lz_score,
            )

        # Case 3: Distress signals only (no confirmed humans) → RELAY
        if len(distress) > 0 and len(humans) == 0:
            rec = (
                f"🟡 {len(distress)} distress signal(s) detected "
                f"at {loc_str}. "
                f"No confirmed survivors yet — relaying signal coordinates "
                f"to ground team for investigation. Continue scanning."
            )
            return MissionDecision(
                action         = "RELAY",
                recommendation = rec,
                urgency        = "HIGH",
                survivor_count = survivor_count,
                critical_count = critical_count,
                lz_safe        = lz_safe,
                lz_score       = lz_score,
            )

        # Case 4: Low confidence detections → HOLD and reconfirm
        low_conf = all(d.confidence < 0.5 for d in detections)
        if low_conf:
            rec = (
                f"🔵 {total} possible detection(s) at {loc_str} "
                f"— confidence below threshold. "
                f"Descending for closer scan before committing."
            )
            return MissionDecision(
                action         = "HOLD",
                recommendation = rec,
                urgency        = "NORMAL",
                survivor_count = survivor_count,
                critical_count = critical_count,
                lz_safe        = lz_safe,
                lz_score       = lz_score,
            )

        # Fallback
        return MissionDecision(
            action         = "HOLD",
            recommendation = (
                f"Monitoring {total} detection(s). "
                f"Assessing situation before action."
            ),
            urgency        = "NORMAL",
            survivor_count = survivor_count,
            critical_count = critical_count,
            lz_safe        = lz_safe,
            lz_score       = lz_score,
        )


# ── Action colour mapping ──────────────────────────────────
# Used by the HMI to colour-code the recommendation panel.

ACTION_COLORS = {
    "LAND":       "#00C853",   # green
    "HOVER_DROP": "#FF6D00",   # orange
    "RELAY":      "#FFD600",   # yellow
    "HOLD":       "#2979FF",   # blue
    "RTB":        "#D50000",   # red
}

URGENCY_COLORS = {
    "CRITICAL": "#FF1744",
    "HIGH":     "#FF6D00",
    "NORMAL":   "#2979FF",
}


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    # Import detection mock
    import sys
    sys.path.insert(0, ".")
    from step5b_inference_engine import InferenceEngine, Detection
    import numpy as np

    engine   = InferenceEngine()
    decision = DecisionEngine()

    # Mock telemetry
    class MockTelem:
        battery = 80.0
        lz_safe = True
        lz_score = 0.82
        mission_status = "ACTIVE"

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dets  = engine.run(dummy)
    dec   = decision.evaluate(dets, MockTelem())

    print(f"\nDecision: {dec.action}")
    print(f"Urgency:  {dec.urgency}")
    print(f"Rec:      {dec.recommendation}")
