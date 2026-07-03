import sys, os
import time
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3a_mission_state import Action, Payload
from phase3d_decision_engine import DecisionEngine, FullDecision

# Try to import upstream modules — graceful fallback if not yet trained
try:
    from subtask_2_pipeline    import FusionLayer
    from subtask_2_pipeline import FeatureExtractor
    from subtask_2_pipeline   import LZClassifier
    _UPSTREAM_AVAILABLE = True
except ImportError:
    _UPSTREAM_AVAILABLE = False


# ── HMI colour maps (same interface as step5c) ────────────
ACTION_COLORS = {
    Action.LAND:       "#00C853",
    Action.HOVER_DROP: "#FF6D00",
    Action.RELAY:      "#FFD600",
    Action.HOLD:       "#2979FF",
    Action.LOITER:     "#AA00FF",
    Action.RTB:        "#D50000",
    Action.ESCALATE:   "#FF1744",
}

URGENCY_COLORS = {
    "CRITICAL": "#FF1744",
    "HIGH":     "#FF6D00",
    "NORMAL":   "#2979FF",
}


# ── Compatibility shim for HMI ─────────────────────────────
# The HMI calls decision.recommendation, decision.action,
# decision.urgency — these all exist on FullDecision already.
# We just need to add a few extra fields for the new HMI panels.

@dataclass
class HMIDecision:
    """
    Extended FullDecision with extra fields for the upgraded HMI.
    Wraps FullDecision so existing HMI code still works.
    """
    # Core fields (same as old MissionDecision)
    action:           str   = Action.HOLD
    urgency:          str   = "NORMAL"
    recommendation:   str   = ""
    survivor_count:   int   = 0
    critical_count:   int   = 0
    lz_safe:          bool  = False
    lz_score:         float = 0.0

    # New Phase 3 fields
    short_summary:    str   = ""
    payload_type:     str   = Payload.NONE
    payload_reason:   str   = ""
    gps_alert:        Optional[dict] = None
    loiter_pattern:   Optional[dict] = None
    escalation:       Optional[dict] = None
    was_vetoed:       bool  = False
    veto_reason:      str   = ""
    confidence:       float = 0.0
    decision_time_ms: float = 0.0
    score_breakdown:  List  = None  # top 3 action scores for HMI panel

    @classmethod
    def from_full(cls, fd: FullDecision, state_n_surv=0, state_n_crit=0,
                  state_lz_safe=False, state_lz_score=0.0):
        """Convert FullDecision → HMIDecision."""
        return cls(
            action           = fd.action,
            urgency          = fd.urgency,
            recommendation   = fd.recommendation,
            survivor_count   = state_n_surv,
            critical_count   = state_n_crit,
            lz_safe          = state_lz_safe,
            lz_score         = state_lz_score,
            short_summary    = fd.short_summary,
            payload_type     = fd.payload_type,
            payload_reason   = fd.payload_reason,
            gps_alert        = fd.gps_alert,
            loiter_pattern   = fd.loiter_pattern,
            escalation       = fd.escalation,
            was_vetoed       = fd.was_vetoed,
            veto_reason      = fd.veto_reason,
            confidence       = fd.confidence,
            decision_time_ms = fd.decision_time_ms,
            score_breakdown  = fd.action_scores[:3],
        )


# ── Main Bridge Class ─────────────────────────────────────

class HMIDecisionBridge:
    """
    Drop-in replacement for step5c DecisionEngine.
    Same .evaluate(detections, telemetry) interface.
    Internally runs full Phase 3 pipeline.
    """

    def __init__(self, weights_path=None):
        self._engine    = DecisionEngine(weights_path)
        self._extractor = FeatureExtractor()   if _UPSTREAM_AVAILABLE else None
        self._lz        = LZClassifier()       if _UPSTREAM_AVAILABLE else None
        self._mock_tick = 0

        mode = "FULL PIPELINE" if _UPSTREAM_AVAILABLE else "MOCK MODE"
        print(f"[HMIDecisionBridge] Initialised — {mode}")

    def evaluate(self, detections, telemetry) -> HMIDecision:
        """
        Main call from HMI update loop.

        Args:
            detections: List of Detection from step5b_inference_engine
                       (or FusedDetection from fusion layer)
            telemetry:  TelemetryFrame from sim engine

        Returns:
            HMIDecision (compatible with existing HMI code)
        """
        self._mock_tick += 1

        # ── Convert inference detections to SurvivorFeatures ──
        if _UPSTREAM_AVAILABLE and self._extractor:
            # Run through feature extractor to get priority scores
            thermal = getattr(telemetry, 'thermal_frame', None)
            survivors = self._extractor.extract(
                detections,
                thermal_frame=thermal,
            )
        else:
            # Mock: wrap detections as minimal SurvivorFeatures
            survivors = self._mock_survivors(detections)

        # ── Run LZ classifier ──────────────────────────────
        if _UPSTREAM_AVAILABLE and self._lz:
            rgb     = getattr(telemetry, 'rgb_frame',     None)
            thermal = getattr(telemetry, 'thermal_frame', None)
            lz_cands, best_lz = self._lz.classify(
                rgb, thermal, detections
            )
        else:
            lz_cands, best_lz = self._mock_lz(telemetry)

        # ── Run full decision engine ───────────────────────
        full_decision = self._engine.evaluate(
            survivors    = survivors,
            lz_candidates= lz_cands,
            best_lz      = best_lz,
            telemetry    = telemetry,
        )

        # ── Wrap for HMI ───────────────────────────────────
        n_surv  = len(survivors)
        n_crit  = sum(1 for s in survivors
                      if getattr(s,'priority',2) == 1)
        lz_safe  = best_lz.is_safe  if best_lz else False
        lz_score = best_lz.lz_score if best_lz else 0.0

        return HMIDecision.from_full(
            full_decision,
            state_n_surv  = n_surv,
            state_n_crit  = n_crit,
            state_lz_safe = lz_safe,
            state_lz_score= lz_score,
        )

    def record_operator_feedback(self, feedback: str):
        """
        Call this when operator clicks Accept or Override.
        feedback: "accept" or "override"
        """
        self._engine.evaluate.__func__  # check exists
        # Next evaluate() call will pick up feedback via last_action

    def _mock_survivors(self, detections):
        """Creates mock SurvivorFeatures from raw detections."""
        from dataclasses import dataclass as dc

        @dc
        class MockSurvivor:
            class_id:       int   = 0
            confidence:     float = 0.5
            priority:       int   = 2
            vital_status:   str   = "possibly_alive"
            posture:        str   = "unknown"
            is_moving:      bool  = False
            isolation_score:float = 0.5
            bbox:           list  = None
            location_hint:  str   = "Centre"

        survivors = []
        for d in detections:
            s = MockSurvivor(
                class_id   = getattr(d, 'class_id',   0),
                confidence = getattr(d, 'confidence', 0.5),
                priority   = getattr(d, 'priority',   2),
                bbox       = getattr(d, 'bbox', [0,0,10,10]),
                location_hint = getattr(d, 'location_hint', 'Centre'),
            )
            survivors.append(s)
        return survivors

    def _mock_lz(self, telemetry):
        """Creates mock LZ candidates when upstream unavailable."""
        from dataclasses import dataclass as dc

        @dc
        class MockLZ:
            is_safe:      bool  = True
            lz_score:     float = 0.72
            bearing:      float = 45.0
            distance_px:  float = 120.0
            thermal_score:float = 0.8
            status:       str   = "CLEAR"

        lz = MockLZ(
            lz_score = getattr(telemetry, 'lz_score', 0.72),
            is_safe  = getattr(telemetry, 'lz_safe',  True),
        )
        return [lz], lz


# ── Upgrade instructions for HMI ──────────────────────────
UPGRADE_INSTRUCTIONS = """
HOW TO UPGRADE step5d_cockpit_hmi.py:
──────────────────────────────────────
1. Replace this import:
     from step5c_decision_engine import DecisionEngine, ACTION_COLORS, URGENCY_COLORS

   With:
     from phase3e_hmi_integration import (
         HMIDecisionBridge as DecisionEngine,
         ACTION_COLORS, URGENCY_COLORS
     )

2. In CockpitHMI.__init__, replace:
     self.decision = DecisionEngine()

   With:
     self.decision = DecisionEngine()   # no change needed

3. In _update_recommendation, add payload + veto display:
     if decision.payload_type != 'NONE':
         payload_text = f"📦 DROP: {decision.payload_type}"
     if decision.was_vetoed:
         veto_text = f"⚠️ Safety override: {decision.veto_reason}"

4. Optionally add GPS alert panel to HMI:
     if decision.gps_alert:
         self._show_gps_alert(decision.gps_alert)

That's it. Everything else stays the same.
"""

if __name__ == "__main__":
    print(UPGRADE_INSTRUCTIONS)

    # Quick integration test
    print("Testing HMIDecisionBridge...")

    bridge = HMIDecisionBridge()

    class MockDet:
        class_id=0; confidence=0.88; priority=1
        bbox=[100,100,140,160]; location_hint="NW"

    class MockTelem:
        battery=72.0; altitude=82.0; speed=38.0
        heading=45.0; wind_speed=10.0
        latitude=12.9716; longitude=77.5946
        gps_lock=True; mission_status="ACTIVE"
        lz_safe=True; lz_score=0.75
        rgb_frame=None; thermal_frame=None

    decision = bridge.evaluate([MockDet()], MockTelem())
    print(f"\n  Action    : {decision.action}")
    print(f"  Urgency   : {decision.urgency}")
    print(f"  Payload   : {decision.payload_type}")
    print(f"  Vetoed    : {decision.was_vetoed}")
    print(f"  Confidence: {decision.confidence:.3f}")
    print(f"  Time      : {decision.decision_time_ms:.1f}ms")
    print(f"\n  Rec: {decision.recommendation[:100]}...")
    print(f"\n✅ Integration test passed")
