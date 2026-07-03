import time
import math
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

from phase3a_mission_state import (
    MissionState, MissionStateBuilder,
    Action, Payload, ALL_ACTIONS
)
from phase3b_scoring_engine import ScoringEngine, ActionScore


# ── Full Decision Output ───────────────────────────────────

@dataclass
class FullDecision:
    # ── Primary action ────────────────────────────────────
    action:       str   = Action.HOLD
    urgency:      str   = "NORMAL"    # CRITICAL / HIGH / NORMAL
    confidence:   float = 0.0         # engine confidence in decision

    # ── Human-readable recommendation ─────────────────────
    recommendation: str = ""
    short_summary:  str = ""          # 1-line for status bar

    # ── Payload (for HOVER_DROP) ──────────────────────────
    payload_type:   str  = Payload.NONE
    payload_reason: str  = ""

    # ── GPS alert (for RELAY/LAND/HOVER_DROP) ─────────────
    gps_alert:      Optional[Dict] = None
    # {lat, lon, n_survivors, priority, message}

    # ── Loiter pattern (for LOITER) ───────────────────────
    loiter_pattern: Optional[Dict] = None
    # {type, radius_m, altitude_m, heading_step}

    # ── Escalation (for ESCALATE) ─────────────────────────
    escalation:     Optional[Dict] = None
    # {reason, n_critical, n_fires, message}

    # ── LZ info ───────────────────────────────────────────
    lz_bearing:     Optional[float] = None
    lz_distance_m:  Optional[float] = None
    lz_score:       float = 0.0

    # ── Safety veto info ──────────────────────────────────
    was_vetoed:     bool  = False
    veto_reason:    str   = ""
    original_action:str   = ""   # what engine wanted before veto

    # ── Score breakdown (for HMI transparency) ────────────
    action_scores:  List[ActionScore] = field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────
    frame_id:       int   = 0
    timestamp:      float = field(default_factory=time.time)
    decision_time_ms: float = 0.0    # how long decision took


# ── Safety Layer ──────────────────────────────────────────

class SafetyLayer:
    """
    Hard safety rules that VETO any action unconditionally.
    These cannot be overridden by the ML scoring layer.
    They represent constraints that must always hold
    for airworthiness and crew safety.
    """

    def check(self, action: str, state: MissionState):
        """
        Returns (vetoed: bool, veto_reason: str, forced_action: str)
        If vetoed, forced_action is what we do instead.
        """
        # ── VETO 1: Battery critical → must RTB ───────────
        if state.battery_critical and action != Action.RTB:
            return True, (
                f"Battery critical ({state.battery:.0f}%) — "
                "override to RTB"
            ), Action.RTB

        # ── VETO 2: Never LAND on fire ─────────────────────
        if action == Action.LAND and state.fire_below_lz:
            return True, (
                "Fire detected below LZ — landing unsafe"
            ), Action.HOVER_DROP

        # ── VETO 3: Never LAND with LZ score < 0.35 ────────
        if action == Action.LAND and state.lz_score < 0.35:
            return True, (
                f"LZ score too low ({state.lz_score:.2f} < 0.35) "
                "— switch to HOVER_DROP"
            ), Action.HOVER_DROP

        # ── VETO 4: Never RELAY without GPS lock ───────────
        if action == Action.RELAY and not state.gps_lock:
            return True, (
                "GPS lock lost — coordinates unreliable, HOLD instead"
            ), Action.HOLD

        # ── VETO 5: No HOVER_DROP with no survivors ─────────
        if action == Action.HOVER_DROP and not state.has_survivors:
            return True, (
                "No survivors detected — supply drop cancelled"
            ), Action.HOLD

        return False, "", action


# ── Payload Selector ──────────────────────────────────────

class PayloadSelector:
    """
    Determines what type of supply to drop based on survivor state.
    Only called when action == HOVER_DROP.
    """

    def select(self, state: MissionState):
        """Returns (Payload type, reason string)."""

        # If any survivor shows low thermal = cold/hypothermic
        cold_survivors = any(
            getattr(s, 'vital_status', '') in ['cold', 'status_unknown']
            for s in state.survivors
        )
        # If any survivor lying flat = likely injured
        injured = any(
            getattr(s, 'posture', '') in ['lying', 'flat_on_ground']
            for s in state.survivors
        )
        # If distress signals present = likely conscious but stuck
        comms_needed = state.n_distress > 0 and state.n_critical == 0

        # If multiple survivors = food/water priority
        group = state.n_survivors >= 3

        # Priority order: medical > comms > water > rope
        if injured or cold_survivors or state.has_critical:
            return Payload.MEDICAL_KIT, (
                "critical/injured survivor detected — medical kit"
            )
        elif comms_needed:
            return Payload.COMMS_DEVICE, (
                "distress signals detected — drop comms device"
            )
        elif group:
            return Payload.WATER_FOOD, (
                f"{state.n_survivors} survivors — water/food rations"
            )
        else:
            return Payload.ROPE_HARNESS, (
                "survivor may need extraction — drop rope/harness"
            )


# ── GPS Alert Builder ─────────────────────────────────────

class GPSAlertBuilder:
    """Builds a ground team GPS alert from current state."""

    def build(self, state: MissionState, action: str):
        """
        Returns alert dict or None.
        Called for RELAY, LAND, HOVER_DROP actions.
        """
        if not state.gps_lock:
            return None
        if not state.has_survivors and state.n_distress == 0:
            return None

        priority_str = (
            "CRITICAL" if state.has_critical else
            "HIGH"     if state.n_high > 0    else "MEDIUM"
        )
        message = (
            f"SAR eVTOL [{action}] — "
            f"{state.n_survivors} survivor(s) detected "
            f"({state.n_critical} critical). "
            f"Coordinates: {state.gps_coords_str()}. "
            f"Altitude: {state.altitude:.0f}m AGL. "
            f"Priority: {priority_str}."
        )
        return {
            "latitude":    state.latitude,
            "longitude":   state.longitude,
            "n_survivors": state.n_survivors,
            "n_critical":  state.n_critical,
            "priority":    priority_str,
            "message":     message,
            "timestamp":   time.strftime("%H:%M:%S"),
        }


# ── Loiter Pattern Generator ──────────────────────────────

class LoiterPatternGenerator:
    """
    Generates an autonomous loiter flight pattern when
    the eVTOL needs to maintain presence without landing.

    Pattern types:
      CIRCLE  — orbit around survivor location
      FIGURE8 — covers wider area, good for multiple survivors
      HOLD    — stationary hover (no pattern, just maintain position)
    """

    def generate(self, state: MissionState):
        """Returns loiter pattern dict."""
        # Select pattern based on scenario
        if state.n_survivors > 2:
            pattern_type = "FIGURE8"
            radius_m     = 60.0
        elif state.has_critical:
            pattern_type = "CIRCLE"
            radius_m     = 30.0
        else:
            pattern_type = "HOLD"
            radius_m     = 0.0

        return {
            "type":          pattern_type,
            "radius_m":      radius_m,
            "altitude_m":    max(40.0, state.altitude),
            "heading_step":  15,    # degrees per waypoint
            "n_waypoints":   int(360 / 15) if pattern_type != "HOLD" else 1,
            "reason":        (
                f"No safe LZ — maintaining "
                f"{pattern_type} at {radius_m:.0f}m radius "
                f"until ground team arrives"
            ),
        }


# ── Recommendation Writer ─────────────────────────────────

class RecommendationWriter:
    """Writes human-readable recommendation text for the HMI."""

    def write(
        self,
        action: str,
        state: MissionState,
        payload_type: str,
        gps_alert: Optional[dict],
        loiter: Optional[dict],
        veto_reason: str,
    ):
        """Returns (recommendation: str, short_summary: str)"""

        loc_hints = list(set(
            getattr(s, 'location_hint', '?')
            for s in state.survivors
        ))[:3]
        loc_str = ", ".join(loc_hints) if loc_hints else "unknown"

        if action == Action.LAND:
            rec = (
                f"🟢 {state.n_survivors} survivor(s) detected at "
                f"{loc_str}. LZ CLEAR (score: {state.lz_score:.0%}). "
                f"Recommend immediate landing — bearing "
                f"{state.lz_bearing:.0f}° at {state.lz_distance_m:.0f}m."
            )
            short = f"LAND — {state.n_survivors} survivors | LZ {state.lz_score:.0%}"

        elif action == Action.HOVER_DROP:
            rec = (
                f"🟡 {state.n_survivors} survivor(s) at {loc_str}. "
                f"Landing blocked "
                f"({'fire detected' if state.fire_detected else 'unsafe LZ'}). "
                f"Deploying {payload_type.replace('_',' ')}. "
                f"GPS coordinates relayed to ground team."
            )
            short = f"HOVER+DROP {payload_type} | {state.n_survivors} survivors"

        elif action == Action.RELAY:
            rec = (
                f"🔵 {state.n_survivors} survivor(s) / "
                f"{state.n_distress} distress signal(s) at {loc_str}. "
                f"Relaying GPS coordinates to ground rescue team. "
                f"Continue scanning area."
            )
            short = f"RELAY GPS | {state.n_survivors} survivors detected"

        elif action == Action.HOLD:
            if state.n_survivors == 0:
                rec = (
                    f"⚪ No survivors detected. Continuing scan pattern. "
                    f"({state.consecutive_clear_frames} clear frames)"
                )
            else:
                rec = (
                    f"🔵 {state.n_survivors} low-confidence detection(s). "
                    f"Descending for closer look before committing."
                )
            short = "HOLD — scanning"

        elif action == Action.LOITER:
            pattern_str = loiter['type'] if loiter else "HOLD"
            rec = (
                f"🟡 {state.n_survivors} survivor(s) confirmed. "
                f"No safe LZ available. "
                f"Entering {pattern_str} loiter pattern at "
                f"{(loiter or {}).get('radius_m', 0):.0f}m radius. "
                f"Ground team alerted — maintaining visual contact."
            )
            short = f"LOITER {pattern_str} | awaiting ground team"

        elif action == Action.RTB:
            if state.battery_critical:
                rec = (
                    f"⛔ CRITICAL BATTERY ({state.battery:.0f}%). "
                    f"Returning to base immediately. "
                    f"{'GPS alert sent to ground team.' if gps_alert else ''}"
                )
            else:
                rec = (
                    f"⚠️  Low battery ({state.battery:.0f}%). "
                    f"Completing current pass then RTB. "
                    f"Ground team has survivor coordinates."
                )
            short = f"RTB — battery {state.battery:.0f}%"

        elif action == Action.ESCALATE:
            rec = (
                f"🔴 ESCALATING TO COMMAND: "
                f"{state.n_critical} critical, {state.n_fires} fire(s). "
                f"Scene beyond single-drone capability. "
                f"Requesting additional assets. "
                f"Maintaining position and relaying data."
            )
            short = f"ESCALATE | {state.n_critical} critical + {state.n_fires} fires"

        else:
            rec   = f"Action: {action}"
            short = action

        # Append veto note if applicable
        if veto_reason:
            rec += f"\n[Safety override: {veto_reason}]"

        return rec, short


# ── Main Decision Engine ──────────────────────────────────

class DecisionEngine:
    """
    Main engine — called each frame to produce a FullDecision.
    Wires all layers together.
    """

    def __init__(self, weights_path=None):
        self.safety    = SafetyLayer()
        self.scoring   = ScoringEngine(weights_path)
        self.payload   = PayloadSelector()
        self.gps       = GPSAlertBuilder()
        self.loiter    = LoiterPatternGenerator()
        self.writer    = RecommendationWriter()
        self.state_builder = MissionStateBuilder()

    def evaluate(
        self,
        survivors,       # List[SurvivorFeatures] from 2.4
        lz_candidates,   # List[LZCandidate] from 2.5
        best_lz,         # LZCandidate or None from 2.5
        telemetry,       # TelemetryFrame from sim engine
        operator_feedback: Optional[str] = None,
        # "accept" or "override" — for online weight updates
    ) -> FullDecision:
        """
        Main entry point. Returns complete FullDecision.
        """
        t_start = time.perf_counter()

        # ── Build mission state ───────────────────────────
        state = self.state_builder.build(
            telemetry, survivors, lz_candidates, best_lz
        )

        # ── Online learning from operator feedback ────────
        if operator_feedback == "accept" and hasattr(self, '_last_action'):
            self.scoring.update_weights_from_feedback(
                state, self._last_action, reward=+1.0
            )
        elif operator_feedback == "override" and hasattr(self, '_last_action'):
            self.scoring.update_weights_from_feedback(
                state, self._last_action, reward=-0.5
            )
            self.state_builder.record_override()

        # ── Score all actions ─────────────────────────────
        action_scores = self.scoring.score_all(state)
        best_action   = action_scores[0].action

        # ── Safety layer check ────────────────────────────
        vetoed, veto_reason, final_action = self.safety.check(
            best_action, state
        )
        original_action = best_action if vetoed else ""

        # ── Payload selection ─────────────────────────────
        payload_type, payload_reason = (
            self.payload.select(state)
            if final_action == Action.HOVER_DROP
            else (Payload.NONE, "")
        )

        # ── GPS alert ─────────────────────────────────────
        gps_alert = (
            self.gps.build(state, final_action)
            if final_action in [Action.RELAY, Action.LAND,
                                 Action.HOVER_DROP, Action.ESCALATE]
            else None
        )

        # ── Loiter pattern ────────────────────────────────
        loiter_pattern = (
            self.loiter.generate(state)
            if final_action == Action.LOITER
            else None
        )

        # ── Escalation ────────────────────────────────────
        escalation = None
        if final_action == Action.ESCALATE:
            escalation = {
                "reason":     "mass_casualty_or_multi_hazard",
                "n_critical": state.n_critical,
                "n_fires":    state.n_fires,
                "message":    (
                    f"ESCALATE: {state.n_critical} critical survivors, "
                    f"{state.n_fires} fire sources. "
                    f"Additional assets required."
                ),
            }

        # ── Recommendation text ───────────────────────────
        rec, short = self.writer.write(
            final_action, state, payload_type,
            gps_alert, loiter_pattern, veto_reason
        )

        # ── Urgency level ─────────────────────────────────
        if state.battery_critical or state.n_critical >= 3:
            urgency = "CRITICAL"
        elif state.has_critical or state.battery_low:
            urgency = "HIGH"
        else:
            urgency = "NORMAL"

        # ── Engine confidence ─────────────────────────────
        top_score = action_scores[0].total_score
        second    = action_scores[1].total_score if len(action_scores) > 1 else 0
        confidence = min(1.0, top_score - second + 0.5)

        # ── Record & return ───────────────────────────────
        self.state_builder.record_action(final_action)
        self._last_action = final_action
        t_ms = (time.perf_counter() - t_start) * 1000

        return FullDecision(
            action          = final_action,
            urgency         = urgency,
            confidence      = round(confidence, 3),
            recommendation  = rec,
            short_summary   = short,
            payload_type    = payload_type,
            payload_reason  = payload_reason,
            gps_alert       = gps_alert,
            loiter_pattern  = loiter_pattern,
            escalation      = escalation,
            lz_bearing      = state.lz_bearing,
            lz_distance_m   = state.lz_distance_m,
            lz_score        = state.lz_score,
            was_vetoed      = vetoed,
            veto_reason     = veto_reason,
            original_action = original_action,
            action_scores   = action_scores,
            frame_id        = state.frame_id,
            timestamp       = state.timestamp,
            decision_time_ms= round(t_ms, 2),
        )


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    engine = DecisionEngine()

    class MockTelem:
        battery=75.0; altitude=82.0; speed=38.0; heading=45.0
        wind_speed=12.0; latitude=12.9716; longitude=77.5946
        gps_lock=True; mission_status="ACTIVE"

    scenarios = [
        ("2 survivors, safe LZ",
         dict(n_survivors=2, n_critical=1, any_alive=True,
              lz_safe=True, lz_score=0.78, fire_detected=False,
              max_confidence=0.91, battery=72.0)),
        ("Fire rescue, no LZ",
         dict(n_survivors=1, n_critical=1, any_alive=True,
              lz_safe=False, lz_score=0.22, fire_detected=True,
              fire_below_lz=True, max_confidence=0.88, battery=65.0)),
        ("Low battery, survivors",
         dict(n_survivors=1, n_critical=1, any_alive=True,
              lz_safe=True, lz_score=0.70, fire_detected=False,
              max_confidence=0.85, battery=18.0)),
        ("Mass casualty",
         dict(n_survivors=6, n_critical=4, any_alive=True,
              lz_safe=False, lz_score=0.15, fire_detected=True,
              n_fires=3, max_confidence=0.92, battery=55.0)),
    ]

    print("\n" + "="*60)
    print("  🧪 Decision Engine Test — 4 Scenarios")
    print("="*60)

    for name, params in scenarios:
        state = MissionState(**params)
        telem = MockTelem()
        # Patch battery into telemetry
        telem.battery = params.get("battery", 75.0)

        # Manually build state for testing
        engine.state_builder._frame_id += 1
        decision = FullDecision()

        # Quick direct scoring test
        scores = engine.scoring.score_all(state)
        safety = engine.safety
        vetoed, veto_r, final = safety.check(scores[0].action, state)
        action = final

        print(f"\n  📍 Scenario: {name}")
        print(f"     Action  : {action}")
        print(f"     Vetoed  : {vetoed} {('→ ' + veto_r) if vetoed else ''}")
        print(f"     Scores  : ", end="")
        for s in scores[:3]:
            print(f"{s.action}={s.total_score:.3f}", end="  ")
        print()
