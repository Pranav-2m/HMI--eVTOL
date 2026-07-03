import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import setup_path
setup_path()

# ── Actions the engine can output ─────────────────────────
class Action:
    LAND        = "LAND"         # safe to land, direct rescue
    HOVER_DROP  = "HOVER_DROP"   # too dangerous to land, drop supplies
    RELAY       = "RELAY"        # relay GPS to ground team
    HOLD        = "HOLD"         # continue scanning
    LOITER      = "LOITER"       # autonomous loiter pattern (no LZ)
    RTB         = "RTB"          # return to base
    ESCALATE    = "ESCALATE"     # beyond scope, alert command

ALL_ACTIONS = [
    Action.LAND, Action.HOVER_DROP, Action.RELAY,
    Action.HOLD, Action.LOITER, Action.RTB, Action.ESCALATE
]

# ── Payload types for supply drops ────────────────────────
class Payload:
    MEDICAL_KIT   = "MEDICAL_KIT"    # first aid, bandages, meds
    WATER_FOOD    = "WATER_FOOD"     # rations, water pouches
    COMMS_DEVICE  = "COMMS_DEVICE"   # radio/beacon for survivors
    ROPE_HARNESS  = "ROPE_HARNESS"   # extraction equipment
    NONE          = "NONE"


# ── Mission State ──────────────────────────────────────────

@dataclass
class MissionState:
    """
    Complete snapshot of mission context at one point in time.
    Populated each frame before the decision engine runs.
    """

    # ── Timestamp & frame ─────────────────────────────────
    frame_id:       int   = 0
    timestamp:      float = field(default_factory=time.time)
    mission_elapsed:float = 0.0    # seconds since mission start

    # ── Telemetry ─────────────────────────────────────────
    battery:        float = 100.0  # % remaining
    altitude:       float = 80.0   # metres AGL
    speed:          float = 0.0    # km/h
    heading:        float = 0.0    # degrees
    wind_speed:     float = 0.0    # knots
    latitude:       float = 0.0
    longitude:      float = 0.0
    gps_lock:       bool  = True

    # ── Survivor detections (from 2.4 feature extractor) ──
    survivors:      List[Any] = field(default_factory=list)
    # each item is a SurvivorFeatures object

    # ── Pre-computed survivor summaries ───────────────────
    n_survivors:    int   = 0
    n_critical:     int   = 0      # priority == 1
    n_high:         int   = 0      # priority == 2
    n_distress:     int   = 0      # class_id == 1
    n_fires:        int   = 0      # class_id == 2
    any_alive:      bool  = False  # any survivor with vital=likely_alive
    any_moving:     bool  = False  # any survivor showing motion
    any_isolated:   bool  = False  # any survivor with isolation > 0.7
    max_confidence: float = 0.0    # highest detection confidence
    mean_confidence:float = 0.0    # mean detection confidence

    # ── LZ information (from 2.5 LZ classifier) ───────────
    lz_candidates:  List[Any] = field(default_factory=list)
    best_lz:        Optional[Any] = None   # LZCandidate or None
    lz_safe:        bool  = False
    lz_score:       float = 0.0
    lz_bearing:     float = 0.0    # degrees to best LZ
    lz_distance_m:  float = 0.0    # estimated metres to best LZ

    # ── Scene hazards ─────────────────────────────────────
    fire_detected:  bool  = False
    fire_below_lz:  bool  = False  # fire in/near the best LZ
    smoke_detected: bool  = False  # heuristic from thermal
    visibility:     str   = "CLEAR"  # CLEAR / REDUCED / POOR

    # ── Mission history (across frames) ───────────────────
    consecutive_clear_frames: int = 0   # frames with no detections
    last_action:    str   = Action.HOLD
    action_history: List[str] = field(default_factory=list)
    operator_override_count: int = 0    # how many times op overrode AI

    # ── Computed risk scores (filled by scoring engine) ───
    action_scores:  Dict[str, float] = field(default_factory=dict)

    # ── Ground team status ────────────────────────────────
    ground_team_alerted: bool  = False
    ground_team_eta_s:   float = 0.0   # seconds until team arrives

    # ── Previous decisions (for context) ──────────────────
    decision_log:   List[str] = field(default_factory=list)


    # ── Convenience methods ───────────────────────────────

    @property
    def battery_critical(self): return self.battery < 12
    @property
    def battery_low(self):      return self.battery < 28
    @property
    def battery_ok(self):       return self.battery >= 28

    @property
    def has_survivors(self):    return self.n_survivors > 0
    @property
    def has_critical(self):     return self.n_critical > 0

    @property
    def mission_duration_min(self):
        return self.mission_elapsed / 60

    @property
    def estimated_range_km(self):
        """Rough estimate of remaining flight range based on battery."""
        return max(0.0, (self.battery - 12) * 0.08)

    def gps_coords_str(self):
        return f"{self.latitude:.5f},{self.longitude:.5f}"

    def to_feature_vector(self):
        """
        Converts state to a numeric feature vector for the RL agent.
        Normalises all values to [0, 1] range.

        This is what the RL agent 'sees' as its observation space.
        Shape: (20,) float32 array

        Index  Feature
        ─────  ──────────────────────────────
        0      battery / 100
        1      altitude / 150
        2      n_survivors / 10 (capped)
        3      n_critical / 5
        4      n_fires / 5
        5      lz_score (already 0–1)
        6      lz_safe (0 or 1)
        7      fire_detected (0 or 1)
        8      fire_below_lz (0 or 1)
        9      max_confidence
        10     mean_confidence
        11     any_alive (0 or 1)
        12     any_moving (0 or 1)
        13     any_isolated (0 or 1)
        14     wind_speed / 30
        15     consecutive_clear_frames / 20 (capped)
        16     gps_lock (0 or 1)
        17     ground_team_alerted (0 or 1)
        18     n_distress / 5
        19     mission_elapsed / 1800 (30 min cap)
        """
        import numpy as np
        return np.array([
            self.battery / 100.0,
            min(1.0, self.altitude / 150.0),
            min(1.0, self.n_survivors / 10.0),
            min(1.0, self.n_critical / 5.0),
            min(1.0, self.n_fires / 5.0),
            self.lz_score,
            float(self.lz_safe),
            float(self.fire_detected),
            float(self.fire_below_lz),
            self.max_confidence,
            self.mean_confidence,
            float(self.any_alive),
            float(self.any_moving),
            float(self.any_isolated),
            min(1.0, self.wind_speed / 30.0),
            min(1.0, self.consecutive_clear_frames / 20.0),
            float(self.gps_lock),
            float(self.ground_team_alerted),
            min(1.0, self.n_distress / 5.0),
            min(1.0, self.mission_elapsed / 1800.0),
        ], dtype='float32')


# ── State Builder ──────────────────────────────────────────
# Converts raw engine outputs into a MissionState.

class MissionStateBuilder:
    """
    Builds a MissionState from raw outputs of all upstream modules.
    Called each frame before the decision engine runs.
    """

    def __init__(self):
        self._frame_id = 0
        self._mission_start = time.time()
        self._consecutive_clear = 0
        self._last_action = Action.HOLD
        self._action_history = []
        self._override_count = 0
        self._ground_alerted = False

    def build(
        self,
        telemetry,        # TelemetryFrame from sim engine
        survivors,        # List[SurvivorFeatures] from 2.4
        lz_candidates,    # List[LZCandidate] from 2.5
        best_lz,          # LZCandidate or None
    ) -> MissionState:

        self._frame_id += 1
        elapsed = time.time() - self._mission_start

        # ── Survivor statistics ────────────────────────────
        n_surv     = len(survivors)
        n_crit     = sum(1 for s in survivors if s.priority == 1)
        n_high     = sum(1 for s in survivors if s.priority == 2)
        n_dist     = sum(1 for s in survivors
                         if hasattr(s, 'class_id') and s.class_id == 1)
        n_fire     = sum(1 for s in survivors
                         if hasattr(s, 'class_id') and s.class_id == 2)
        any_alive  = any(getattr(s, 'vital_status', '') == 'likely_alive'
                         for s in survivors)
        any_mov    = any(getattr(s, 'is_moving', False) for s in survivors)
        any_iso    = any(getattr(s, 'isolation_score', 0) > 0.7
                         for s in survivors)
        confs      = [s.confidence for s in survivors] if survivors else [0]
        max_conf   = max(confs)
        mean_conf  = sum(confs) / len(confs)

        # ── Clear frame counter ────────────────────────────
        if n_surv == 0:
            self._consecutive_clear += 1
        else:
            self._consecutive_clear = 0

        # ── LZ info ───────────────────────────────────────
        lz_safe    = best_lz is not None and best_lz.is_safe
        lz_score   = best_lz.lz_score   if best_lz else 0.0
        lz_bearing = best_lz.bearing    if best_lz else 0.0

        # Estimate distance in metres (altitude-based pixel scale)
        alt = getattr(telemetry, 'altitude', 80.0)
        px_per_m = 640 / (alt * 0.8)   # rough projection
        lz_dist_m = (best_lz.distance_px / px_per_m) \
                    if best_lz else 0.0

        # ── Hazard detection ──────────────────────────────
        fire_det     = n_fire > 0
        fire_below   = (fire_det and best_lz is not None
                        and best_lz.thermal_score < 0.3)

        # Visibility heuristic from wind speed
        wind = getattr(telemetry, 'wind_speed', 0.0)
        if wind > 25:   visibility = "POOR"
        elif wind > 15: visibility = "REDUCED"
        else:           visibility = "CLEAR"

        return MissionState(
            frame_id       = self._frame_id,
            timestamp      = time.time(),
            mission_elapsed= elapsed,

            battery        = getattr(telemetry, 'battery',   100.0),
            altitude       = getattr(telemetry, 'altitude',   80.0),
            speed          = getattr(telemetry, 'speed',       0.0),
            heading        = getattr(telemetry, 'heading',     0.0),
            wind_speed     = wind,
            latitude       = getattr(telemetry, 'latitude',    0.0),
            longitude      = getattr(telemetry, 'longitude',   0.0),
            gps_lock       = getattr(telemetry, 'gps_lock',  True),

            survivors      = survivors,
            n_survivors    = n_surv,
            n_critical     = n_crit,
            n_high         = n_high,
            n_distress     = n_dist,
            n_fires        = n_fire,
            any_alive      = any_alive,
            any_moving     = any_mov,
            any_isolated   = any_iso,
            max_confidence = max_conf,
            mean_confidence= mean_conf,

            lz_candidates  = lz_candidates,
            best_lz        = best_lz,
            lz_safe        = lz_safe,
            lz_score       = lz_score,
            lz_bearing     = lz_bearing,
            lz_distance_m  = round(lz_dist_m, 1),

            fire_detected  = fire_det,
            fire_below_lz  = fire_below,
            visibility     = visibility,

            consecutive_clear_frames = self._consecutive_clear,
            last_action    = self._last_action,
            action_history = list(self._action_history[-10:]),
            operator_override_count  = self._override_count,
            ground_team_alerted      = self._ground_alerted,
        )

    def record_action(self, action: str):
        self._last_action = action
        self._action_history.append(action)

    def record_override(self):
        self._override_count += 1

    def record_ground_alert(self):
        self._ground_alerted = True


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    state = MissionState(
        battery=72.0, n_survivors=3, n_critical=1,
        lz_safe=True, lz_score=0.78, fire_detected=False
    )
    vec = state.to_feature_vector()
    print(f"Feature vector shape: {vec.shape}")
    print(f"Feature vector: {vec}")
    print(f"\nbattery_ok: {state.battery_ok}")
    print(f"has_critical: {state.has_critical}")
    print(f"estimated range: {state.estimated_range_km:.1f} km")
