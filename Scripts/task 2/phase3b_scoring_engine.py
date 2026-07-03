import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import setup_path
setup_path()
from phase3a_mission_state import MissionState, Action, ALL_ACTIONS


# ── Score Result ──────────────────────────────────────────

@dataclass
class ActionScore:
    action:      str
    total_score: float
    rule_score:  float
    ml_score:    float
    bonus_score: float
    vetoed:      bool  = False
    veto_reason: str   = ""


# ── Scoring Engine ────────────────────────────────────────

class ScoringEngine:
    """
    Scores all actions for a given mission state.
    Combines rule-based priors with learned ML weights.
    """

    # Action index mapping (must match RL trainer)
    ACTION_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}
    N_ACTIONS  = len(ALL_ACTIONS)
    N_FEATURES = 20  # matches state.to_feature_vector() length

    def __init__(self, weights_path=None):
        # Try to load trained weights
        self.W = self._load_weights(weights_path)
        # W shape: (N_ACTIONS, N_FEATURES)
        # Each row = weight vector for one action
        # Dot product with feature vector → raw ML score

    def _load_weights(self, path=None):
        """
        Loads trained weight matrix.
        Falls back to expert-initialised weights if no file found.
        """
        search = [path] if path else []
        search += list(Path("sar_data/models").rglob("scoring_weights.npy"))

        for p in search:
            if p and Path(p).exists():
                try:
                    W = np.load(str(p))
                    if W.shape == (self.N_ACTIONS, self.N_FEATURES):
                        print(f"[ScoringEngine] Weights loaded: {p}")
                        return W
                except Exception as e:
                    print(f"[ScoringEngine] Weight load failed: {e}")

        print("[ScoringEngine] Using expert-initialised weights")
        return self._expert_weights()

    def _expert_weights(self):
        """
        Hand-crafted initial weight matrix based on SAR expert knowledge.
        Each row is the weight vector for one action.
        Positive weight = feature encourages this action.
        Negative weight = feature discourages this action.

        Feature indices (from MissionState.to_feature_vector()):
          0=battery  1=altitude  2=n_survivors  3=n_critical
          4=n_fires  5=lz_score  6=lz_safe      7=fire_det
          8=fire_below_lz  9=max_conf  10=mean_conf
          11=any_alive  12=any_moving  13=any_isolated
          14=wind  15=clear_frames  16=gps_lock
          17=ground_alerted  18=n_distress  19=elapsed
        """
        W = np.zeros((self.N_ACTIONS, self.N_FEATURES))

        # ── LAND ─────────────────────────────────────────
        # Encouraged by: survivors, LZ safe, alive signal,
        #                high confidence, good battery, low wind
        # Discouraged by: fire, fire below LZ, bad LZ score
        idx = self.ACTION_IDX[Action.LAND]
        W[idx] = [ 0.4,  0.0,  1.5,  2.0,  -1.5,  2.0,  3.0,
                  -2.0, -3.0,  1.2,  1.0,   1.5,  0.5,  0.5,
                  -1.0, -0.5,  0.5, -0.5,   0.8,  0.0]

        # ── HOVER_DROP ───────────────────────────────────
        # Encouraged by: survivors (can't land), fire nearby,
        #                bad LZ, alive signal, high wind
        # Discouraged by: safe LZ (land instead), low survivors
        idx = self.ACTION_IDX[Action.HOVER_DROP]
        W[idx] = [ 0.2,  0.5,  1.2,  1.5,   1.0, -1.5, -2.0,
                   1.5,  1.0,  0.8,  0.8,   1.2,  0.0,  0.8,
                   1.0, -0.3,  0.3,  0.0,   0.5,  0.0]

        # ── RELAY ────────────────────────────────────────
        # Encouraged by: distress signals, can't land/drop,
        #                GPS lock, ground team not yet alerted
        # Discouraged by: already alerted ground team
        idx = self.ACTION_IDX[Action.RELAY]
        W[idx] = [ 0.1,  0.0,  0.8,  0.5,   0.3, -0.5, -0.5,
                   0.5,  0.3,  0.5,  0.4,   0.3,  0.5,  1.0,
                  -0.2,  0.0,  1.5, -1.5,   2.0,  0.0]

        # ── HOLD ─────────────────────────────────────────
        # Encouraged by: no survivors, scanning, good battery
        # Discouraged by: critical survivors detected
        idx = self.ACTION_IDX[Action.HOLD]
        W[idx] = [ 0.5,  0.0, -1.5, -2.5,  -0.5, -0.3,  0.0,
                  -0.3, -0.3, -0.5, -0.5,  -1.0, -0.3, -0.2,
                  -0.2,  1.5,  0.3,  0.0,  -0.5,  0.0]

        # ── LOITER ───────────────────────────────────────
        # Encouraged by: no safe LZ found, survivors present,
        #                waiting for ground team
        # Discouraged by: low battery, safe LZ available
        idx = self.ACTION_IDX[Action.LOITER]
        W[idx] = [-0.5,  0.3,  1.0,  0.8,   0.2, -1.5, -2.0,
                   0.5,  0.5,  0.5,  0.3,   0.8,  0.0,  0.3,
                   0.3,  0.3,  0.3,  1.0,   0.3,  0.0]

        # ── RTB ──────────────────────────────────────────
        # Encouraged by: low battery, mission elapsed, no survivors
        # Discouraged by: critical survivors, high confidence detections
        idx = self.ACTION_IDX[Action.RTB]
        W[idx] = [-2.5,  0.0, -1.5, -2.0,  -0.5, -0.3,  0.0,
                   0.0,  0.0, -0.8, -0.6,  -1.5, -0.5, -0.3,
                  -0.3,  0.5, -0.2,  0.5,  -0.5,  1.5]

        # ── ESCALATE ─────────────────────────────────────
        # Encouraged by: many critical, beyond capability,
        #                long mission, GPS loss
        # Discouraged by: simple scenarios
        idx = self.ACTION_IDX[Action.ESCALATE]
        W[idx] = [-0.5,  0.0,  2.0,  2.5,   2.0, -0.5,  0.0,
                   2.0,  2.0,  0.5,  0.5,   1.0, -0.3,  1.0,
                   1.5,  0.3, -1.0,  0.0,   1.5,  1.5]

        return W

    def save_weights(self, path=None):
        """Saves current weight matrix to file."""
        save_path = path or Path("sar_data/models/scoring_weights.npy")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), self.W)
        print(f"[ScoringEngine] Weights saved → {save_path}")

    # ── Scoring ───────────────────────────────────────────

    def score_all(self, state: MissionState) -> List[ActionScore]:
        """
        Computes scores for all actions and returns sorted list.
        Applies: rule scores + ML scores + bonuses + vetos.
        """
        fv = state.to_feature_vector()   # (20,) float32

        scores = []
        for action in ALL_ACTIONS:
            idx = self.ACTION_IDX[action]

            # ── Rule-based score ──────────────────────────
            rule_sc = self._rule_score(action, state)

            # ── ML score ─────────────────────────────────
            # Linear combination of features with trained weights
            # then sigmoid to squash to (0, 1)
            raw = float(np.dot(self.W[idx], fv))
            ml_sc = self._sigmoid(raw)

            # ── Context bonus ─────────────────────────────
            bonus = self._context_bonus(action, state)

            # ── Total ─────────────────────────────────────
            total = (rule_sc * 0.50       # rules = 50% weight
                   + ml_sc   * 0.35       # ML    = 35% weight
                   + bonus   * 0.15)      # bonus = 15% weight

            scores.append(ActionScore(
                action      = action,
                total_score = round(total, 4),
                rule_score  = round(rule_sc, 4),
                ml_score    = round(ml_sc,   4),
                bonus_score = round(bonus,   4),
            ))

        # Sort descending by total score
        scores.sort(key=lambda s: -s.total_score)
        return scores

    def _sigmoid(self, x):
        """Sigmoid activation: maps any real number to (0, 1)."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _rule_score(self, action: str, state: MissionState) -> float:
        """
        Deterministic rule-based score for an action.
        Returns float 0.0–1.0.

        These encode expert SAR knowledge as explicit scoring rules,
        rather than if/else branches — so they can be blended with
        the ML score rather than hard-overriding it.
        """
        s = state

        if action == Action.LAND:
            # Ideal LAND conditions:
            score  = 0.0
            score += 0.35 * s.lz_score              # LZ quality
            score += 0.20 if s.lz_safe else 0.0
            score += 0.20 * min(1.0, s.n_critical / 3.0)
            score += 0.10 * s.max_confidence
            score -= 0.20 if s.fire_below_lz else 0.0
            score -= 0.15 if s.wind_speed > 20 else 0.0
            score -= 0.10 if not s.battery_ok else 0.0
            return max(0.0, min(1.0, score))

        elif action == Action.HOVER_DROP:
            score  = 0.0
            score += 0.25 * min(1.0, s.n_survivors / 4.0)
            score += 0.20 if not s.lz_safe else 0.0
            score += 0.15 if s.fire_detected else 0.0
            score += 0.10 * min(1.0, s.n_critical / 2.0)
            score += 0.10 if s.any_alive else 0.0
            score -= 0.10 if s.lz_safe else 0.0   # land instead
            return max(0.0, min(1.0, score))

        elif action == Action.RELAY:
            score  = 0.0
            score += 0.30 if not s.ground_team_alerted else 0.0
            score += 0.20 * min(1.0, s.n_distress / 2.0)
            score += 0.15 * s.max_confidence
            score += 0.10 if s.gps_lock else 0.0
            score += 0.10 if s.any_isolated else 0.0
            score -= 0.20 if s.ground_team_alerted else 0.0
            return max(0.0, min(1.0, score))

        elif action == Action.HOLD:
            score  = 0.3   # always some base chance to hold & scan
            score += 0.30 * (s.consecutive_clear_frames / 20.0)
            score -= 0.20 * min(1.0, s.n_survivors / 3.0)
            score -= 0.15 * s.max_confidence
            return max(0.0, min(1.0, score))

        elif action == Action.LOITER:
            score  = 0.0
            score += 0.30 if s.has_survivors and not s.lz_safe else 0.0
            score += 0.20 if s.ground_team_alerted else 0.0
            score += 0.15 * min(1.0, s.n_critical / 2.0)
            score -= 0.20 if s.battery_low else 0.0
            score -= 0.15 if s.lz_safe else 0.0
            return max(0.0, min(1.0, score))

        elif action == Action.RTB:
            score  = 0.0
            if s.battery_critical:  score += 0.70
            elif s.battery_low:     score += 0.35
            score += 0.20 * min(1.0, s.mission_elapsed / 1800.0)
            score -= 0.30 if s.has_critical else 0.0
            score += 0.10 if not s.has_survivors else 0.0
            return max(0.0, min(1.0, score))

        elif action == Action.ESCALATE:
            score  = 0.0
            score += 0.25 if s.n_critical >= 3 else 0.0
            score += 0.20 if s.n_fires >= 2 else 0.0
            score += 0.15 if not s.gps_lock else 0.0
            score += 0.15 if s.mission_elapsed > 1500 else 0.0
            score += 0.10 if s.wind_speed > 28 else 0.0
            return max(0.0, min(1.0, score))

        return 0.0

    def _context_bonus(self, action: str, state: MissionState) -> float:
        """
        Situational bonuses that apply in specific scenarios.
        These encode rare but important edge cases.
        """
        bonus = 0.0

        if action == Action.LAND:
            # Bonus: clear scene, high confidence, stable conditions
            if (state.lz_score > 0.80 and state.n_critical > 0
                    and not state.fire_detected
                    and state.wind_speed < 15):
                bonus += 0.3

        elif action == Action.HOVER_DROP:
            # Bonus: survivor alive but completely surrounded by fire
            if state.any_alive and state.n_fires >= 2:
                bonus += 0.4

        elif action == Action.RELAY:
            # Bonus: isolated survivor with distress signal
            if state.any_isolated and state.n_distress > 0:
                bonus += 0.3

        elif action == Action.LOITER:
            # Bonus: ground team alerted, waiting for ETA
            if state.ground_team_alerted and state.lz_safe:
                bonus += 0.3

        elif action == Action.ESCALATE:
            # Bonus: complex multi-hazard scene beyond single-drone scope
            if (state.n_critical >= 2 and state.n_fires >= 1
                    and not state.lz_safe):
                bonus += 0.5

        return min(1.0, bonus)

    def update_weights_from_feedback(
        self,
        state: MissionState,
        chosen_action: str,
        reward: float,
        lr: float = 0.01,
    ):
        """
        Online learning: updates weights based on operator feedback.

        When an operator accepts a recommendation:   reward = +1.0
        When an operator overrides a recommendation: reward = -0.5

        Uses a simple gradient update (policy gradient approximation):
          W[action] += lr * reward * features

        This gradually shifts weights toward actions the operator approves.
        """
        fv  = state.to_feature_vector()
        idx = self.ACTION_IDX.get(chosen_action)
        if idx is None:
            return

        # Gradient update
        self.W[idx] += lr * reward * fv
        # Clip weights to prevent explosion
        self.W[idx] = np.clip(self.W[idx], -5.0, 5.0)


# ── Standalone test ───────────────────────────────────────
if __name__ == "__main__":
    engine = ScoringEngine()
    state  = MissionState(
        battery=72.0, n_survivors=2, n_critical=1,
        any_alive=True, lz_safe=True, lz_score=0.78,
        fire_detected=False, max_confidence=0.91,
        mean_confidence=0.84, wind_speed=10.0,
        gps_lock=True,
    )

    scores = engine.score_all(state)

    print("\n📊 Action Scores (scenario: 2 survivors, safe LZ)\n")
    print(f"{'Action':<14} {'Total':>7} {'Rules':>7} "
          f"{'ML':>7} {'Bonus':>7}")
    print("-" * 45)
    for s in scores:
        marker = " ◄" if scores[0].action == s.action else ""
        print(f"{s.action:<14} {s.total_score:>7.4f} "
              f"{s.rule_score:>7.4f} {s.ml_score:>7.4f} "
              f"{s.bonus_score:>7.4f}{marker}")
    print(f"\n✅ Best action: {scores[0].action}")
