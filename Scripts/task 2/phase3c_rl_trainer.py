import numpy as np
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict

from phase3a_mission_state import MissionState, Action, ALL_ACTIONS
from phase3b_scoring_engine import ScoringEngine


# ── SAR Simulation Environment ────────────────────────────

class SimSAREnvironment:

    SCENARIO_TYPES = [
        "routine_scan", "single_survivor", "multiple_crit",
        "fire_rescue",  "low_battery",     "comms_relay",
        "beyond_scope"
    ]

    def reset(self):
        weights = {
            "routine_scan":1,
            "single_survivor": 2,
            "multiple_crit": 3,
            "fire_rescue": 4,
            "low_battery": 2,
            "comms_relay": 3,
            "beyond_scope": 4
        }

        pool = []
        for k, w in weights.items():
            pool.extend([k]*w)
            
        scenario = random.choice(self.SCENARIO_TYPES)
        return self._make_scenario(scenario)

    def _make_scenario(self, scenario_type: str) -> MissionState:
        """Creates a MissionState matching the scenario type."""

        if scenario_type == "routine_scan":
            return MissionState(
                battery        = random.uniform(60, 95),
                n_survivors    = 0, n_critical = 0, n_fires = 0,
                lz_safe        = True,
                lz_score       = random.uniform(0.6, 0.9),
                fire_detected  = False,
                max_confidence = 0.0, mean_confidence = 0.0,
                wind_speed     = random.uniform(0, 15),
                gps_lock       = True,
                consecutive_clear_frames = random.randint(5, 20),
                any_alive      = False,
            )

        elif scenario_type == "single_survivor":
            conf = random.uniform(0.70, 0.95)
            return MissionState(
                battery        = random.uniform(50, 90),
                n_survivors    = 1, n_critical = 1, n_fires = 0,
                n_distress     = random.choice([0, 0, 1]),
                lz_safe        = True,
                lz_score       = random.uniform(0.60, 0.85),
                fire_detected  = False,
                max_confidence = conf, mean_confidence = conf,
                wind_speed     = random.uniform(0, 18),
                gps_lock       = True,
                any_alive      = True,
                any_moving     = random.choice([True, False]),
                any_isolated   = random.choice([True, False]),
            )

        elif scenario_type == "multiple_crit":
            n = random.randint(2, 4)
            conf = random.uniform(0.65, 0.92)
            return MissionState(
                battery        = random.uniform(40, 80),
                n_survivors    = n,
                n_critical     = random.randint(1, n),
                n_fires        = 0,
                lz_safe        = random.choice([True, True, False]),
                lz_score       = random.uniform(0.45, 0.80),
                fire_detected  = False,
                max_confidence = conf, mean_confidence = conf * 0.9,
                wind_speed     = random.uniform(5, 22),
                gps_lock       = True,
                any_alive      = True,
                any_moving     = random.choice([True, False]),
                any_isolated   = True,
            )

        elif scenario_type == "fire_rescue":
            conf = random.uniform(0.75, 0.95)
            return MissionState(
                battery        = random.uniform(45, 85),
                n_survivors    = random.randint(1, 3),
                n_critical     = random.randint(1, 2),
                n_fires        = random.randint(1, 3),
                lz_safe        = False,
                lz_score       = random.uniform(0.10, 0.45),
                fire_detected  = True,
                fire_below_lz  = random.choice([True, False]),
                max_confidence = conf, mean_confidence = conf * 0.85,
                wind_speed     = random.uniform(8, 25),
                gps_lock       = True,
                any_alive      = True,
                any_isolated   = True,
            )

        elif scenario_type == "low_battery":
            bat = random.uniform(10, 28)
            conf = random.uniform(0.60, 0.88)
            return MissionState(
                battery        = bat,
                n_survivors    = random.randint(0, 2),
                n_critical     = random.randint(0, 1),
                n_fires        = 0,
                lz_safe        = random.choice([True, False]),
                lz_score       = random.uniform(0.40, 0.75),
                fire_detected  = False,
                max_confidence = conf, mean_confidence = conf,
                wind_speed     = random.uniform(0, 20),
                gps_lock       = True,
                mission_elapsed= random.uniform(900, 1800),
            )

        elif scenario_type == "comms_relay":
            conf = random.uniform(0.55, 0.80)
            return MissionState(
                battery        = random.uniform(50, 85),
                n_survivors    = random.randint(1, 2),
                n_distress     = random.randint(1, 3),
                n_critical     = 0,
                n_fires        = 0,
                lz_safe        = random.choice([True, False]),
                lz_score       = random.uniform(0.40, 0.70),
                fire_detected  = False,
                max_confidence = conf, mean_confidence = conf,
                wind_speed     = random.uniform(0, 15),
                gps_lock       = random.choice([True, True, False]),
                any_isolated   = True,
                ground_team_alerted = random.choice([False, False, True]),
            )

        else:  # beyond_scope
            return MissionState(
                battery        = random.uniform(30, 70),
                n_survivors    = random.randint(4, 8),
                n_critical     = random.randint(3, 5),
                n_fires        = random.randint(2, 4),
                lz_safe        = False,
                lz_score       = random.uniform(0.05, 0.35),
                fire_detected  = True,
                fire_below_lz  = True,
                max_confidence = random.uniform(0.7, 0.95),
                mean_confidence= random.uniform(0.6, 0.85),
                wind_speed     = random.uniform(18, 35),
                gps_lock       = random.choice([True, False]),
                any_alive      = True,
                any_isolated   = True,
                mission_elapsed= random.uniform(600, 1800),
            )

    def compute_reward(
        self,
        action: str,
        state: MissionState,
    ) -> float:
        
        r = 0.5

        # 1. Battery critical → must RTB
        if state.battery_critical:
            return 8.0 if action == Action.RTB else -10.0

        # 2. GPS loss with survivors → must RELAY
        if not state.gps_lock and state.has_survivors:
            return 8.0 if action == Action.RELAY else -6.0

        # 3. Mass casualty → must ESCALATE (only for truly overwhelming scenarios)
        if state.n_critical >= 4 or state.n_fires >= 3:
            return 10.0 if action == Action.ESCALATE else -8.0

        # 3b. Moderate mass casualty → ESCALATE preferred but HOVER_DROP acceptable
        if state.n_critical >= 3 and state.n_fires >= 2:
            if action == Action.ESCALATE:   return 8.0
            if action == Action.HOVER_DROP: return 3.0
            return -5.0

        # 4. Unsafe LZ with survivors → must HOVER_DROP
        if state.has_survivors and not state.lz_safe:
            return 9.0 if action == Action.HOVER_DROP else -6.0

        # ── LAND rewards ─────────────────────────────────
        if action == Action.LAND:
            if state.lz_safe and state.has_survivors:
                r += 12.0
                if state.has_critical:
                    r += 3.0   # bonus for rescuing critical
                if state.any_alive:
                    r += 2.0   # bonus for confirmed alive
            elif not state.lz_safe:
                r -= 5.0       # dangerous: landing on unsafe LZ
            elif state.fire_below_lz:
                r -= 8.0       # very dangerous: fire below
            elif not state.has_survivors:
                r -= 2.0       # pointless landing

        # ── HOVER_DROP rewards ────────────────────────────
        elif action == Action.HOVER_DROP:
            if state.has_survivors and not state.lz_safe:
                r += 7.0       # correct: can't land, drop supplies
                if state.any_alive:
                    r += 2.0
            elif state.has_survivors and state.lz_safe:
                r -= 1.0       # suboptimal: should land instead
            elif not state.has_survivors:
                r -= 2.0       # wasteful drop with no survivors

        # ── RELAY rewards ─────────────────────────────────
        elif action == Action.RELAY:
            if state.has_survivors and not state.ground_team_alerted:
                r += 6.0
                if state.n_distress > 0:
                    r += 2.0
                if state.gps_lock:
                    r += 1.5   # accurate relay
                else:
                    r -= 1.0   # unreliable without GPS
            elif state.ground_team_alerted:
                r -= 1.5       # already alerted, redundant

        # ── HOLD rewards ──────────────────────────────────
        elif action == Action.HOLD:
            if not state.has_survivors:
                r += 0.5       # correct: scanning with no detections
            elif state.has_critical:
                r -= 4.0       # terrible: holding while critical alive
            elif state.has_survivors:
                r -= 1.5       # bad: holding when action needed

        # ── LOITER rewards ────────────────────────────────
        elif action == Action.LOITER:
            if (state.has_survivors and not state.lz_safe
                    and state.ground_team_alerted):
                r += 3.0       # correct: waiting for ground team
            elif state.lz_safe and state.has_survivors:
                r -= 1.0       # should land instead
            elif not state.has_survivors:
                r -= 1.0       # pointless loiter

        # ── RTB rewards ───────────────────────────────────
        elif action == Action.RTB:
            if state.battery_low and not state.has_critical:
                r += 3.0       # responsible: low battery, no critical
            elif state.battery_ok and state.has_critical:
                r -= 8.0       # terrible: abandoning critical survivors
            elif state.battery_ok:
                r -= 1.0       # premature RTB

        # ── ESCALATE rewards ──────────────────────────────
        elif action == Action.ESCALATE:
            beyond_scope = (state.n_critical >= 3
                           or state.n_fires >= 2
                           or not state.gps_lock)
            if beyond_scope:
                r += 4.0       # correct: recognising limits
            else:
                r -= 2.0       # premature escalation

        return r


# ── REINFORCE Trainer ─────────────────────────────────────

class RLTrainer:

    def __init__(
        self,
        n_episodes: int = 2000,
        lr:         float = 0.005,
        gamma:      float = 0.95,    # discount factor (unused in 1-step)
        entropy_coef: float = 0.00198,  # entropy regularisation
    ):
        self.n_episodes   = n_episodes
        self.lr           = lr
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.env          = SimSAREnvironment()
        self.engine       = ScoringEngine()

        # Training metrics
        self.episode_rewards  = []
        self.episode_actions  = []
        self.action_counts    = {a: 0 for a in ALL_ACTIONS}

    def _softmax(self, x):
        """Numerically stable softmax."""
        x_shifted = x - np.max(x)
        e_x = np.exp(np.clip(x_shifted, -20, 20))
        return e_x / e_x.sum()

    def train(self, verbose=True):
        
        print(f"\n{'='*55}")
        print(f" RL Training — REINFORCE ({self.n_episodes} episodes)")
        print(f"{'='*55}")
        print(f"  lr={self.lr}  entropy_coef={self.entropy_coef}")
        print(f"  Scenarios: {SimSAREnvironment.SCENARIO_TYPES}\n")

        W = self.engine.W.copy()  # (N_ACTIONS, N_FEATURES)
        N_ACTIONS  = self.engine.N_ACTIONS
        ACTION_IDX = self.engine.ACTION_IDX
        t_start    = time.time()

        for ep in range(self.n_episodes):

            # ── 1. Sample scenario ────────────────────────
            state = self.env.reset()
            fv    = state.to_feature_vector()  # (20,)

            # ── 2. Compute action probabilities ───────────
            # logits = W @ fv  shape: (N_ACTIONS,)
            logits = W @ fv
            probs  = self._softmax(logits)

            # ── 3. Sample action ──────────────────────────
            action_idx = np.random.choice(N_ACTIONS, p=probs)
            action     = ALL_ACTIONS[action_idx]

            # ── 4. Compute reward ─────────────────────────
            reward = self.env.compute_reward(action, state)

            

            one_hot = np.zeros(N_ACTIONS)
            one_hot[action_idx] = 1.0
            grad    = (one_hot - probs)  # shape: (N_ACTIONS,)

            
            W += self.lr * reward * np.outer(grad, fv)

            
            log_probs = np.log(probs + 1e-10)
            entropy_grad = -(1 + log_probs)   # ∇H w.r.t. logits
            W += self.entropy_coef * np.outer(entropy_grad, fv)

            # Clip weights to prevent explosion
            W = np.clip(W, -8.0, 8.0)

            # ── Track metrics ─────────────────────────────
            self.episode_rewards.append(reward)
            self.episode_actions.append(action)
            self.action_counts[action] += 1

            if verbose and (ep + 1) % 200 == 0:
                recent = self.episode_rewards[-200:]
                avg_r  = np.mean(recent)
                print(f"  Episode {ep+1:>5}/{self.n_episodes}  "
                      f"avg_reward={avg_r:>6.2f}  "
                      f"last={reward:>6.2f}  "
                      f"action={action}")

        # Update engine weights
        self.engine.W = W
        t_elapsed = time.time() - t_start

        self._print_summary(t_elapsed)
        return self.engine

    def _print_summary(self, elapsed):
        recent = self.episode_rewards[-500:]
        print(f"\n{'='*55}")
        print(f"  ✅ RL TRAINING COMPLETE")
        print(f"{'='*55}")
        print(f"  Duration       : {elapsed:.1f}s")
        print(f"  Final avg reward (last 500): {np.mean(recent):.3f}")
        print(f"  Min reward     : {min(self.episode_rewards):.2f}")
        print(f"  Max reward     : {max(self.episode_rewards):.2f}")
        print(f"\n  Action distribution:")
        total = sum(self.action_counts.values())
        for a, c in sorted(self.action_counts.items(),
                           key=lambda x: -x[1]):
            bar = "█" * int(c / total * 30)
            print(f"    {a:<14}: {c:>5} ({c/total*100:.1f}%) {bar}")

    def save(self, path=None):
        """Saves trained weights and training metrics."""
        base = Path("C:/Users/User/Documents/internship/IISC/Project 1/sar_data/models")
        base.mkdir(parents=True, exist_ok=True)

        weight_path = path or base / "scoring_weights.npy"
        self.engine.save_weights(weight_path)

        # Save training metrics
        metrics = {
            "n_episodes":   self.n_episodes,
            "final_avg_reward": float(np.mean(self.episode_rewards[-500:])),
            "action_counts": self.action_counts,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (base / "rl_training_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        print(f"  📄 Metrics saved → {base / 'rl_training_metrics.json'}")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="SAR Decision Engine RL Trainer"
    )
    parser.add_argument("--episodes", type=int, default=2000,
        help="Number of training episodes (default: 2000)")
    parser.add_argument("--lr",       type=float, default=0.005,
        help="Learning rate (default: 0.005)")
    parser.add_argument("--quick",    action="store_true",
        help="Quick run: 200 episodes")
    args = parser.parse_args()

    episodes = 200 if args.quick else args.episodes

    trainer = RLTrainer(n_episodes=episodes, lr=args.lr)
    trained_engine = trainer.train(verbose=True)
    trainer.save()

    # Quick verification: test trained engine on a scenario
    print("\n🔬 Verification on test scenario:")
    test_state = MissionState(
        battery=65.0, n_survivors=2, n_critical=1,
        any_alive=True, lz_safe=True, lz_score=0.72,
        fire_detected=False, max_confidence=0.88,
        wind_speed=12.0, gps_lock=True,
    )
    scores = trained_engine.score_all(test_state)
    print(f"  Best action: {scores[0].action} "
          f"(score={scores[0].total_score:.4f})")