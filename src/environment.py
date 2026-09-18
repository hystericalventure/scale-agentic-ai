"""
Phase 1b: StudentEnv -- the closed-loop simulation environment.

Each student is a small stateful process. On every session:
  - the ZPD Scaffolding Agent proposes a scaffold_level (0=independent,
    1=partial hint, 2=worked example) and the environment scores the
    resulting mastery_gain / frustration / behavioral signals
  - latent ground-truth "overload" is generated from fatigue + sensory
    sensitivity + scaffold mismatch, used later ONLY to evaluate the
    Overload Prediction Agent (the agent itself never sees this directly --
    it only sees the behavioral signals, same as a real system would)

This file has no dependency on any specific agent, so it's shared
infrastructure for Phase 2-5.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np

SCAFFOLD_LEVELS = [0, 1, 2]  # independent, partial hint, worked example
SESSIONS_PER_WEEK = 5


class StudentEnv:
    def __init__(self, profile_row, weeks=10, seed=0):
        self.p = profile_row
        self.rng = np.random.default_rng(seed)
        self.weeks = weeks
        self.total_sessions = weeks * SESSIONS_PER_WEEK

        self.mastery = float(profile_row["baseline_mastery"])
        self.fatigue = 0.0
        self.trust = float(profile_row["consent_disposition"])  # starts at disposition
        self.session_idx = 0
        self.recent_errors = []  # rolling window for state discretization

    def _ideal_scaffold(self):
        # low mastery -> needs more scaffolding
        if self.mastery < 0.35:
            return 2
        elif self.mastery < 0.65:
            return 1
        return 0

    def discretize_state(self):
        mastery_bin = min(int(self.mastery * 5), 4)
        fatigue_bin = min(int(self.fatigue * 3), 2)
        recent_err = np.mean(self.recent_errors[-3:]) if self.recent_errors else 0.3
        err_bin = min(int(recent_err * 3), 2)
        return (mastery_bin, fatigue_bin, err_bin)

    def step(self, scaffold_level, break_taken=False):
        """Advance one session given the ZPD agent's chosen scaffold_level."""
        ideal = self._ideal_scaffold()
        mismatch = abs(scaffold_level - ideal) / 2.0  # normalized 0-1

        # fatigue dynamics: rises within week, partially resets on break or weekend
        if break_taken:
            self.fatigue = max(0.0, self.fatigue - 0.35)
        is_week_start = (self.session_idx % SESSIONS_PER_WEEK) == 0
        if is_week_start:
            self.fatigue *= 0.4  # weekend recovery
        self.fatigue = np.clip(self.fatigue + 0.12 + 0.05 * self.p["sensory_sensitivity"], 0, 1)

        # behavioral signals
        base_latency = 0.3 + 0.4 * self.fatigue + 0.3 * self.p["sensory_sensitivity"]
        response_latency = np.clip(base_latency + 0.5 * mismatch + self.rng.normal(0, 0.05), 0, 2)

        base_error = np.clip(0.6 - self.mastery, 0.05, 0.6)
        error_rate = np.clip(base_error + 0.4 * mismatch + self.rng.normal(0, 0.04), 0, 1)
        self.recent_errors.append(error_rate)

        hesitation = np.clip(0.2 + 0.5 * self.fatigue * self.p["sensory_sensitivity"] + self.rng.normal(0, 0.03), 0, 1)

        # latent overload probability (ground truth, hidden from agents)
        # calibrated so well-matched scaffolding keeps overload occasional (~15-20%)
        # while poor scaffolding materially raises risk -- this gap is what the
        # Overload Prediction Agent and the scaffolding policy are jointly evaluated on
        overload_logit = (1.5 * self.fatigue + 1.3 * self.p["sensory_sensitivity"]
                           + 2.5 * mismatch - 3.0)
        overload_prob = 1 / (1 + np.exp(-overload_logit))
        true_overload = int(self.rng.random() < overload_prob)

        # learning dynamics: best mastery gain near-zero mismatch, penalized both
        # for under- and over-scaffolding (dependency effect for over-scaffolding)
        learning_rate = 0.03 * (1 - mismatch) * (1 - 0.3 * self.fatigue)
        over_scaffold_penalty = 0.3 if scaffold_level > ideal else 0.0
        mastery_gain = max(0.0, learning_rate * (1 - over_scaffold_penalty))
        self.mastery = np.clip(self.mastery + mastery_gain, 0, 1)

        # frustration: high on under-scaffolding + high error, low otherwise
        frustration = np.clip(0.6 * mismatch * (scaffold_level < ideal) + 0.4 * error_rate, 0, 1)

        reward = mastery_gain * 10 - frustration * 1.5

        obs = {
            "response_latency": response_latency,
            "error_rate": error_rate,
            "hesitation": hesitation,
            "fatigue_proxy": self.fatigue,  # in a real system this would be inferred, not observed directly;
                                             # kept here for the Overload Agent's feature engineering step
        }
        info = {"true_overload": true_overload, "mismatch": mismatch, "mastery": self.mastery}
        self.session_idx += 1
        done = self.session_idx >= self.total_sessions
        return obs, reward, done, info
