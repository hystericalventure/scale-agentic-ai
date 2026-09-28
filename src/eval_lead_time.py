"""
Evaluation Parameter 4: Overload prediction lead time.

The agent forecasts one session ahead by design (Section 4.2). This script
measures something additional: for true overload events, how many consecutive
PRIOR sessions already showed elevated risk (>=0.5)? This quantifies whether
the early-warning signal builds up gradually (useful, actionable lead time)
or appears only at the last moment (less useful in practice).
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import train as train_zpd
from overload_agent import collect_dataset, build_features, WINDOW, collect_trajectory

RISK_THRESHOLD = 0.5
MAX_LOOKBACK = 4


def measure_lead_time(profiles, zpd_agent, clf, weeks=10, seed=0):
    rng = np.random.default_rng(seed)
    lead_times = []

    def learned_policy(state, r):
        return zpd_agent.act(state, episode=10**9, rng=r, greedy=True)

    for i, (_, profile) in enumerate(profiles.iterrows()):
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + i)
        traj = collect_trajectory(env, learned_policy, rng)
        X, y = build_features(traj)
        if len(X) == 0:
            continue
        risk_scores = clf.predict_proba(X)[:, 1]

        # X[t] corresponds to traj rows [WINDOW ... len(traj)-1], predicting overload at that row
        for t in range(len(y)):
            if y[t] == 1:  # true overload event at this (shifted) position
                lead = 0
                look = t - 1
                while look >= 0 and risk_scores[look] >= RISK_THRESHOLD and lead < MAX_LOOKBACK:
                    lead += 1
                    look -= 1
                lead_times.append(lead)

    return lead_times


if __name__ == "__main__":
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print("Training ZPD agent...")
    zpd_agent, _ = train_zpd(train_profiles, weeks=10, seed=1)
    print("Training Overload classifier...")
    X_train, y_train, _ = collect_dataset(train_profiles, zpd_agent, weeks=10, seed=2)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                  class_weight="balanced", random_state=0)
    clf.fit(X_train, y_train)

    print("Measuring early-warning lead time on held-out students...")
    lead_times = measure_lead_time(test_profiles, zpd_agent, clf, weeks=10, seed=3)
    lead_times = np.array(lead_times)

    print(f"\n=== Overload Prediction Lead Time (n={len(lead_times)} true overload events) ===")
    print(f"Mean consecutive prior sessions with elevated risk: {lead_times.mean():.2f}")
    print(f"Events with 0 prior warning (only caught at the event itself): {(lead_times == 0).mean():.1%}")
    print(f"Events with >=1 session of prior warning: {(lead_times >= 1).mean():.1%}")
    print(f"Events with >=2 sessions of prior warning: {(lead_times >= 2).mean():.1%}")

    pd.DataFrame({"lead_time_sessions": lead_times}).to_csv(
        os.path.join(RESULTS_DIR, "overload_lead_time.csv"), index=False)
    print("\nSaved to results/overload_lead_time.csv")
