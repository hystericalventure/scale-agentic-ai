"""
Evaluation Parameter 3: Learning efficiency (sessions-to-mastery).

How many sessions does a student need before crossing a meaningful mastery
threshold (0.6)? This is a different, pedagogically meaningful question from
"final mastery" -- two conditions can reach the same final mastery at very
different speeds, and speed matters for real classroom pacing.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
import numpy as np
import pandas as pd

from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import train as train_zpd
from consent_agent import ConsentTrustAgent
from overload_agent import WINDOW
from full_system import train_overload_classifier, build_risk_features, rule_based_scaffold

MASTERY_THRESHOLD = 0.6
MAX_SESSIONS = 50  # 10 weeks x 5 sessions


def run_condition_with_trajectory(profiles, zpd_agent, overload_clf, feature_cols, condition, weeks=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (_, profile) in enumerate(profiles.iterrows()):
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + i)
        consent_agent = ConsentTrustAgent(profile)
        current_level = 0
        last_error_rate = None
        history = []
        session_of_threshold = None
        done = False
        session_idx = 0

        while not done:
            break_taken = False
            if condition in ("scale_full", "push_only") and len(history) >= WINDOW:
                feats = build_risk_features(history, feature_cols)
                risk = overload_clf.predict_proba(feats)[0, 1]
                if risk >= 0.5:
                    break_taken = True

            if condition == "rule_based":
                proposed_level = rule_based_scaffold(last_error_rate, current_level)
            else:
                state = env.discretize_state()
                action = zpd_agent.act(state, episode=10**9, rng=rng, greedy=True)
                proposed_level = SCAFFOLD_LEVELS[action]

            if condition == "scale_full":
                accepted, _ = consent_agent.decide(current_level, proposed_level, rng)
            else:
                accepted = True

            executed_level = proposed_level if accepted else current_level
            obs, reward, done, info = env.step(executed_level, break_taken=break_taken)

            if condition == "scale_full":
                consent_agent.update(accepted, reward)

            history.append(obs)
            last_error_rate = obs["error_rate"]
            current_level = executed_level
            session_idx += 1

            if session_of_threshold is None and info["mastery"] >= MASTERY_THRESHOLD:
                session_of_threshold = session_idx

        rows.append({
            "condition": condition,
            "student_id": profile["student_id"],
            "disability_type": profile["disability_type"],
            "sessions_to_mastery": session_of_threshold if session_of_threshold else MAX_SESSIONS + 1,
            "reached_threshold": session_of_threshold is not None,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print("Training ZPD agent...")
    zpd_agent, _ = train_zpd(train_profiles, weeks=10, seed=1)
    print("Training Overload classifier...")
    overload_clf, feature_cols = train_overload_classifier(train_profiles, zpd_agent)

    print(f"\nMeasuring sessions-to-mastery (threshold={MASTERY_THRESHOLD}) across 3 conditions x 3 seeds...")
    all_results = []
    for run_seed in [7, 218, 429]:
        for condition in ["scale_full", "rule_based", "push_only"]:
            df = run_condition_with_trajectory(test_profiles, zpd_agent, overload_clf, feature_cols, condition, weeks=10, seed=run_seed)
            df["run_seed"] = run_seed
            all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "learning_efficiency.csv"), index=False)

    summary = combined.groupby("condition").agg(
        mean_sessions_to_mastery=("sessions_to_mastery", "mean"),
        pct_reached_threshold=("reached_threshold", "mean"),
    ).round(2)
    summary["pct_reached_threshold"] = (summary["pct_reached_threshold"] * 100).round(1)
    print("\n=== Learning Efficiency (sessions to reach mastery >= 0.6) ===")
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, "learning_efficiency_summary.csv"))
    print("\nSaved to results/learning_efficiency.csv and learning_efficiency_summary.csv")
