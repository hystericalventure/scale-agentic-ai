"""
Phase 6: Full-system integration.

Combines all four agents into ONE closed loop per student per session:
  - Overload Agent scores proactive risk from the last WINDOW sessions'
    raw behavioral signals -> if risk is high, triggers a preemptive break
    (break_taken=True) BEFORE the scaffold decision for that session
  - ZPD Agent proposes a scaffold level from the (now updated) state
  - Consent Agent decides accept/reject; rejection falls back to current level
  - Teacher Co-Pilot flags students at the end of the run

Three conditions compared (the paper's headline table):
  1. SCALE (full)      -- all four agents active, consent-gated
  2. Rule-based baseline -- simple heuristic scaffold rule (no learning,
     no consent step, no proactive breaks) -- closest analogue to a
     non-agentic adaptive system / ADAPT's "static planner" comparator
  3. Push-only (no consent) -- learned ZPD + proactive breaks, but every
     proposal auto-executes -- isolates the specific contribution of the
     Consent & Trust Agent
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import train as train_zpd
from consent_agent import ConsentTrustAgent
from overload_agent import collect_dataset, WINDOW
from teacher_copilot_agent import flag_for_attention

OVERLOAD_RISK_THRESHOLD = 0.5


def train_overload_classifier(train_profiles, zpd_agent, seed=2):
    X_train, y_train, _ = collect_dataset(train_profiles, zpd_agent, weeks=10, seed=seed)
    clf = RandomForestClassifier(n_estimators=60, max_depth=6, min_samples_leaf=5,
                                  class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf, X_train.columns.tolist()


def build_risk_features(history, feature_cols):
    feats = ["response_latency", "error_rate", "hesitation", "fatigue_proxy"]
    window = history[-WINDOW:]
    row = {}
    for f in feats:
        vals = [h[f] for h in window]
        row[f"{f}_mean"] = sum(vals) / len(vals)
        row[f"{f}_trend"] = vals[-1] - vals[0]
    return np.array([[row[c] for c in feature_cols]])


def rule_based_scaffold(last_error_rate, current_level):
    if last_error_rate is None:
        return 1
    if last_error_rate > 0.4:
        return min(current_level + 1, 2)
    elif last_error_rate < 0.15:
        return max(current_level - 1, 0)
    return current_level


def run_condition(profiles, zpd_agent, overload_clf, feature_cols, condition, weeks=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (_, profile) in enumerate(profiles.iterrows()):
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + i)
        consent_agent = ConsentTrustAgent(profile)
        current_level = 0
        last_error_rate = None
        history = []
        total_reward, overload_count, accepts, proposals, breaks_triggered = 0.0, 0, 0, 0, 0
        done = False

        while not done:
            # --- Overload Agent: proactive break decision ---
            break_taken = False
            if condition in ("scale_full", "push_only") and len(history) >= WINDOW:
                feats = build_risk_features(history, feature_cols)
                risk = overload_clf.predict_proba(feats)[0, 1]
                if risk >= OVERLOAD_RISK_THRESHOLD:
                    break_taken = True
                    breaks_triggered += 1

            # --- ZPD / rule-based scaffold proposal ---
            if condition == "rule_based":
                proposed_level = rule_based_scaffold(last_error_rate, current_level)
            else:
                state = env.discretize_state()
                action = zpd_agent.act(state, episode=10**9, rng=rng, greedy=True)
                proposed_level = SCAFFOLD_LEVELS[action]
            proposals += 1

            # --- Consent gating ---
            if condition == "scale_full":
                accepted, _ = consent_agent.decide(current_level, proposed_level, rng)
            else:  # rule_based and push_only both auto-execute
                accepted = True

            executed_level = proposed_level if accepted else current_level
            obs, reward, done, info = env.step(executed_level, break_taken=break_taken)

            if condition == "scale_full":
                consent_agent.update(accepted, reward)
            if accepted:
                accepts += 1

            history.append(obs)
            last_error_rate = obs["error_rate"]
            current_level = executed_level
            total_reward += reward
            overload_count += info["true_overload"]

        rows.append({
            "condition": condition,
            "student_id": profile["student_id"],
            "disability_type": profile["disability_type"],
            "total_reward": total_reward,
            "overload_events": overload_count,
            "final_mastery": info["mastery"],
            "acceptance_rate": accepts / proposals,
            "final_trust": consent_agent.trust if condition == "scale_full" else np.nan,
            "breaks_triggered": breaks_triggered,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print("Training ZPD agent...")
    zpd_agent, _ = train_zpd(train_profiles, weeks=10, seed=1)

    print("Training Overload Prediction classifier...")
    overload_clf, feature_cols = train_overload_classifier(train_profiles, zpd_agent)

    N_SEEDS = 3
    print(f"Running full-system integration across 3 conditions x {N_SEEDS} seeds on held-out students...\n")
    all_results = []
    for run_seed in [7 + s * 211 for s in range(N_SEEDS)]:
        for condition in ["scale_full", "rule_based", "push_only"]:
            df = run_condition(test_profiles, zpd_agent, overload_clf, feature_cols, condition, weeks=10, seed=run_seed)
            df["run_seed"] = run_seed
            all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "full_system_comparison.csv"), index=False)

    # per-seed means, then mean +/- std across seeds (avoids pseudo-replication)
    per_seed = combined.groupby(["condition", "run_seed"]).agg(
        mean_reward=("total_reward", "mean"),
        mean_overload_events=("overload_events", "mean"),
        mean_final_mastery=("final_mastery", "mean"),
        mean_acceptance_rate=("acceptance_rate", "mean"),
        mean_breaks_triggered=("breaks_triggered", "mean"),
    )
    summary = per_seed.groupby("condition").agg(["mean", "std"]).round(3)
    print("=== HEADLINE RESULTS TABLE (mean +/- std across 5 seeds) ===")
    print(summary.to_string())

    # Teacher Co-Pilot flags computed on the full-system condition (seed 7 run, for the worked example)
    scale_df = combined[(combined.condition == "scale_full") & (combined.run_seed == 7)].copy()
    overload_threshold = scale_df["overload_events"].median()
    scale_df["flagged"] = scale_df.apply(
        lambda r: len(flag_for_attention(r, overload_threshold=overload_threshold)) > 0, axis=1)
    print(f"\nSCALE (full), seed 7: {scale_df['flagged'].mean():.1%} of students flagged for teacher attention")

    summary.to_csv(os.path.join(RESULTS_DIR, "headline_summary.csv"))
    print("\nSaved full_system_comparison.csv and headline_summary.csv to RESULTS_DIR/")
