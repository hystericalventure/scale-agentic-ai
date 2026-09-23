"""
Phase 3: Overload Prediction Agent (Algorithm 2 in the paper).

Deliberately PROACTIVE, not reactive (this is the core differentiator vs.
ADAPT's GRU anomaly detector, which flags AFTER an anomalous reading).
Features at session t are built ONLY from sessions t-2 and t-1 (never t) --
the label is whether session t is a true_overload event. This is a genuine
one-step-ahead forecasting setup, not concurrent classification.

Proxy note: a GRU is the "real" architecture for this in the full SCALE
design; no torch is available offline here, so we use a RandomForest over
engineered rolling-window features as a documented, honest substitute.
This must be stated plainly in the paper (Methods + Limitations).
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import ZPDAgent, train as train_zpd

WINDOW = 2  # sessions of history used as features


def collect_trajectory(env, policy_fn, rng):
    """Run one student's full episode under a given policy, returning a
    per-session log of raw behavioral signals + true_overload labels."""
    rows = []
    state = env.discretize_state()
    done = False
    while not done:
        action = policy_fn(state, rng)
        obs, reward, done, info = env.step(SCAFFOLD_LEVELS[action])
        row = dict(obs)
        row["true_overload"] = info["true_overload"]
        rows.append(row)
        state = env.discretize_state()
    return pd.DataFrame(rows)


def build_features(traj_df):
    """From a raw per-session trajectory, build (t-2,t-1) rolling features
    to predict true_overload at t. Returns X (features), y (labels)."""
    feats = ["response_latency", "error_rate", "hesitation", "fatigue_proxy"]
    X, y = [], []
    for t in range(WINDOW, len(traj_df)):
        window = traj_df.iloc[t - WINDOW:t]
        row = {}
        for f in feats:
            row[f"{f}_mean"] = window[f].mean()
            row[f"{f}_trend"] = window[f].iloc[-1] - window[f].iloc[0]
        X.append(row)
        y.append(traj_df.iloc[t]["true_overload"])
    return pd.DataFrame(X), np.array(y)


def collect_dataset(profiles, zpd_agent, weeks=10, seed=0):
    """Mix the learned ZPD policy with a random policy across students to
    get a diverse, non-degenerate overload/no-overload dataset."""
    rng = np.random.default_rng(seed)
    all_X, all_y, all_student_ids = [], [], []

    def learned_policy(state, r):
        return zpd_agent.act(state, episode=10**9, rng=r, greedy=True)

    def random_policy(state, r):
        return r.integers(0, len(SCAFFOLD_LEVELS))

    for i, (_, profile) in enumerate(profiles.iterrows()):
        policy_fn = learned_policy if i % 2 == 0 else random_policy
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + i)
        traj = collect_trajectory(env, policy_fn, rng)
        X, y = build_features(traj)
        all_X.append(X)
        all_y.append(y)
        all_student_ids.extend([profile["student_id"]] * len(y))

    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    student_ids = np.array(all_student_ids)
    return X, y, student_ids


if __name__ == "__main__":
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print("Re-training ZPD agent to generate realistic trajectories for overload-agent training data...")
    zpd_agent, _ = train_zpd(train_profiles, weeks=10, seed=1)

    print("Collecting proactive-overload training data (student-level split, no leakage)...")
    X_train, y_train, sid_train = collect_dataset(train_profiles, zpd_agent, weeks=10, seed=2)
    X_test, y_test, sid_test = collect_dataset(test_profiles, zpd_agent, weeks=10, seed=3)

    print(f"Train rows: {len(X_train)}  (positive rate: {y_train.mean():.3f})")
    print(f"Test rows:  {len(X_test)}  (positive rate: {y_test.mean():.3f})")

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                  class_weight="balanced", random_state=0)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n--- Overload Prediction Agent: held-out, one-step-ahead evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.3f}")
    print(f"Precision: {precision_score(y_test, pred):.3f}")
    print(f"Recall:    {recall_score(y_test, pred):.3f}")
    print(f"F1:        {f1_score(y_test, pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.3f}")

    importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(6).round(3))

    results = pd.DataFrame({"y_true": y_test, "y_proba": proba, "y_pred": pred, "student_id": sid_test})
    results.to_csv(os.path.join(RESULTS_DIR, "overload_eval.csv"), index=False)
    importances.to_csv(os.path.join(RESULTS_DIR, "overload_feature_importance.csv"))
    print("\nSaved evaluation + feature importances to RESULTS_DIR/")
