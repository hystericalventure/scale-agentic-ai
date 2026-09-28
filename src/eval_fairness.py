"""
Evaluation Parameter 1: Fairness across disability sub-types.

Re-analyzes the existing full_system_comparison.csv (no new simulation needed) --
breaks down SCALE's headline metrics by disability sub-type to check whether the
system helps every subgroup, not just the population average.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
import pandas as pd

df = pd.read_csv(os.path.join(RESULTS_DIR, "full_system_comparison.csv"))
scale = df[df.condition == "scale_full"]

fairness = scale.groupby("disability_type").agg(
    n_students=("student_id", "nunique"),
    mean_reward=("total_reward", "mean"),
    mean_overload_events=("overload_events", "mean"),
    mean_final_mastery=("final_mastery", "mean"),
    mean_acceptance_rate=("acceptance_rate", "mean"),
).round(3).sort_values("mean_reward", ascending=False)

print("=== Fairness across disability sub-types (SCALE full system) ===")
print(fairness.to_string())

spread = fairness["mean_final_mastery"].max() - fairness["mean_final_mastery"].min()
print(f"\nMastery spread across sub-types: {spread:.3f} "
      f"(smaller = more equitable across disability types)")

fairness.to_csv(os.path.join(RESULTS_DIR, "fairness_by_subtype.csv"))
print("\nSaved to results/fairness_by_subtype.csv")
