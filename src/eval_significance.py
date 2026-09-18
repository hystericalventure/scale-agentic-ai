"""
Evaluation Parameter 2: Statistical significance testing.

Paired t-test comparing SCALE (full) against the rule-based baseline, pairing by
(student_id, run_seed) so each comparison is apples-to-apples on the same student
under the same random draw.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
import pandas as pd
from scipy import stats

df = pd.read_csv(os.path.join(RESULTS_DIR, "full_system_comparison.csv"))

scale = df[df.condition == "scale_full"].set_index(["student_id", "run_seed"])
rule = df[df.condition == "rule_based"].set_index(["student_id", "run_seed"])

paired = scale.join(rule, lsuffix="_scale", rsuffix="_rule", how="inner")
print(f"Paired observations (same student, same seed): {len(paired)}")

results = {}
for metric in ["total_reward", "overload_events", "final_mastery"]:
    a = paired[f"{metric}_scale"]
    b = paired[f"{metric}_rule"]
    t_stat, p_val = stats.ttest_rel(a, b)
    mean_diff = (a - b).mean()
    results[metric] = {"mean_diff": mean_diff, "t_stat": t_stat, "p_value": p_val}
    sig = "significant (p < 0.001)" if p_val < 0.001 else ("significant (p < 0.05)" if p_val < 0.05 else "not significant")
    print(f"\n{metric}:")
    print(f"  SCALE mean: {a.mean():.3f}   Rule-based mean: {b.mean():.3f}   Mean diff: {mean_diff:.3f}")
    print(f"  Paired t-test: t={t_stat:.2f}, p={p_val:.2e}  -> {sig}")

pd.DataFrame(results).T.to_csv(os.path.join(RESULTS_DIR, "significance_tests.csv"))
print("\nSaved to results/significance_tests.csv")
