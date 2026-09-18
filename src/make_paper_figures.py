import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 11})
R = os.path.join(RESULTS_DIR, "")

# ---- Figure 3: ZPD training curve (learned vs baselines, bar) ----
learned = pd.read_csv(R + "zpd_eval_learned.csv")
random_ = pd.read_csv(R + "zpd_eval_random.csv")
static_ = pd.read_csv(R + "zpd_eval_static_independent.csv")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
labels = ["Learned\n(Q-learning)", "Random\nscaffold", "Static\n(independent)"]
colors = ["#4a7c6f", "#a3a3a3", "#c97b63"]

for ax, metric, title in zip(
    axes,
    ["total_reward", "overload_events", "final_mastery"],
    ["Mean reward", "Mean overload events", "Mean final mastery"],
):
    vals = [learned[metric].mean(), random_[metric].mean(), static_[metric].mean()]
    ax.bar(labels, vals, color=colors)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Figure 3. ZPD Scaffolding Agent vs. baselines (held-out students)", y=1.04)
plt.tight_layout()
plt.savefig(R + "fig3_zpd_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ---- Figure 4: Weekly acceptance-rate / trust-building curve ----
weekly = np.load(R + "weekly_acceptance_rate.npy")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(range(1, len(weekly) + 1), weekly * 100, marker="o", color="#5b4a9e", linewidth=2)
ax.set_xlabel("Week")
ax.set_ylabel("Acceptance rate (%)")
ax.set_title("Figure 4. Consent acceptance rate rises as student trust builds")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(R + "fig4_trust_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# ---- Figure 5: Overload Prediction Agent - feature importance ----
importances = pd.read_csv(R + "overload_feature_importance.csv", index_col=0).squeeze("columns")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
importances.head(6).sort_values().plot(kind="barh", ax=ax, color="#3f7d8c")
ax.set_title("Figure 5. Overload Prediction Agent: top feature importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(R + "fig5_overload_features.png", dpi=200, bbox_inches="tight")
plt.close()

# ---- Figure 6: Headline comparison across conditions (full system) ----
summary = pd.read_csv(R + "headline_summary.csv", index_col=0)
summary = summary.loc[["rule_based", "push_only", "scale_full"]]
labels = ["Rule-based\nbaseline", "Push-only\n(no consent)", "SCALE\n(full)"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
metrics = ["mean_reward", "mean_overload_events", "mean_final_mastery"]
titles = ["Mean reward", "Mean overload events", "Mean final mastery"]
colors = ["#c97b63", "#a3a3a3", "#4a7c6f"]
for ax, metric, title in zip(axes, metrics, titles):
    ax.bar(labels, summary[metric].values, color=colors)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Figure 6. Full-system comparison, held-out students", y=1.04)
plt.tight_layout()
plt.savefig(R + "fig6_full_system_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print("All figures saved to", R)
import os
for f in sorted(os.listdir(R)):
    if f.endswith(".png"):
        print(" -", f)
