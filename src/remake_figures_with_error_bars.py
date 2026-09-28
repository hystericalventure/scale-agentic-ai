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

# ---- Figure 3: ZPD agent vs baselines, 5-seed mean +/- std ----
zpd_stats = {
    "Learned\n(Q-learning)": {"reward": (6.23, 0.03), "overload": (14.70, 0.36), "mastery": (1.000, 0.000)},
    "Random\nscaffold": {"reward": (-7.24, 0.30), "overload": (24.63, 0.47), "mastery": (0.912, 0.003)},
    "Static\n(independent)": {"reward": (-40.82, 0.02), "overload": (30.67, 0.07), "mastery": (0.570, 0.000)},
}
labels = list(zpd_stats.keys())
colors = ["#4a7c6f", "#a3a3a3", "#c97b63"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, key, title in zip(axes, ["reward", "overload", "mastery"],
                           ["Mean reward", "Mean overload events", "Mean final mastery"]):
    means = [zpd_stats[l][key][0] for l in labels]
    stds = [zpd_stats[l][key][1] for l in labels]
    ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Figure 3. ZPD Scaffolding Agent vs. baselines (mean \u00b1 std, 5 seeds, held-out students)", y=1.05)
plt.tight_layout()
plt.savefig(R + "fig3_zpd_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ---- Figure 6: full-system comparison, 3-seed mean +/- std ----
full_stats = {
    "Rule-based\nbaseline": {"reward": (-3.654, 0.057), "overload": (22.938, 0.352), "mastery": (0.816, 0.007)},
    "Push-only\n(no consent)": {"reward": (5.015, 0.058), "overload": (14.117, 0.397), "mastery": (1.000, 0.000)},
    "SCALE\n(full)": {"reward": (1.220, 0.265), "overload": (15.279, 0.248), "mastery": (0.989, 0.003)},
}
labels2 = list(full_stats.keys())
colors2 = ["#c97b63", "#a3a3a3", "#4a7c6f"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, key, title in zip(axes, ["reward", "overload", "mastery"],
                           ["Mean reward", "Mean overload events", "Mean final mastery"]):
    means = [full_stats[l][key][0] for l in labels2]
    stds = [full_stats[l][key][1] for l in labels2]
    ax.bar(labels2, means, yerr=stds, capsize=5, color=colors2)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Figure 6. Full-system comparison (mean \u00b1 std, 3 seeds, held-out students)", y=1.05)
plt.tight_layout()
plt.savefig(R + "fig6_full_system_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print("done")
