"""
Phase 1a: Synthetic student population generator for SCALE.

Data grounding (updated): disability-prevalence and performance/engagement
GAP MAGNITUDES below are anchored to real, computed statistics from the
Open University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta &
Zdrahal, 2017), computed directly from the uploaded studentInfo.csv and
studentVle.csv (N=32,593 students; 26,074 with VLE activity):
  - disability prevalence:        9.71%
  - pass/distinction rate:        48.2% (non-disabled) vs 38.1% (disabled)
    -> performance ratio disabled/non-disabled ~= 0.79
  - mean total VLE clicks:        1538.8 (non-disabled) vs 1327.2 (disabled)
    -> engagement ratio disabled/non-disabled ~= 0.86

OULAD does NOT provide disability sub-type (ASD/ADHD/Dyslexia/etc.) --
only a binary Y/N flag -- so the six-subtype breakdown and per-session
behavioral dynamics below remain synthetic. What is now data-grounded is
the *magnitude* of the overall performance/engagement disadvantage, used
as a plausibility check on the synthetic baseline_mastery distribution
(see validation print at the bottom of this file).
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_STUDENTS = 400

DISABILITY_TYPES = ["ASD", "ADHD", "Dyslexia", "ID", "Physical_Sensory", "Mixed"]
# Sub-type split still illustrative (OULAD has no sub-type field) -- flagged in Limitations.
DISABILITY_PROBS = [0.20, 0.25, 0.20, 0.10, 0.10, 0.15]

# OULAD-derived calibration anchors (computed, not assumed -- see docstring)
OULAD_PERFORMANCE_RATIO = 0.381 / 0.482   # disabled pass-rate / non-disabled pass-rate
OULAD_ENGAGEMENT_RATIO = 1327.2 / 1538.8  # disabled clicks / non-disabled clicks
ASSUMED_GENERAL_POPULATION_MASTERY = 0.45  # modeling choice, stated explicitly in the paper

def sample_profiles(n=N_STUDENTS, seed=42):
    rng = np.random.default_rng(seed)
    disability = rng.choice(DISABILITY_TYPES, size=n, p=DISABILITY_PROBS)

    sensory_sensitivity = np.clip(rng.beta(2, 2, n), 0, 1)
    executive_function = np.clip(rng.beta(2, 2, n), 0, 1)
    # baseline_mastery centered so the population mean matches the OULAD-derived
    # performance ratio relative to ASSUMED_GENERAL_POPULATION_MASTERY
    target_mean = ASSUMED_GENERAL_POPULATION_MASTERY * OULAD_PERFORMANCE_RATIO
    baseline_mastery = np.clip(rng.normal(target_mean, 0.12, n), 0.05, 0.8)
    reading_level_baseline = np.clip(rng.normal(0.4, 0.15, n), 0.05, 0.95)
    consent_disposition = np.clip(rng.beta(3, 2, n), 0, 1)
    trust_growth_rate = np.clip(rng.normal(0.08, 0.02, n), 0.02, 0.2)
    # engagement_capacity: scaled so population mean matches OULAD engagement ratio;
    # used downstream to modulate fatigue recovery in the simulation environment
    engagement_capacity = np.clip(rng.normal(OULAD_ENGAGEMENT_RATIO, 0.1, n), 0.3, 1.1)

    for i, d in enumerate(disability):
        if d == "ASD":
            sensory_sensitivity[i] = np.clip(sensory_sensitivity[i] + 0.25, 0, 1)
        elif d == "ADHD":
            executive_function[i] = np.clip(executive_function[i] - 0.2, 0, 1)
        elif d == "Dyslexia":
            reading_level_baseline[i] = np.clip(reading_level_baseline[i] - 0.15, 0.05, 0.95)
        elif d == "ID":
            baseline_mastery[i] = np.clip(baseline_mastery[i] - 0.1, 0.05, 0.8)
        elif d == "Physical_Sensory":
            sensory_sensitivity[i] = np.clip(sensory_sensitivity[i] + 0.15, 0, 1)

    df = pd.DataFrame({
        "student_id": [f"S{idx:04d}" for idx in range(n)],
        "disability_type": disability,
        "baseline_mastery": baseline_mastery,
        "sensory_sensitivity": sensory_sensitivity,
        "executive_function": executive_function,
        "reading_level_baseline": reading_level_baseline,
        "consent_disposition": consent_disposition,
        "trust_growth_rate": trust_growth_rate,
        "engagement_capacity": engagement_capacity,
    })
    return df

if __name__ == "__main__":
    df = sample_profiles()
    df.to_csv(os.path.join(DATA_DIR, "student_profiles.csv"), index=False)
    print(df.shape)
    print(df.groupby("disability_type")[["sensory_sensitivity", "executive_function", "baseline_mastery"]].mean().round(3))
    print()
    print("--- OULAD grounding check ---")
    print(f"Synthetic mean baseline_mastery: {df['baseline_mastery'].mean():.3f}")
    print(f"Assumed general-population mastery: {ASSUMED_GENERAL_POPULATION_MASTERY}")
    print(f"Implied ratio: {df['baseline_mastery'].mean()/ASSUMED_GENERAL_POPULATION_MASTERY:.3f}  "
          f"(target from OULAD: {OULAD_PERFORMANCE_RATIO:.3f})")
    print(f"Synthetic mean engagement_capacity: {df['engagement_capacity'].mean():.3f}  "
          f"(target from OULAD: {OULAD_ENGAGEMENT_RATIO:.3f})")

