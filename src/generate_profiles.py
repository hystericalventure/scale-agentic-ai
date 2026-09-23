"""
Phase 1a: Synthetic student population generator for SCALE.

Population scope (updated): this population represents learners with
specific learning disabilities (SLD) and closely related neurodevelopmental
conditions, not disability in general. The six sub-types below are chosen
to reflect the real internal structure of the SLD population rather than
an arbitrary disability mix: isolated reading-, math-, and writing-based
SLD (dyslexia, dyscalculia, dysgraphia), ADHD as a "related neurodevelopmental
condition" frequently discussed alongside SLD in IEP contexts, and two
comorbid profiles (SLD+ADHD, and multi-domain SLD) reflecting well-documented
co-occurrence rates in the literature (~25-45% SLD-ADHD comorbidity; ~26-40%
cross-domain SLD comorbidity; van Bergen et al., Psychological Science, 2025).

Data grounding: disability-prevalence and performance/engagement GAP
MAGNITUDES below are anchored to real, computed statistics from the Open
University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal,
2017), computed directly from the uploaded studentInfo.csv and
studentVle.csv (N=32,593 students; 26,074 with VLE activity):
  - disability prevalence:        9.71%
  - pass/distinction rate:        48.2% (non-disabled) vs 38.1% (disabled)
    -> performance ratio disabled/non-disabled ~= 0.79
  - mean total VLE clicks:        1538.8 (non-disabled) vs 1327.2 (disabled)
    -> engagement ratio disabled/non-disabled ~= 0.86

OULAD does NOT provide a disability sub-type field, only a binary Y/N flag,
and it does not distinguish SLD from other disability categories within
that flag. The six-subtype breakdown and per-session behavioral dynamics
below therefore remain a synthetic, literature-informed modelling choice,
not a data-derived one (flagged in Limitations). What is data-grounded is
the *magnitude* of the overall performance/engagement disadvantage, used
as a plausibility check on the synthetic baseline_mastery distribution
(see validation print at the bottom of this file); OULAD's binary flag is
the best available real-world anchor even though it is not SLD-exclusive.
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

# Six SLD-and-related-neurodevelopmental sub-types (see module docstring).
DISABILITY_TYPES = ["Dyslexia", "Dyscalculia", "Dysgraphia", "ADHD",
                     "SLD_ADHD_Comorbid", "Multi_SLD_Comorbid"]
# Split is illustrative (OULAD has no sub-type field) but ordered to reflect
# real relative prevalence: isolated dyslexia is the single most common SLD,
# followed by isolated ADHD; comorbid profiles reflect documented co-occurrence
# rates rather than being an arbitrary catch-all -- flagged in Limitations.
DISABILITY_PROBS = [0.25, 0.15, 0.10, 0.20, 0.20, 0.10]

# OULAD-derived calibration anchors (computed, not assumed -- see docstring)
OULAD_PERFORMANCE_RATIO = 0.381 / 0.482   # disabled pass-rate / non-disabled pass-rate
OULAD_ENGAGEMENT_RATIO = 1327.2 / 1538.8  # disabled clicks / non-disabled clicks
ASSUMED_GENERAL_POPULATION_MASTERY = 0.45  # modeling choice, stated explicitly in the paper

# Per-subtype baseline_mastery shift, used both to perturb individual draws
# below and to analytically compute the expected population-level pull so the
# post-shift mean can be recalibrated back onto the OULAD-derived ratio.
BASELINE_MASTERY_SHIFT = {
    "Dyslexia": -0.12, "Dyscalculia": -0.10, "Dysgraphia": -0.08,
    "ADHD": 0.0, "SLD_ADHD_Comorbid": -0.10, "Multi_SLD_Comorbid": -0.15,
}

def sample_profiles(n=N_STUDENTS, seed=42):
    rng = np.random.default_rng(seed)
    disability = rng.choice(DISABILITY_TYPES, size=n, p=DISABILITY_PROBS)

    sensory_sensitivity = np.clip(rng.beta(2, 2, n), 0, 1)
    executive_function = np.clip(rng.beta(2, 2, n), 0, 1)
    # baseline_mastery centered so the population mean, AFTER the per-subtype
    # shifts below are applied, matches the OULAD-derived performance ratio
    # relative to ASSUMED_GENERAL_POPULATION_MASTERY. Six of these sub-types
    # carry a negative shift (SLD/ADHD populations skew lower), so the
    # pre-shift draw is compensated upward by the expected population-level
    # pull, computed analytically from DISABILITY_PROBS and the shifts above,
    # rather than left to drift off the OULAD anchor.
    expected_shift = sum(p * BASELINE_MASTERY_SHIFT[t] for t, p in zip(DISABILITY_TYPES, DISABILITY_PROBS))
    target_mean = ASSUMED_GENERAL_POPULATION_MASTERY * OULAD_PERFORMANCE_RATIO - expected_shift
    baseline_mastery = np.clip(rng.normal(target_mean, 0.12, n), 0.05, 0.8)
    reading_level_baseline = np.clip(rng.normal(0.4, 0.15, n), 0.05, 0.95)
    consent_disposition = np.clip(rng.beta(3, 2, n), 0, 1)
    trust_growth_rate = np.clip(rng.normal(0.08, 0.02, n), 0.02, 0.2)
    # engagement_capacity: scaled so population mean matches OULAD engagement ratio;
    # used downstream to modulate fatigue recovery in the simulation environment
    engagement_capacity = np.clip(rng.normal(OULAD_ENGAGEMENT_RATIO, 0.1, n), 0.3, 1.1)

    for i, d in enumerate(disability):
        baseline_mastery[i] = np.clip(baseline_mastery[i] + BASELINE_MASTERY_SHIFT[d], 0.05, 0.8)
        if d == "Dyslexia":
            # reading-based SLD: reading-loaded academic content is harder
            reading_level_baseline[i] = np.clip(reading_level_baseline[i] - 0.15, 0.05, 0.95)
        elif d == "Dyscalculia":
            # math-based SLD: documented elevated math-anxiety / stress response
            sensory_sensitivity[i] = np.clip(sensory_sensitivity[i] + 0.08, 0, 1)
        elif d == "Dysgraphia":
            # writing-based SLD: production/output-heavy tasks are harder
            executive_function[i] = np.clip(executive_function[i] - 0.1, 0, 1)
        elif d == "ADHD":
            # related neurodevelopmental condition: attention-regulation
            # effort raises fatigue/hesitation proneness
            executive_function[i] = np.clip(executive_function[i] - 0.2, 0, 1)
            sensory_sensitivity[i] = np.clip(sensory_sensitivity[i] + 0.15, 0, 1)
        elif d == "SLD_ADHD_Comorbid":
            # compounded profile, reflecting ~25-45% documented SLD-ADHD
            # comorbidity (van Bergen et al., 2025)
            sensory_sensitivity[i] = np.clip(sensory_sensitivity[i] + 0.15, 0, 1)
        elif d == "Multi_SLD_Comorbid":
            # cross-domain SLD (reading+math+writing), reflecting documented
            # ~26-40% cross-SLD comorbidity rates
            reading_level_baseline[i] = np.clip(reading_level_baseline[i] - 0.10, 0.05, 0.95)
            sensory_sensitivity[i] = np.clip(sensory_sensitivity[i] + 0.05, 0, 1)

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
