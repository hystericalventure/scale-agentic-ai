"""
Phase 5: Teacher Co-Pilot Agent (Algorithm 4 in the paper).

Deliberately lightweight and rule-based (scoped this way from the start --
this is not where SCALE's novelty/results burden sits). Two jobs:
  1. Generate a plain-language rationale for the latest scaffold decision
     and consent outcome (XAI-style, template-anchored to actual numbers,
     same "anchor to specific data inputs" principle ADAPT used).
  2. Flag students who need teacher attention, and check whether those
     flags are actually informative (correlate with worse outcomes) --
     rather than just asserting they're useful.

Honesty note for the paper: "explainability success rate" here means
template-coverage (every decision gets a grounded rationale), NOT a
human-judged understandability score -- that requires a real user study,
explicitly listed as future work.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np
import pandas as pd


def generate_rationale(row):
    trust = row["final_trust"]
    acc_rate = row["acceptance_rate"]
    overload = row["overload_events"]
    mastery = row["final_mastery"]

    parts = []
    parts.append(f"Student {row['student_id']} ({row['disability_type']}): "
                  f"final mastery {mastery:.2f}, {overload} predicted overload "
                  f"events across the term.")
    parts.append(f"Consent acceptance rate was {acc_rate:.0%}, and trust in the "
                  f"system's suggestions is currently {trust:.2f} (0-1 scale).")
    return " ".join(parts)


def flag_for_attention(row, trust_threshold=0.5, overload_threshold=None):
    reasons = []
    if row["final_trust"] < trust_threshold:
        reasons.append(f"low trust ({row['final_trust']:.2f} < {trust_threshold})")
    if overload_threshold is not None and row["overload_events"] > overload_threshold:
        reasons.append(f"elevated overload events ({row['overload_events']} > {overload_threshold})")
    if row["acceptance_rate"] < 0.4:
        reasons.append(f"low acceptance rate ({row['acceptance_rate']:.0%})")
    return reasons


def run_teacher_copilot(consent_results_path):
    df = pd.read_csv(consent_results_path)
    overload_threshold = df["overload_events"].median()

    df["rationale"] = df.apply(generate_rationale, axis=1)
    df["flags"] = df.apply(lambda r: flag_for_attention(r, overload_threshold=overload_threshold), axis=1)
    df["flagged"] = df["flags"].apply(lambda x: len(x) > 0)

    # explainability coverage: every row has a non-empty rationale by construction
    explainability_coverage = (df["rationale"].str.len() > 0).mean()

    # flag informativeness check: do flagged students actually have worse outcomes?
    flagged_stats = df.groupby("flagged")[["overload_events", "final_mastery"]].mean()

    return df, explainability_coverage, flagged_stats


if __name__ == "__main__":
    df, coverage, flagged_stats = run_teacher_copilot(
        os.path.join(RESULTS_DIR, "consent_with.csv")
    )

    print("--- Teacher Co-Pilot Agent ---")
    print(f"Explainability coverage (every decision has a grounded rationale): {coverage:.1%}")
    print(f"\nStudents flagged for attention: {df['flagged'].sum()} / {len(df)} "
          f"({df['flagged'].mean():.1%})")
    print("\nOutcome comparison, flagged vs not-flagged (checks flags are informative, not arbitrary):")
    print(flagged_stats.round(3))

    print("\nExample rationale + flags for 3 students:")
    for _, row in df.sample(3, random_state=0).iterrows():
        print(f"\n{row['rationale']}")
        print(f"Flags: {row['flags'] if row['flags'] else 'none'}")

    df.to_csv(os.path.join(RESULTS_DIR, "teacher_copilot_output.csv"), index=False)
    print("\nSaved to RESULTS_DIR/teacher_copilot_output.csv")
