"""
SCALE Live Demonstration
=========================
Walks one synthetic student through several real sessions, showing exactly
what each of the four agents decides and why -- using the actual trained
models, not scripted/fake output. Run this to demonstrate the working
system to an audience.

Usage: python3 demo.py
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
DATA_DIR = os.path.join(_REPO_ROOT, "data")

import numpy as np
import pandas as pd
from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import train as train_zpd
from consent_agent import ConsentTrustAgent
from overload_agent import collect_dataset, WINDOW
from full_system import train_overload_classifier, build_risk_features
from teacher_copilot_agent import generate_rationale, flag_for_attention

SCAFFOLD_NAMES = {0: "independent work", 1: "a partial hint", 2: "a full worked example"}

def line(char="-", n=70):
    print(char * n)

def demo():
    print()
    line("=")
    print("SCALE DEMONSTRATION -- four agents, one student, real decisions")
    line("=")

    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print("\n[Setup] Training ZPD agent on 320 students...")
    zpd_agent, _ = train_zpd(train_profiles, weeks=10, seed=1)
    print("[Setup] Training Overload Prediction classifier...")
    overload_clf, feature_cols = train_overload_classifier(train_profiles, zpd_agent)
    print("[Setup] Done. Both models are now making live decisions below.\n")

    # pick one held-out student the models have never seen
    student = test_profiles.iloc[3]
    print(f"Selected student: {student['student_id']}  "
          f"(disability type: {student['disability_type']})")
    print(f"  baseline mastery: {student['baseline_mastery']:.2f}   "
          f"sensory sensitivity: {student['sensory_sensitivity']:.2f}   "
          f"consent disposition: {student['consent_disposition']:.2f}")

    env = StudentEnv(student, weeks=10, seed=42)
    consent_agent = ConsentTrustAgent(student)
    current_level = 0
    history = []
    overload_count, accepts, proposals, breaks = 0, 0, 0, 0

    N_SESSIONS_TO_SHOW = 8
    for session in range(N_SESSIONS_TO_SHOW):
        line()
        print(f"SESSION {session + 1}")

        # --- Overload Agent: proactive check ---
        break_taken = False
        if len(history) >= WINDOW:
            feats = build_risk_features(history, feature_cols)
            risk = overload_clf.predict_proba(feats)[0, 1]
            print(f"  [Overload Agent]  predicted overload risk = {risk:.2f}", end="")
            if risk >= 0.5:
                break_taken = True
                breaks += 1
                print("  -> RISK HIGH: triggering a preemptive break before this session")
            else:
                print("  -> risk acceptable, proceeding normally")
        else:
            print("  [Overload Agent]  not enough history yet (needs 2 prior sessions)")

        # --- ZPD Agent: propose scaffold level ---
        state = env.discretize_state()
        action = zpd_agent.act(state, episode=10**9, rng=np.random.default_rng(session), greedy=True)
        proposed_level = SCAFFOLD_LEVELS[action]
        proposals += 1
        print(f"  [ZPD Agent]       proposes: {SCAFFOLD_NAMES[proposed_level]} "
              f"(state: mastery_bin={state[0]}, fatigue_bin={state[1]}, error_bin={state[2]})")

        # --- Consent Agent: accept or reject ---
        accepted, accept_prob = consent_agent.decide(current_level, proposed_level, np.random.default_rng(session))
        print(f"  [Consent Agent]   current trust = {consent_agent.trust:.2f}, "
              f"P(accept) = {accept_prob:.2f} -> student {'ACCEPTS' if accepted else 'DECLINES'} this suggestion")

        executed_level = proposed_level if accepted else current_level
        obs, reward, done, info = env.step(executed_level, break_taken=break_taken)
        consent_agent.update(accepted, reward)
        if accepted:
            accepts += 1
        current_level = executed_level
        history.append(obs)
        overload_count += info["true_overload"]

        print(f"  [Environment]     executed: {SCAFFOLD_NAMES[executed_level]}  |  "
              f"error_rate={obs['error_rate']:.2f}  latency={obs['response_latency']:.2f}  "
              f"mastery now={info['mastery']:.3f}")
        if done:
            break

    # --- Teacher Co-Pilot: end-of-run summary ---
    line("=")
    print("TEACHER CO-PILOT AGENT -- end-of-session-block summary")
    line("=")
    summary_row = pd.Series({
        "student_id": student["student_id"],
        "disability_type": student["disability_type"],
        "final_mastery": info["mastery"],
        "overload_events": overload_count,
        "acceptance_rate": accepts / proposals,
        "final_trust": consent_agent.trust,
    })
    print(generate_rationale(summary_row))
    flags = flag_for_attention(summary_row, overload_threshold=2)
    print(f"\nFlags for teacher attention: {flags if flags else 'none'}")
    print(f"\nPreemptive breaks triggered by Overload Agent: {breaks}")
    line("=")
    print("End of demonstration.")
    line("=")

if __name__ == "__main__":
    demo()
