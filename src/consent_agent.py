"""
Phase 4: Consent & Trust Agent (Algorithm 3 in the paper).

Core novel mechanism of SCALE (this is what has no equivalent in ADAPT).
Every scaffold change proposed by the ZPD Agent is gated through this
agent before execution:
  - accept probability depends on current trust + magnitude of the
    requested change (bigger jumps are harder to accept)
  - trust updates asymmetrically: accepted+helpful -> trust up,
    accepted+unhelpful -> trust down, REJECTED -> small autonomy bonus
    (respecting a "no" is itself trust-building, not a wasted turn)
  - if rejected, the system falls back to the student's current level
    rather than forcing the change -- this is the behavioral difference
    we evaluate against a "no-consent" control (equivalent to ADAPT's
    push-only model, which has no accept/reject step at all)
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np
import pandas as pd
from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import train as train_zpd


class ConsentTrustAgent:
    def __init__(self, profile):
        self.trust = float(profile["consent_disposition"])
        self.growth_rate = float(profile["trust_growth_rate"])

    def decide(self, current_level, proposed_level, rng):
        magnitude = abs(proposed_level - current_level) / (len(SCAFFOLD_LEVELS) - 1)
        accept_prob = np.clip(self.trust - 0.35 * magnitude, 0.03, 0.98)
        accept = rng.random() < accept_prob
        return bool(accept), float(accept_prob)

    def update(self, accepted, reward):
        if accepted:
            delta = self.growth_rate * (1.0 if reward > 0 else -1.0)
            self.trust = np.clip(self.trust + delta, 0.05, 0.99)
        else:
            # respecting a rejection still builds a small amount of trust
            self.trust = np.clip(self.trust + 0.3 * self.growth_rate, 0.05, 0.99)


def run_with_consent(profiles, zpd_agent, weeks, use_consent, seed=0):
    """use_consent=False reproduces a push-only (ADAPT-style) control:
    the proposed action always executes regardless of student preference."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, (_, profile) in enumerate(profiles.iterrows()):
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + i)
        consent_agent = ConsentTrustAgent(profile)
        current_level = 0
        state = env.discretize_state()
        total_reward, overload_count, accepts, proposals = 0.0, 0, 0, 0
        session = 0
        done = False
        while not done:
            proposed_action = zpd_agent.act(state, episode=10**9, rng=rng, greedy=True)
            proposed_level = SCAFFOLD_LEVELS[proposed_action]
            proposals += 1

            if use_consent:
                accepted, accept_prob = consent_agent.decide(current_level, proposed_level, rng)
            else:
                accepted, accept_prob = True, 1.0

            executed_level = proposed_level if accepted else current_level
            obs, reward, done, info = env.step(executed_level)

            if use_consent:
                consent_agent.update(accepted, reward)
            if accepted:
                accepts += 1
            current_level = executed_level
            state = env.discretize_state()
            total_reward += reward
            overload_count += info["true_overload"]
            session += 1

        rows.append({
            "student_id": profile["student_id"],
            "disability_type": profile["disability_type"],
            "total_reward": total_reward,
            "overload_events": overload_count,
            "final_mastery": info["mastery"],
            "acceptance_rate": accepts / proposals,
            "final_trust": consent_agent.trust if use_consent else np.nan,
        })
    return pd.DataFrame(rows)


def acceptance_rate_over_time(profiles, zpd_agent, weeks, seed=0, n_students=60):
    """Track how acceptance rate evolves week-by-week (trust-building curve)."""
    rng = np.random.default_rng(seed)
    weekly_accepts = np.zeros(weeks)
    weekly_totals = np.zeros(weeks)
    subset = profiles.sample(n=min(n_students, len(profiles)), random_state=seed)

    for i, (_, profile) in enumerate(subset.iterrows()):
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + i)
        consent_agent = ConsentTrustAgent(profile)
        current_level = 0
        state = env.discretize_state()
        done = False
        while not done:
            week = env.session_idx // 5
            proposed_action = zpd_agent.act(state, episode=10**9, rng=rng, greedy=True)
            proposed_level = SCAFFOLD_LEVELS[proposed_action]
            accepted, _ = consent_agent.decide(current_level, proposed_level, rng)
            executed_level = proposed_level if accepted else current_level
            obs, reward, done, info = env.step(executed_level)
            consent_agent.update(accepted, reward)
            current_level = executed_level
            state = env.discretize_state()
            weekly_totals[week] += 1
            weekly_accepts[week] += int(accepted)
    return weekly_accepts / np.maximum(weekly_totals, 1)


if __name__ == "__main__":
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print("Training ZPD agent (shared policy for consent experiment)...")
    zpd_agent, _ = train_zpd(train_profiles, weeks=10, seed=1)

    print("Running WITH consent gating vs WITHOUT (push-only control)...")
    with_consent = run_with_consent(test_profiles, zpd_agent, weeks=10, use_consent=True, seed=5)
    without_consent = run_with_consent(test_profiles, zpd_agent, weeks=10, use_consent=False, seed=5)

    print("\n--- Consent-gated vs push-only, held-out students ---")
    for name, df in [("WITH consent", with_consent), ("WITHOUT consent (push-only)", without_consent)]:
        print(f"{name:28s}  mean_reward={df.total_reward.mean():7.2f}  "
              f"mean_overload={df.overload_events.mean():5.2f}  "
              f"mean_mastery={df.final_mastery.mean():.3f}")
    print(f"\nMean acceptance rate (with consent): {with_consent.acceptance_rate.mean():.3f}")
    print(f"Mean final trust: {with_consent.final_trust.mean():.3f}")

    print("\nComputing week-by-week acceptance-rate trend (trust-building curve)...")
    weekly_rates = acceptance_rate_over_time(test_profiles, zpd_agent, weeks=10, seed=5)
    print("Weekly acceptance rate:", np.round(weekly_rates, 3))

    with_consent.to_csv(os.path.join(RESULTS_DIR, "consent_with.csv"), index=False)
    without_consent.to_csv(os.path.join(RESULTS_DIR, "consent_without.csv"), index=False)
    np.save(os.path.join(RESULTS_DIR, "weekly_acceptance_rate.npy"), weekly_rates)
    print("\nSaved consent experiment results to RESULTS_DIR/")
