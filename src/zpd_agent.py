"""
Phase 2: ZPD Scaffolding Agent (Algorithm 1 in the paper).

Tabular Q-learning over the discretized state space defined in
environment.py's discretize_state() -- (mastery_bin[0-4], fatigue_bin[0-2],
recent_error_bin[0-2]) = 45 states x 3 actions (scaffold levels).

Critically: the agent NEVER calls env._ideal_scaffold() -- it only sees
discretize_state() + reward, exactly like a real system would. This is
what makes "the agent learns the ZPD policy" a genuine result rather than
a restated assumption.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")
import numpy as np
import pandas as pd
from environment import StudentEnv, SCAFFOLD_LEVELS

N_ACTIONS = len(SCAFFOLD_LEVELS)
STATE_SHAPE = (5, 3, 3)  # mastery_bin, fatigue_bin, error_bin

ALPHA = 0.15      # learning rate
GAMMA = 0.9       # discount factor
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY_EPISODES = 250


class ZPDAgent:
    def __init__(self):
        self.Q = np.zeros(STATE_SHAPE + (N_ACTIONS,))

    def epsilon(self, episode):
        frac = min(episode / EPS_DECAY_EPISODES, 1.0)
        return EPS_START + frac * (EPS_END - EPS_START)

    def act(self, state, episode, rng, greedy=False):
        if (not greedy) and rng.random() < self.epsilon(episode):
            return rng.integers(0, N_ACTIONS)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.Q[s_next])
        td_target = r + GAMMA * best_next
        self.Q[s][a] += ALPHA * (td_target - self.Q[s][a])


def train(profiles, weeks=10, episodes_per_student=1, seed=0):
    """One pass per student is enough here because each student IS an
    episode of `weeks*5` sequential decisions -- the Q-table is shared
    and learns across the whole population, not per-student."""
    agent = ZPDAgent()
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for ep, (_, profile) in enumerate(profiles.iterrows()):
        env = StudentEnv(profile, weeks=weeks, seed=seed * 10_000 + ep)
        state = env.discretize_state()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, ep, rng)
            obs, reward, done, info = env.step(SCAFFOLD_LEVELS[action])
            next_state = env.discretize_state()
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)

    return agent, episode_rewards


def evaluate(agent, profiles, weeks=10, seed=999, greedy=True, n_seeds=5):
    """Evaluate the trained (greedy) policy vs. two baselines across multiple
    random seeds (environment stochasticity only -- the greedy policy itself
    is deterministic) and report mean +/- std, not a single run."""
    all_seed_results = {"learned": [], "random": [], "static_independent": []}

    for s in range(n_seeds):
        run_seed = seed + s * 137
        rng = np.random.default_rng(run_seed)
        for policy_name in all_seed_results:
            rows = []
            for i, (_, profile) in enumerate(profiles.iterrows()):
                env = StudentEnv(profile, weeks=weeks, seed=run_seed * 10_000 + i)
                state = env.discretize_state()
                total_reward, overload_count = 0.0, 0
                done = False
                while not done:
                    if policy_name == "learned":
                        action = agent.act(state, episode=10**9, rng=rng, greedy=True)
                    elif policy_name == "random":
                        action = rng.integers(0, N_ACTIONS)
                    else:
                        action = 0
                    obs, reward, done, info = env.step(SCAFFOLD_LEVELS[action])
                    state = env.discretize_state()
                    total_reward += reward
                    overload_count += info["true_overload"]
                rows.append({
                    "student_id": profile["student_id"],
                    "disability_type": profile["disability_type"],
                    "total_reward": total_reward,
                    "overload_events": overload_count,
                    "final_mastery": info["mastery"],
                    "seed": run_seed,
                })
            all_seed_results[policy_name].append(pd.DataFrame(rows))

    return {k: pd.concat(v, ignore_index=True) for k, v in all_seed_results.items()}


if __name__ == "__main__":
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    print(f"Training on {len(train_profiles)} students, evaluating on {len(test_profiles)} held-out students...")
    agent, ep_rewards = train(train_profiles, weeks=10, seed=1)
    print(f"Training complete. Mean reward first 50 eps: {np.mean(ep_rewards[:50]):.2f} "
          f"-> last 50 eps: {np.mean(ep_rewards[-50:]):.2f}")

    eval_results = evaluate(agent, test_profiles, weeks=10, seed=999, n_seeds=5)
    print("\n--- Held-out evaluation across 5 seeds (mean +/- std) ---")
    for name, df in eval_results.items():
        # aggregate per-seed means first, then report mean+/-std across seeds
        per_seed = df.groupby("seed")[["total_reward", "overload_events", "final_mastery"]].mean()
        print(f"{name:20s}  reward={per_seed.total_reward.mean():7.2f}+/-{per_seed.total_reward.std():.2f}  "
              f"overload={per_seed.overload_events.mean():5.2f}+/-{per_seed.overload_events.std():.2f}  "
              f"mastery={per_seed.final_mastery.mean():.3f}+/-{per_seed.final_mastery.std():.3f}")

    for name, df in eval_results.items():
        df.to_csv(os.path.join(RESULTS_DIR, f"zpd_eval_{name}.csv"), index=False)

    np.save(os.path.join(RESULTS_DIR, "zpd_qtable.npy"), agent.Q)
    print("\nSaved Q-table and evaluation CSVs to RESULTS_DIR/")
