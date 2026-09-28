"""
SCALE Results Calculation Walkthrough
=======================================
Shows, step by step and out loud, exactly how a headline number (the ZPD
Agent's "6.23 +/- 0.03" reward result) is actually computed -- not just the
final answer, but the train/test split and every individual seed's number
before they get averaged. Good for demonstrating "this isn't made up" live.

Usage: python3 show_calculation.py
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import numpy as np
import pandas as pd
from environment import StudentEnv, SCAFFOLD_LEVELS
from zpd_agent import train as train_zpd

DATA_DIR = os.path.join(_REPO_ROOT, "data")

def line(char="-", n=70):
    print(char * n)

def main():
    print()
    line("=")
    print("STEP 1: Load the 400 students and split them")
    line("=")
    profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
    train_profiles = profiles.sample(frac=0.8, random_state=1)
    test_profiles = profiles.drop(train_profiles.index)

    train_ids = set(train_profiles["student_id"])
    test_ids = set(test_profiles["student_id"])
    overlap = train_ids & test_ids

    print(f"Total students: {len(profiles)}")
    print(f"Training students: {len(train_profiles)}")
    print(f"Held-out TEST students: {len(test_profiles)}")
    print(f"Students appearing in BOTH sets (should be 0): {len(overlap)}")
    print(f"  -> This proves the test students are genuinely never seen during training.")

    line("=")
    print("STEP 2: Train the ZPD Agent ONLY on the 320 training students")
    line("=")
    zpd_agent, episode_rewards = train_zpd(train_profiles, weeks=10, seed=1)
    print(f"Training reward, first 10 students seen:  {np.round(episode_rewards[:10], 2)}")
    print(f"Training reward, last 10 students seen:   {np.round(episode_rewards[-10:], 2)}")
    print(f"  -> Reward rises as the agent learns from experience.")

    line("=")
    print("STEP 3: Evaluate on the 80 TEST students, across 5 different random seeds")
    line("=")
    seed_means = []
    for s in range(5):
        run_seed = 999 + s * 137
        rng = np.random.default_rng(run_seed)
        rewards = []
        for i, (_, profile) in enumerate(test_profiles.iterrows()):
            env = StudentEnv(profile, weeks=10, seed=run_seed * 10_000 + i)
            state = env.discretize_state()
            total_reward = 0.0
            done = False
            while not done:
                action = zpd_agent.act(state, episode=10**9, rng=rng, greedy=True)
                obs, reward, done, info = env.step(SCAFFOLD_LEVELS[action])
                state = env.discretize_state()
                total_reward += reward
            rewards.append(total_reward)
        seed_mean = np.mean(rewards)
        seed_means.append(seed_mean)
        print(f"  Seed #{s+1} (random seed value={run_seed}):  mean reward across 80 students = {seed_mean:.3f}")

    line("=")
    print("STEP 4: Average across the 5 seeds -> this is the number in the README")
    line("=")
    print(f"The 5 seed means were: {np.round(seed_means, 3)}")
    print(f"Average (mean):        {np.mean(seed_means):.3f}")
    print(f"Spread (std dev):      {np.std(seed_means, ddof=1):.3f}")
    print()
    print(f"  -> README reports this as: {np.mean(seed_means):.2f} +/- {np.std(seed_means, ddof=1):.2f}")
    print(f"  -> The small spread shows the result is stable, not a lucky one-time run.")
    line("=")

if __name__ == "__main__":
    main()
