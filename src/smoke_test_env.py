import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SRC_DIR = os.path.join(_REPO_ROOT, "src")

import pandas as pd
import numpy as np
from environment import StudentEnv, SCAFFOLD_LEVELS

profiles = pd.read_csv(os.path.join(DATA_DIR, "student_profiles.csv"))
rng = np.random.default_rng(1)

# Policy A: always ideal scaffold (oracle) vs Policy B: random scaffold (bad baseline)
def run_episode(profile_row, policy="oracle", seed=0):
    env = StudentEnv(profile_row, weeks=10, seed=seed)
    total_reward, overload_count, final_mastery = 0, 0, None
    done = False
    while not done:
        ideal = env._ideal_scaffold()
        action = ideal if policy == "oracle" else rng.choice(SCAFFOLD_LEVELS)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        overload_count += info["true_overload"]
    return total_reward, overload_count, info["mastery"]

oracle_results = [run_episode(profiles.iloc[i], "oracle", seed=i) for i in range(30)]
random_results = [run_episode(profiles.iloc[i], "random", seed=i) for i in range(30)]

for name, res in [("oracle (ideal scaffold)", oracle_results), ("random scaffold", random_results)]:
    rewards, overloads, masteries = zip(*res)
    print(f"{name}: mean_reward={np.mean(rewards):.2f}  mean_overload_events={np.mean(overloads):.2f}  mean_final_mastery={np.mean(masteries):.3f}")
