"""
Policy evaluation script.

Runs N complete episodes in the Gazebo simulation environment and reports
quantitative performance metrics.  This replaces the informal "85-90% estimate"
in docs/ with reproducible, comparable numbers across runs and checkpoints.

Usage (launch simulation first, then run this in a second terminal):
    ros2 launch mapless_navigation forest_sim.launch.xml  # terminal 1
    ros2 run mapless_navigation evaluate_policy           # terminal 2

Options:
    --episodes  100           Number of episodes (default: 100)
    --model     models/ppo_forest_nav   Model path without .zip
    --algorithm ppo           'ppo' or 'sac' (must match the saved model)

Output example:
    Episode   1/100  SUCCESS    steps=  87  final_dist=0.42 m
    Episode   2/100  COLLISION  steps=  23  final_dist=1.15 m
    ...
    ========================================================
      Results -- 100 episodes
    ========================================================
      Success rate  :  73.0%  (73/100)
      Collision rate:  18.0%  (18/100)
      Timeout rate  :   9.0%  ( 9/100)
      Avg steps     : 124.3
      Avg final dist:  1.87 m
    ========================================================
"""

import argparse
import os
import numpy as np
import rclpy
from stable_baselines3 import PPO, SAC
from mapless_navigation.forest_env import ForestEnv


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate a trained DRL navigation policy')
    parser.add_argument(
        '--episodes', type=int, default=100,
        help='Number of evaluation episodes (default: 100)')
    parser.add_argument(
        '--model', type=str, default='models/ppo_forest_nav',
        help='Path to the trained model without .zip extension')
    parser.add_argument(
        '--algorithm', type=str, default='ppo', choices=['ppo', 'sac'],
        help='Algorithm class matching the saved model (default: ppo)')
    parsed = parser.parse_args()

    if not rclpy.ok():
        rclpy.init(args=args)

    model_path = parsed.model
    model_zip  = model_path if model_path.endswith('.zip') else model_path + '.zip'
    model_base = model_zip[:-4]

    if not os.path.exists(model_zip):
        print(f"[ERROR] Model not found: {model_zip}")
        return

    env = ForestEnv()
    AlgoClass = SAC if parsed.algorithm == 'sac' else PPO
    model = AlgoClass.load(model_base, env=env)
    model.set_training_mode(False)

    n          = parsed.episodes
    successes  = 0
    collisions = 0
    timeouts   = 0
    steps_list = []
    dist_list  = []

    print(f"\nEvaluating {parsed.algorithm.upper()} model over {n} episodes...\n")

    for ep in range(n):
        obs, _   = env.reset()
        done     = False
        truncated = False
        steps    = 0
        last_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            steps      += 1
            last_reward = reward

        # Un-normalise final distance from last observation vector.
        final_dist = float(obs[-2]) * 10.0
        steps_list.append(steps)
        dist_list.append(final_dist)

        if done and last_reward > 0:
            successes += 1
            outcome = 'SUCCESS  '
        elif truncated:
            timeouts  += 1
            outcome = 'TIMEOUT  '
        else:
            collisions += 1
            outcome = 'COLLISION'

        print(f"  Episode {ep+1:>4}/{n}  {outcome}  "
              f"steps={steps:>4}  final_dist={final_dist:.2f} m")

    print(f"\n{'='*56}")
    print(f"  Results -- {n} episodes")
    print(f"{'='*56}")
    print(f"  Success rate  : {successes/n*100:5.1f}%  ({successes}/{n})")
    print(f"  Collision rate: {collisions/n*100:5.1f}%  ({collisions}/{n})")
    print(f"  Timeout rate  : {timeouts/n*100:5.1f}%  ({timeouts}/{n})")
    print(f"  Avg steps     : {np.mean(steps_list):.1f}")
    print(f"  Avg final dist: {np.mean(dist_list):.2f} m")
    print(f"{'='*56}\n")

    env.close()


if __name__ == '__main__':
    main()
