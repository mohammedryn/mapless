"""
Fast PPO / SAC training script using the pure-Python LidarGym environment.

Replaces Gazebo-coupled training (20 steps/sec ceiling) with a NumPy ray-
casting simulator that runs at ~10,000–50,000 steps/sec per process.
With SubprocVecEnv(n_envs=8) the RTX 4050 is the bottleneck, not the sim.

Trained models are saved to the same paths as train_ppo.py, so NavigationNode
and the real-robot deployment stack work unchanged.

Usage examples
--------------
# Train Conv1D + PPO with 8 parallel envs (recommended):
    python -m mapless_navigation.train_fast
    # or after colcon build:
    ros2 run mapless_navigation train_fast

# Change number of parallel workers:
    ros2 run mapless_navigation train_fast --n_envs 4

# Train SAC or flat-MLP baseline:
    ros2 run mapless_navigation train_fast --algorithm sac
    ros2 run mapless_navigation train_fast --policy mlp

# Resume from last checkpoint:
    ros2 run mapless_navigation train_fast --continue_training

Monitor:
    tensorboard --logdir ./ppo_forest_tensorboard
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

from mapless_navigation.lidar_gym import LidarGym, _candidate_config_paths
from mapless_navigation import obs_utils   # N_SCAN constant only; no ROS call here


# ── Conv1D feature extractor (kept in sync with train_ppo.LidarConvExtractor) ─
# Duplicated here to avoid importing train_ppo (which has a top-level rclpy import).

class LidarConvExtractor(BaseFeaturesExtractor):
    """1-D convolutional feature extractor for the 362-dim lidar observation.

    Lidar branch  (360 inputs):
        Conv1D(1->32, k=5, pad=2) -> ReLU
        Conv1D(32->64, k=3, pad=1) -> ReLU
        AdaptiveAvgPool1d(45)      -> Flatten   (2880 units)

    Goal branch (2 inputs):
        Linear(2->32) -> ReLU                   (32 units)

    Head:
        Linear(2912 -> features_dim) -> ReLU
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_scan  = obs_utils.N_SCAN                      # 360
        n_goal  = observation_space.shape[0] - n_scan   # 2
        pool_sz = 45

        self.lidar_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_sz),
            nn.Flatten(),   # 64 × 45 = 2880
        )
        self.goal_branch = nn.Sequential(
            nn.Linear(n_goal, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * pool_sz + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n_scan     = obs_utils.N_SCAN
        lidar_feat = self.lidar_branch(obs[:, :n_scan].unsqueeze(1))
        goal_feat  = self.goal_branch(obs[:, n_scan:])
        return self.head(torch.cat([lidar_feat, goal_feat], dim=1))


# ── Config loading ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not _YAML_OK:
        return {}
    for path in _candidate_config_paths():
        try:
            with open(path) as f:
                data = _yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


# ── Worker factory for SubprocVecEnv ─────────────────────────────────────────

def _make_env(env_config: dict, rank: int, base_seed: int = 0):
    """Return a thunk that spawns and seeds a LidarGym instance."""
    def _init():
        env = LidarGym(env_config=env_config)
        env.reset(seed=base_seed + rank)
        return env
    return _init


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Fast DRL training on LidarGym (no ROS, no Gazebo)")
    parser.add_argument("--timesteps",         type=int,  default=None)
    parser.add_argument("--algorithm",         type=str,  default=None, choices=["ppo", "sac"])
    parser.add_argument("--policy",            type=str,  default=None, choices=["mlp", "conv1d"])
    parser.add_argument("--n_envs",            type=int,  default=8,
                        help="Number of parallel SubprocVecEnv workers (default: 8)")
    parser.add_argument("--continue_training", action="store_true",
                        help="Resume from the latest saved model")
    parsed = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg       = _load_config()
    env_cfg   = cfg.get("env_config", {})

    algo_name   = (parsed.algorithm or cfg.get("algorithm",    "ppo")).lower()
    policy_type = (parsed.policy    or cfg.get("policy_type",  "conv1d")).lower()
    total_steps = parsed.timesteps  or int(cfg.get("total_timesteps", 2_000_000))
    feat_dim    = int(cfg.get("conv_features_dim", 256))
    n_envs      = parsed.n_envs

    lr         = float(cfg.get("learning_rate", 3e-4))
    n_steps    = int(  cfg.get("n_steps",   2048))
    batch_size = int(  cfg.get("batch_size", 64))
    n_epochs   = int(  cfg.get("n_epochs",  10))
    gamma      = float(cfg.get("gamma",      0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_range = float(cfg.get("clip_range", 0.2))
    ent_coef   = float(cfg.get("ent_coef",   0.005))
    ckpt_freq  = int(  cfg.get("checkpoint_freq", 50_000))
    eval_freq  = int(  cfg.get("eval_freq",        25_000))
    eval_eps   = int(  cfg.get("eval_episodes",    20))

    AlgoClass = SAC if algo_name == "sac" else PPO

    # ── Sanity check one env before spawning workers ──────────────────────────
    print("Checking environment compatibility...")
    _check = LidarGym(env_config=env_cfg)
    check_env(_check, warn=True)
    _check.close()
    print("Environment check passed.\n")

    # ── Vectorised training env ───────────────────────────────────────────────
    train_env = SubprocVecEnv(
        [_make_env(env_cfg, rank=i) for i in range(n_envs)],
        start_method="fork",   # fast on Linux; use "spawn" on Windows/macOS
    )

    # Single env for EvalCallback (lightweight, deterministic)
    eval_env = DummyVecEnv([_make_env(env_cfg, rank=999, base_seed=42)])

    # ── Policy kwargs ─────────────────────────────────────────────────────────
    if policy_type == "conv1d":
        policy_kwargs = dict(
            features_extractor_class=LidarConvExtractor,
            features_extractor_kwargs=dict(features_dim=feat_dim),
        )
    else:
        policy_kwargs = None

    model_save_path = os.path.join("models", f"{algo_name}_forest_nav")

    # ── Build or reload model ─────────────────────────────────────────────────
    if parsed.continue_training and os.path.exists(model_save_path + ".zip"):
        print(f"Resuming {algo_name.upper()} from '{model_save_path}.zip'...")
        model = AlgoClass.load(
            model_save_path, env=train_env,
            tensorboard_log="./ppo_forest_tensorboard/")

    elif algo_name == "sac":
        model = SAC(
            "MlpPolicy", train_env,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./ppo_forest_tensorboard/",
        )
    else:
        # n_steps per env — SB3 multiplies internally by n_envs for rollout buffer
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./ppo_forest_tensorboard/",
        )

    print(f"Algorithm  : {algo_name.upper()}")
    print(f"Policy     : {policy_type}")
    print(f"Parallel envs : {n_envs}")
    print(f"Timesteps  : {total_steps:,}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_freq=max(ckpt_freq // n_envs, 1),
        save_path="./models/checkpoints/",
        name_prefix=f"{algo_name}_forest",
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./models/eval_logs/",
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_eps,
        deterministic=True,
        render=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, eval_cb])

    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel saved to '{model_save_path}.zip'")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
