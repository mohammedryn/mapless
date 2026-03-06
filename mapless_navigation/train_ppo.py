"""
PPO / SAC training script for mapless DRL navigation.

Usage examples
--------------
# Train with Conv1D policy (default, recommended):
    ros2 run mapless_navigation train_ppo

# Train flat MLP baseline for ablation comparison:
    ros2 run mapless_navigation train_ppo --policy mlp

# Train SAC instead of PPO:
    ros2 run mapless_navigation train_ppo --algorithm sac

# Resume from last checkpoint:
    ros2 run mapless_navigation train_ppo --continue_training

# Override timesteps:
    ros2 run mapless_navigation train_ppo --timesteps 4000000

Monitor training:
    tensorboard --logdir ./ppo_forest_tensorboard
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import rclpy

from mapless_navigation.forest_env import ForestEnv
from mapless_navigation import obs_utils


# ── Custom 1-D Convolutional Feature Extractor ───────────────────────────────

class LidarConvExtractor(BaseFeaturesExtractor):
    """1-D convolutional feature extractor for the 362-dimensional observation.

    The 360 lidar rays form a *circular* 1-D signal: spatially adjacent rays
    are physically adjacent in the environment.  A flat MLP ignores this
    structure and treats all 362 inputs as independent.  The 1-D convolutions
    here learn local edge/gap detectors across the lidar ring — the inductive
    bias matches the sensor geometry.

    Architecture
    ────────────
    Lidar branch  (360 inputs):
        Conv1D(1->32, k=5, pad=2) -> ReLU
        Conv1D(32->64, k=3, pad=1) -> ReLU
        AdaptiveAvgPool1d(45)      -> Flatten   ->  2880 units

    Goal branch   (2 inputs):
        Linear(2->32) -> ReLU                   ->    32 units

    Head:
        Linear(2880 + 32 -> features_dim) -> ReLU
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_scan  = obs_utils.N_SCAN                     # 360
        n_goal  = observation_space.shape[0] - n_scan  # 2
        pool_sz = 45                                   # ~8x compression

        self.lidar_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_sz),
            nn.Flatten(),                              # 64 x 45 = 2880
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

def _load_training_config() -> dict:
    """Load training.yaml from the installed package share (or fallback path)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(
            get_package_share_directory('mapless_navigation'),
            'config', 'training.yaml',
        )
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        pass
    fallback = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'training.yaml'))
    if os.path.exists(fallback):
        with open(fallback) as f:
            return yaml.safe_load(f)
    return {}


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(
        description='Train a DRL agent for mapless navigation')
    parser.add_argument(
        '--timesteps', type=int, default=None,
        help='Override total_timesteps from training.yaml')
    parser.add_argument(
        '--algorithm', type=str, default=None, choices=['ppo', 'sac'],
        help='Override algorithm from training.yaml')
    parser.add_argument(
        '--policy', type=str, default=None, choices=['mlp', 'conv1d'],
        help='Override policy_type from training.yaml')
    parser.add_argument(
        '--continue_training', action='store_true',
        help='Resume from the latest saved model checkpoint')
    parsed = parser.parse_args()

    if not rclpy.ok():
        rclpy.init(args=args)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = _load_training_config()

    algo_name   = (parsed.algorithm or cfg.get('algorithm',   'ppo')).lower()
    policy_type = (parsed.policy    or cfg.get('policy_type', 'conv1d')).lower()
    total_steps = parsed.timesteps  or int(cfg.get('total_timesteps', 2_000_000))
    feat_dim    = int(cfg.get('conv_features_dim', 256))

    lr         = float(cfg.get('learning_rate', 3e-4))
    n_steps    = int(cfg.get('n_steps',   2048))
    batch_size = int(cfg.get('batch_size', 64))
    n_epochs   = int(cfg.get('n_epochs',  10))
    gamma      = float(cfg.get('gamma',      0.99))
    gae_lambda = float(cfg.get('gae_lambda', 0.95))
    clip_range = float(cfg.get('clip_range', 0.2))
    ent_coef   = float(cfg.get('ent_coef',   0.005))
    ckpt_freq  = int(cfg.get('checkpoint_freq', 50_000))
    eval_freq  = int(cfg.get('eval_freq',       25_000))
    eval_eps   = int(cfg.get('eval_episodes',   20))

    AlgoClass = SAC if algo_name == 'sac' else PPO

    # ── Environment ───────────────────────────────────────────────────────────
    env = ForestEnv()
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment check passed.\n")

    # ── Policy kwargs ─────────────────────────────────────────────────────────
    # conv1d — custom 1-D convolutional lidar encoder (recommended).
    # mlp    — flat MlpPolicy baseline; useful for ablation comparisons to
    #          quantify the benefit of the conv architecture.
    if policy_type == 'conv1d':
        policy_kwargs = dict(
            features_extractor_class=LidarConvExtractor,
            features_extractor_kwargs=dict(features_dim=feat_dim),
        )
    else:
        policy_kwargs = None

    model_save_path = os.path.join('models', f'{algo_name}_forest_nav')

    # ── Build or reload model ─────────────────────────────────────────────────
    if parsed.continue_training and os.path.exists(model_save_path + '.zip'):
        print(f"Resuming {algo_name.upper()} from '{model_save_path}.zip'...")
        model = AlgoClass.load(
            model_save_path, env=env,
            tensorboard_log='./ppo_forest_tensorboard/')

    elif algo_name == 'sac':
        model = SAC(
            'MlpPolicy', env,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log='./ppo_forest_tensorboard/',
        )
    else:
        model = PPO(
            'MlpPolicy', env,
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
            tensorboard_log='./ppo_forest_tensorboard/',
        )

    print(f"Algorithm  : {algo_name.upper()}")
    print(f"Policy     : {policy_type}")
    print(f"Timesteps  : {total_steps:,}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path='./models/checkpoints/',
        name_prefix=f'{algo_name}_forest',
    )
    eval_cb = EvalCallback(
        eval_env=env,
        best_model_save_path='./models/best_model/',
        log_path='./models/eval_logs/',
        eval_freq=eval_freq,
        n_eval_episodes=eval_eps,
        deterministic=True,
        render=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, eval_cb])

    os.makedirs('models', exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel saved to '{model_save_path}.zip'")

    env.close()


if __name__ == '__main__':
    main()
