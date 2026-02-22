import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from mapless_navigation.forest_env import ForestEnv
import argparse
import rclpy

def main(args=None):
    parser = argparse.ArgumentParser(description='Train PPO agent for Forest Navigation')
    parser.add_argument('--timesteps', type=int, default=2000000, help='Total timesteps for training')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved model')
    parsed_args = parser.parse_args()

    # Initialize ROS2 context if not already (ForestEnv does it, but good practice)
    if not rclpy.ok():
        rclpy.init(args=args)

    # Create environment
    env = ForestEnv()
    
    # Check environment
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")

    # PPO Configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        tensorboard_log="./ppo_forest_tensorboard/"
    )

    model_path = "models/ppo_forest_nav"
    
    if parsed_args.continue_training and os.path.exists(model_path + ".zip"):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=env, tensorboard_log="./ppo_forest_tensorboard/")

    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models/checkpoints/',
        name_prefix='ppo_forest'
    )

    # Train
    print(f"Starting training for {parsed_args.timesteps} timesteps...")
    model.learn(total_timesteps=parsed_args.timesteps, callback=checkpoint_callback)
    
    # Save model
    model_path = "ppo_forest_nav"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()

if __name__ == '__main__':
    main()
