import gymnasium as gym
import evogym
import evogym.envs
import numpy as np
from stable_baselines3 import PPO
import pickle
import os

# ================= USER SETTINGS =================
ROBOT_NAME = "walker"   # Which robot to analyze
STAGE_NUM = 3           # Which stage you want to load (1, 2, or 3)

# Environment name
ENV_NAME = "Walker-v0"

# Number of steps to run
EVAL_STEPS = 500

# Base folder where robots are saved
BASE_LOAD_FOLDER = f"saved_robots/{ROBOT_NAME}/{ENV_NAME}/stage_{STAGE_NUM}"
# ===================================================

def load_robot():
    """Load the robot's body and connections."""
    robot_path = os.path.join(BASE_LOAD_FOLDER, "body", f"{ROBOT_NAME}_robot.pkl")
    with open(robot_path, "rb") as f:
        robot_data = pickle.load(f)
    return robot_data['body'], robot_data['connections']

def load_model():
    """Load the trained model."""
    model_path = os.path.join(BASE_LOAD_FOLDER, "brain", f"{ROBOT_NAME}_ppo_model")
    model = PPO.load(model_path)
    return model

def create_env(body, connections):
    """Create the EvoGym environment with the loaded robot."""
    env = gym.make(
        ENV_NAME,
        body=body,
        connections=connections,
        render_mode='human',
        render_options={'verbose': True}
    )
    return env

def evaluate_robot():
    """Evaluate the loaded robot."""
    # Load robot and model
    body, connections = load_robot()
    model = load_model()

    # Create environment
    env = create_env(body, connections)

    # Reset environment (make sure to only get obs)
    obs, _ = env.reset()

    # Evaluation loop
    for _ in range(EVAL_STEPS):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == '__main__':
    evaluate_robot()
