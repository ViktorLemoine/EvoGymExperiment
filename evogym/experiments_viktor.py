import gymnasium as gym
import evogym
from evogym import sample_robot
import numpy as np
from stable_baselines3 import PPO
import pickle
import os

# import envs from the envs folder and register them
import envs


ROBOT_NAME = "walker"

TRAINING_STAGES = [30000, 30000, 40000, 40000]
    
ENV_NAME = "ObstacleTraverser-v0"

# Base folder to save models and robots
BASE_SAVE_FOLDER = f"saved_robots/{ROBOT_NAME}/{ENV_NAME}"
# ===================================================

body = np.array([
    [3, 3, 3, 3, 3],  # 1: Rigid, 2: Horizontal Actuator
    [3, 3, 3, 3, 3],  # 0: Empty space
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
], dtype=np.uint8)
# Get the connections for the robot
connections = evogym.envs.get_full_connectivity(body)


def create_env(body, connections):
    """Create the EvoGym environment with the given robot."""
    env = gym.make(
        ENV_NAME,
        body=body,
        connections=connections,
        render_mode='human',
        render_options={'verbose': True}
    )
    return env

def save_robot(body, connections, stage_idx):
    """Save the robot body and connections."""
    robot_folder = os.path.join(BASE_SAVE_FOLDER, f"stage_{stage_idx+1}", "body")
    os.makedirs(robot_folder, exist_ok=True)
    filepath = os.path.join(robot_folder, f"{ROBOT_NAME}_robot.pkl")
    with open(filepath, "wb") as f:
        pickle.dump({'body': body, 'connections': connections}, f)

def save_model(model, stage_idx):
    """Save the model (brain)."""
    model_folder = os.path.join(BASE_SAVE_FOLDER, f"stage_{stage_idx+1}", "brain")
    os.makedirs(model_folder, exist_ok=True)
    filepath = os.path.join(model_folder, f"{ROBOT_NAME}_ppo_model")
    model.save(filepath)

def train_robot():
    """Train the robot over multiple stages."""
    # Get the connections for the robot
    connections = evogym.envs.get_full_connectivity(body)

    # Create the environment
    env = create_env(body, connections)

    # Create the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Training over multiple stages
    for stage_idx, steps in enumerate(TRAINING_STAGES):
        print(f"\n========== Training Stage {stage_idx+1} | {steps} steps ==========")
        model.learn(total_timesteps=steps)
        save_model(model, stage_idx)
        save_robot(body, connections, stage_idx)
        print(f"Stage {stage_idx+1} completed and saved.\n")

    env.close()
    print("\nTraining complete!")

if __name__ == '__main__':
    train_robot()
