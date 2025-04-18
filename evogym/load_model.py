import gymnasium as gym
import evogym
from evogym import sample_robot
import numpy as np
from stable_baselines3 import PPO
import pickle

# Load model
model = PPO.load("walker_ppo_model")

# Load robot
with open("walker_robot.pkl", "rb") as f:
    robot_data = pickle.load(f)

body = robot_data['body']
connections = robot_data['connections']

env = gym.make(
    'Walker-v0',
    body=body,
    connections=connections,
    render_mode='human',
    render_options={'verbose': True}
)

obs = env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs = env.reset()