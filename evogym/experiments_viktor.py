import gymnasium as gym
import evogym
from evogym import sample_robot
import numpy as np
from stable_baselines3 import PPO
import pickle

# import envs from the envs folder and register them
import envs

if __name__ == '__main__':

    """
    # create a random robot
    body, connections = sample_robot((4,4))
    print
    """
    #create robot myself
    body = np.array([
        [1, 1, 1, 1, 1],  # 1: Rigid, 2: Horizontal Actuator
        [3, 1, 3, 3, 1],  # 0: Empty space
        [0, 4, 0, 4, 0],
        [0, 4, 0, 4, 0]
    ], dtype=np.uint8)
    body = np.array([
        [3, 3, 3, 3, 3],  # 1: Rigid, 2: Horizontal Actuator
        [3, 3, 3, 3, 3],  # 0: Empty space
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3]
    ], dtype=np.uint8)
    # Get the connections for the robot
    connections = evogym.envs.get_full_connectivity(body)
    
    
    # make the SimpleWalkingEnv using gym.make and with the robot information
    env = gym.make(
        'Walker-v0',
        body=body,
        connections=connections,
        render_mode='human',
        render_options={'verbose': True}
    )

    # Create and train a PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)  # Adjust timesteps as needed

    # Evaluate the trained agent
    obs = env.reset()
    for i in range(500):

        action, _states = model.predict(obs)  # Use the trained agent's prediction
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs = env.reset()

    env.close()

    # Save trained model and robot
    model.save("walker_ppo_model")

    # Save robot body and connections
    with open("walker_robot.pkl", "wb") as f:
        pickle.dump({'body': body, 'connections': connections}, f)
