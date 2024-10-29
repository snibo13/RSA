import torch
import gymnasium as gym
from stable_baselines3 import PPO  # Import the PPO algorithm
from stable_baselines3.common.env_util import (
    make_vec_env,
)  # Import the environment utility function
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation

model = PPO.load("PPO-SinglePendulum")


def make_env_human():
    # Create the environment
    env = gym.make("InvertedPendulum-v4", render_mode="human")
    env = FlattenObservation(env)
    # Create the vectorized environment
    return env


env = make_vec_env(make_env_human)
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()
