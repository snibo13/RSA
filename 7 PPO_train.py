import torch
import gymnasium as gym
from stable_baselines3 import PPO  # Import the PPO algorithm
from stable_baselines3.common.env_util import (
    make_vec_env,
)  # Import the environment utility function
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
import signal


# If terminated early, save the model in its current state
def signal_handler(sig, frame):
    model.save("PPO-SinglePendulum")
    env.close()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def make_env():
    # Create the environment
    env = gym.make("InvertedPendulum-v4", render_mode=None)
    # env = FilterObservation(env, ["arm_qpos", "cube_pos"])
    env = FlattenObservation(env)
    # Create the vectorized environment
    return env


parrallel_env = 8
env = make_vec_env(make_env, n_envs=parrallel_env)

# Create the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log="./ppo_tensorboard/",
    batch_size=128,
)
# print(model.logger)

model.learn(
    total_timesteps=int(7e6),
    progress_bar=True,
)

env.close()
# Save the model
model.save("PPO-SinglePendulum")

# Evaluate the model
env = gym.make("InvertedPendulum-v4", render_mode="human")
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()


# If terminated early, save the model in its current state
def signal_handler(sig, frame):
    model.save("ReachCube")
    env.close()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
signal.pause()
