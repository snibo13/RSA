import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=0)

mean_b4_train, std_b4_train = evaluate_policy(
    model, env, n_eval_episodes=100, deterministic=True
)
print(f"Mean reward before training: {mean_b4_train:.2f} +/- {std_b4_train:.2f}")

model.learn(total_timesteps=10_000)

mean_after_train, std_after_train = evaluate_policy(
    model, env, n_eval_episodes=100, deterministic=True
)

print(f"Mean reward after training: {mean_after_train:.2f} +/- {std_after_train:.2f}")
