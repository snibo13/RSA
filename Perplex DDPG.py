import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt


# Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.max_action * torch.tanh(self.layer3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.layer1(torch.cat([state, action], 1)))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_episodes = 500
max_timesteps = 2500
batch_size = 1000
exploration_noise = 0.1
tau = 0.005


# Initialize environment and hyperparameters
env = gym.make("InvertedDoublePendulum-v5")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = Actor(state_dim, action_dim, max_action).to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

# Replay buffer setup
replay_buffer = deque(maxlen=100000)
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)

epidose_length = []

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        # Select action with exploration noise
        action = actor(torch.FloatTensor(state).to(device)).cpu().data.numpy()
        action += np.random.normal(0, exploration_noise, size=action_dim)

        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.append(Transition(state, action, next_state, reward, done))

        # Train step
        if len(replay_buffer) > batch_size:
            transitions = random.sample(replay_buffer, batch_size)
            batch = Transition(*zip(*transitions))

            # Compute target Q-value
            next_actions = actor_target(torch.FloatTensor(batch.next_state).to(device))
            target_Q_values = critic_target(
                torch.FloatTensor(batch.next_state).to(device), next_actions
            )

            # Compute critic loss and update critic
            current_Q_values = critic(
                torch.FloatTensor(batch.state).to(device),
                torch.FloatTensor(batch.action).to(device),
            )
            critic_loss = F.mse_loss(
                current_Q_values.squeeze(), target_Q_values.squeeze()
            )
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Compute actor loss and update actor
            actor_loss = -critic(
                torch.FloatTensor(batch.state).to(device),
                actor(torch.FloatTensor(batch.state).to(device)),
            ).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target networks with polyak averaging
            for param_target, param in zip(
                actor_target.parameters(), actor.parameters()
            ):
                param_target.data.copy_(
                    tau * param.data + (1 - tau) * param_target.data
                )

            for param_target, param in zip(
                critic_target.parameters(), critic.parameters()
            ):
                param_target.data.copy_(
                    tau * param.data + (1 - tau) * param_target.data
                )

        state = next_state
        episode_reward += reward

        if done:
            epidose_length.append(t)
            break

    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()
# Plot the episode length
plt.plot(epidose_length)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.show()
