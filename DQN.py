import random
import math
from collections import namedtuple, deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, nq, nu, n_hidden, width):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(nq, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(n_hidden)])
        self.layer3 = nn.Linear(width, nu)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.layer3(x)


def select_action(state, steps_done, action_space):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)


def train_step():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    optimizer.step()


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    # Environment setup
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    nq = env.observation_space.shape[0]
    nu = env.action_space.n

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks and optimizer
    policy_net = DQN(nq, nu, 2, 128).to(device)
    target_net = DQN(nq, nu, 2, 128).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    # Replay memory
    memory_capacity = 10000
    memory = ReplayMemory(memory_capacity)

    steps_done = 0

    num_episodes = 600 if device == torch.device("cuda") else 100

    episode_durations = []

    print(f"Training on {device}...")

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        done = False
        t = 0

        while not done:
            t += 1

            action = select_action(state, steps_done, env.action_space)
            next_state, reward, done, truncated, _ = env.step(action.item())

            reward_tensor = torch.tensor([reward], device=device)

            if done:
                next_state_tensor = None
            else:
                next_state_tensor = torch.tensor(
                    next_state, device=device, dtype=torch.float32
                ).unsqueeze(0)

            memory.push(state, action, next_state_tensor, reward_tensor)

            state = next_state_tensor

            train_step()

            # Soft update of target network's weights
            for target_param, policy_param in zip(
                target_net.parameters(), policy_net.parameters()
            ):
                target_param.data.copy_(
                    TAU * policy_param.data + (1.0 - TAU) * target_param.data
                )

            if done:
                episode_durations.append(t + 1)
                break

        print(f"Episode {episode} finished after {t} timesteps")

    print("Training complete")

    env.close()

    plt.figure()
    plt.plot(episode_durations)
