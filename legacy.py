import math
import random
import argparse
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDPG(nn.Module):
    def __init__(self, n_observations, n_actions, h_nodes=128, action_smoothing=False):
        super(DDPG, self).__init__()
        self.layer1 = nn.Linear(n_observations, h_nodes)
        self.layer2 = nn.Linear(h_nodes, h_nodes)
        self.layer3 = nn.Linear(h_nodes, n_actions)
        self.log_std_head = nn.Linear(h_nodes, n_actions)

        self.action_smoothing = action_smoothing

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        # reparameterisation trick
        mu = self.log_std_head(x)
        action = torch.tanh(mu)

        if self.action_smoothing:
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mu, std)
            z = normal.rsample()
            action = torch.tanh(z)  # Squash the action to [-1, 1]
        return action


class Actor:
    def __init__(
        self,
        state_dim=8,
        action_dim=2,
        lr=0.001,
        gamma=0.99,
        epsilon_init=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        polyak=1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = ReplayMemory(buffer_size)
        self.model = DDPG(state_dim, action_dim)
        self.target = DDPG(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.polyak = polyak
        self.batch_size = 64

    def act(self, state):
        action = self.model(torch.tensor(state, dtype=torch.float32))
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update_target(self):
        for target_param, param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.polyak * param.data + (1 - self.polyak) * target_param.data
            )

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)

        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        next_state = torch.stack(next_state)
        done = torch.tensor(done).float().to(device)


def Unity_main():
    env = UnityEnvironment(
        file_name="../../../../../Maze gen/Builds/Maze_small_seed.app"
    )
    env.reset()
    behaviour_names = env.behavior_specs.keys()
    behaviour_name = list(behaviour_names)[0]
    spec = env.behavior_specs[behaviour_name]
    n_action = spec.action_spec.continuous_size

    # getting first observation to set params
    decision_steps, terminal_steps = env.get_steps(behaviour_name)
    n_agents = len(decision_steps.agent_id)
    if n_agents < 1:
        raise Exception("No agents")

    # give random action for first step
    actions = np.random.randn(n_agents, n_action)
    action_tuple = ActionTuple(continuous=actions)
    env.set_actions(behaviour_name, action_tuple)
    env.step()
    decision_steps, terminal_steps = env.get_steps(behaviour_name)

    for __ in range(1):
        total_ind_rw = np.zeros(n_agents)
        while any(terminal_steps.interrupted) != True:

            actions = np.random.randn(n_agents, n_action)
            action_tuple = ActionTuple(continuous=actions)
            env.set_actions(behaviour_name, action_tuple)
            env.step()

            print(decision_steps.obs)
            print(decision_steps.agent_id)
            decision_steps, terminal_steps = env.get_steps(behaviour_name)
            decision_steps

            total_ind_rw += decision_steps.reward
        print(terminal_steps.obs)
        print(terminal_steps.reward)
        print()
        # next time step not past
        print(decision_steps.obs)
        print(total_ind_rw)
        print(np.sum(total_ind_rw))
        print()
    exit()

    # one actor netwok decides action of n_agets in turn
    obs_in_use = np.zeros((n_agents, 8))
    prev_obs_in_use = obs_in_use.copy()
    actor = Actor(
        8,
        n_action,
        lr=0.001,
        gamma=0.99,
        epsilon_init=1.0,
        epsilon_decay=0.995,
        buffer_size=10000,
    )

    total_reward = 0
    for _ in range(2):
        obs = decision_steps.obs

        for k in range(n_agents):
            obs_in_use[k, :4] = obs[0][k][2:-1:3]
            obs_in_use[k, 4:] = obs[1][k]

        actions = actor.act(obs_in_use)

        action_tuple = ActionTuple(continuous=actions)
        env.set_actions(behaviour_name, action_tuple)
        env.step()

        prev_obs_in_use[:] = obs_in_use[:]

        decision_steps, terminal_steps = env.get_steps(behaviour_name)
        obs = decision_steps.obs
        rewards = decision_steps.reward
        done = (
            terminal_steps.interrupted
            if any(terminal_steps.interrupted)
            else [False] * n_agents
        )

        for k in range(n_agents):
            obs_in_use[k, :4] = obs[0][k][2:-1:3]
            obs_in_use[k, 4:] = obs[1][k]
            actor.remember(
                prev_obs_in_use[k], actions[k], rewards[k], obs_in_use[k], done[k]
            )

            total_reward += rewards[k]
            actor.replay(32)

    env.close()


if __name__ == "__main__":
    # par = argparse.ArgumentParser(
    #     prog="train.py", description="trains RL model", epilog=""
    # )
    # par.add_argument(
    #     "-i", "--run-id", type=str, help="path to input file", required=True
    # )

    # args = par.parse_args()
    run_id = "test"
    writer = SummaryWriter(f"./runs/{run_id}")
    Unity_main()
