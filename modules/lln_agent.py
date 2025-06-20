import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

from torch.utils.tensorboard import SummaryWriter

# from modules.actor_acitic import Critic

""" LNN Agents using T3D 

"""


class RecurrentReplayBuffer:
    def __init__(self, max_size=64):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        states, actions, rewards, next_states, dones = map(
            np.array, zip(self.buffer[-1])
        )
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions).unsqueeze(1),  # for single action
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class LiquidTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.liquid_step = LiquidTimeStep(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.h = torch.zeros(self.hidden_size)

    def forward(self, x):
        h_out = self.liquid_step(x, h=self.h)
        return self.output_layer(h_out)


# Actor using LNN
class LNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.liquid = LiquidNeuralNetwork(state_dim, hidden_dim, action_dim)
        self.max_action = 1

    def sample(self, state):
        state = torch.FloatTensor(state)
        action = self.liquid(state).detach()  # .cpu().numpy()
        # action += np.random.normal(0, 0.1, size=action.shape)
        action = torch.normal(action, 0.1)
        return action.clip(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)


class LNNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        log_dir="runs/LNN",
    ):

        self.gamma = 0.99
        self.tau = 0.01
        self.replay_size = 200
        self.alpha = 0.2
        self.policy_delay = 1

        hidden_dim = 64

        self.actor = LNNActor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target_optimizer = optim.Adam(
            self.critic_target.parameters(), lr=0.001
        )
        self.replay_buffer = RecurrentReplayBuffer(max_size=self.replay_size)

        self.writer = SummaryWriter(log_dir)
        self.step = 1

    def train(self):
        if len(self.replay_buffer) < 1:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            next_actions = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # new_actions = self.actor.sample(states)
        # q1_new, q2_new = self.critic(states, new_actions)
        # min_q_new = torch.min(q1_new, q2_new)
        # actor_loss = - min_q_new.mean()
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        if self.step % self.policy_delay == 0:
            actor_loss = self.actor_loss(states)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # self.soft_update(self.actor_target, self.actor)
            self.soft_update()

            self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.step)

        # Logging
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.step)
        # self.writer.add_scalar("Loss/Alpha", alpha_loss.item(), self.total_steps)
        # self.writer.add_scalar("Alpha", self.alpha, self.step)
        self.step += 1

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def soft_update(self):
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _train(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        critic_loss = self.critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self.actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.writer.add_scalar("loss/critic", critic_loss, global_step=self.step)
        self.writer.add_scalar("loss/actor", actor_loss, global_step=self.step)
        self.step += 1

    def actor_loss(self, state):
        action = self.actor.sample(state)
        q1, q2 = self.critic(state, action)
        q = torch.min(q1, q2)
        actor_loss = (-q).mean()
        return actor_loss

    def critic_loss(self, state, action, reward, next_state, done):
        next_action = self.actor.sample(next_state)
        next_q1, next_q2 = self.target_critic(next_state, next_action)
        min_next_q = torch.min(next_q1, next_q2)
        target_q = reward + self.gamma * (1 - done) * min_next_q
        q1, q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        return critic_loss

    def select_action(self, state):
        with torch.no_grad():
            action = self.actor.sample(state)
        return action

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
