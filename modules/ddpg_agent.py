import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

from torch.utils.tensorboard import SummaryWriter

from modules.actor_acitic import Actor, Critic, ReplayBuffer


class DDPGAgent:
    def __init__(self, state_dim, action_dim, config, log_dir="runs/DDPG"):

        hyperparam = config["hyperparameters"]
        mem_param = config["memory"]

        self.gamma = hyperparam["gamma"]
        self.tau = hyperparam["tau"]
        self.batch_size = mem_param["batch_size"]
        self.replay_buffer_size = mem_param["buffer_size"]
        self.noise = hyperparam["noise"]

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = 1
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = 64
        self.noise = 0.1

        self.writer = SummaryWriter(log_dir)
        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().cpu().numpy()[0]
        action += np.random.normal(0, self.noise, size=action.shape)
        return action.clip(-self.max_action, self.max_action)

    def remember(self, *args):
        self.replay_buffer.add(*args)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Critic update
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            y = rewards + self.gamma * target_q * (1 - dones)

        critic_loss = nn.MSELoss()(self.critic(states, actions), y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target networks update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        # Log
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.total_steps)
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.total_steps)
        self.total_steps += 1

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filename="ddpg_model"):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load(self, filename="ddpg_model"):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict)

    def state_dict(self):
        return self.actor.state_dict()

    # def __del__(self):
    #     self.writer.close()
