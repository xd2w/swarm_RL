import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

from torch.utils.tensorboard import SummaryWriter

from modules.actor_acitic import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action * self.max_action, log_prob.sum(1, keepdim=True)
        # TODO check this // log_prob.sum(1, keepdim=True)


# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, config, log_dir="runs/SAC"):
        max_action = 1
        self.actor = Actor(state_dim, action_dim, 1)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        hyperparam = config["hyperparameters"]
        mem_param = config["memory"]

        self.max_action = 1
        self.replay_buffer = ReplayBuffer(max_size=mem_param["buffer_size"])

        lr = hyperparam["lr"]

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # self.gamma = 0.99
        # self.tau = 0.995
        # self.batch_size = 64
        # self.alpha = 1  # init alpha

        self.gamma = hyperparam["gamma"]
        self.tau = hyperparam["tau"]
        self.alpha = hyperparam["alpha"]

        self.batch_size = mem_param["batch_size"]

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.writer = SummaryWriter(log_dir)
        self.total_steps = 0

    def remember(self, *args):
        self.replay_buffer.add(*args)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1))
        if eval:
            mean, _ = self.actor(state)
            return torch.tanh(mean).detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - min_q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update
        self.soft_update(self.critic_target, self.critic)

        # Logging
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.total_steps)
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.total_steps)
        self.writer.add_scalar("Loss/Alpha", alpha_loss.item(), self.total_steps)
        self.writer.add_scalar("Alpha", self.alpha, self.total_steps)
        self.total_steps += 1

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filename="sac_model"):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load(self, filename="sac_model"):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))

    def state_dict(self):
        return self.actor.state_dict()

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict)

    # def __del__(self):
    #     self.writer.close()
