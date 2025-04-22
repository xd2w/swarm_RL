import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Reuse the LiquidNeuronLayer from earlier
class LiquidNeuronLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.tau_linear = nn.Linear(input_dim, hidden_dim)
        self.dt = dt

    def forward(self, x_seq):
        # Assume input is a single timestep (not full sequence)
        x = x_seq
        tau = F.softplus(self.tau_linear(x)) + 1e-2
        h = torch.zeros(x.shape[0], self.Wh.shape[0], device=x.device)

        z = torch.tanh(F.linear(x, self.Wx) + F.linear(h, self.Wh) + self.b)
        dh = (-1.0 / tau) * (h - z)
        h = h + self.dt * dh
        return h


# Actor using LNN
class LNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.liquid = LiquidNeuronLayer(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.liquid(state)
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h).clamp(-20, 2)  # SAC-style bounds
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)


# Critic using LNN
class LNNCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.liquid = LiquidNeuronLayer(state_dim + action_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        h = self.liquid(sa)
        return self.q(h)


class DDPGAgent:
    def __init__(
        self, state_dim, action_dim, hyperparam, log_dir="runs/DDPG", hidden_dim=64
    ):
        self.actor = LNNActor(state_dim, action_dim, hidden_dim)
        # self.actor_target = LNNActor(state_dim, action_dim, hidden_dim)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = LNNCritic(state_dim, action_dim, hidden_dim)
        # self.critic_target = LNNCritic(state_dim, action_dim, hidden_dim)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = 1
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.noise = hyperparam["noise"]  # 0.1

        self.writer = SummaryWriter(log_dir)
        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state).detach().cpu().numpy()
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

    def __del__(self):
        self.writer.close()
