import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Hyperparameters
state_dim = 8  # State space dimension
action_dim = 2  # Action space dimension (2 continuous actions)
hidden_dim = 256
batch_size = 64
gamma = 0.99  # Discount factor
tau = 0.005  # Target update soft parameter
alpha = 0.2  # Entropy coefficient
lr = 3e-4  # Learning rate
buffer_size = 100000  # Replay buffer size


# Define the Actor (policy) network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, noise_size=0.1):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.mu_head = nn.Linear(hidden_dim, action_dim)
        # self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.noise_size = noise_size

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(
            self.fc3(x) + self.noise_size * np.random.normal(0, 1, action_dim)
        )
        return action


# Define the Critic (Q-value) network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value_head(x)
        return q_value


# Soft Q-target network (for stability)
class TargetCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(TargetCriticNetwork, self).__init__()
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        return self.critic(state, action)


# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Initialize the networks
actor = ActorNetwork(state_dim, action_dim, hidden_dim)
critic1 = CriticNetwork(state_dim, action_dim, hidden_dim)
critic2 = CriticNetwork(state_dim, action_dim, hidden_dim)
target_critic1 = TargetCriticNetwork(state_dim, action_dim, hidden_dim)
target_critic2 = TargetCriticNetwork(state_dim, action_dim, hidden_dim)

# Copy the parameters to target networks
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=lr)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=lr)

# Replay buffer
replay_buffer = ReplayBuffer(buffer_size)


# SAC Training loop for 10 agents
def train_step():
    if replay_buffer.size() < batch_size:
        return  # Not enough samples in the buffer to perform a training step

    # Sample a batch of experiences
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards).view(-1, 1)
    next_states = torch.stack(next_states)
    dones = torch.stack(dones).view(-1, 1)

    # Critic update
    with torch.no_grad():
        next_actions, next_log_probs, _, _ = actor(next_states)
        next_q1 = target_critic1(next_states, next_actions)
        next_q2 = target_critic2(next_states, next_actions)
        next_q_value = torch.min(next_q1, next_q2) - alpha * next_log_probs
        target_q_value = rewards + (1 - dones) * gamma * next_q_value

    q1 = critic1(states, actions)
    q2 = critic2(states, actions)

    critic1_loss = F.mse_loss(q1, target_q_value)
    critic2_loss = F.mse_loss(q2, target_q_value)

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    # Actor (policy) update
    actions, log_probs, _, _ = actor(states)
    q1 = critic1(states, actions)
    q2 = critic2(states, actions)
    q_value = torch.min(q1, q2)
    actor_loss = (alpha * log_probs - q_value).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Target update (soft update)
    for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Training loop for 10 agents
num_episodes = 1000
for episode in range(num_episodes):
    states = torch.randn(10, state_dim)  # Simulate 10 agents (batch of 10 states)

    for agent_idx in range(10):
        state = states[agent_idx].unsqueeze(0)

        # Actor selects an action (output is in range [-1, 1])
        action, log_prob, _, _ = actor(state)

        # Simulate environment step (here you'd use ML-Agents to get real rewards)
        next_state = state + 0.1 * action  # Just an example of how the state evolves
        reward = torch.tensor([random.random()])  # Example: a random reward
        done = torch.tensor([0])  # Assume episode doesn't end

        # Save experience in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Perform training step
        train_step()

    if episode % 100 == 0:
        print(f"Episode {episode} - Training step completed.")
