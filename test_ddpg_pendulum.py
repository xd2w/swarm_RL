import gym
import torch
import numpy as np
from modules.ddpg_agent import DDPGAgent
from modules.sac_agent import SACAgent
from modules.lln_agent import LNNAgent
import yaml

# Load environment
env = gym.make("Pendulum-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load trained agent
# agent = SACAgent(state_dim, action_dim, config, log_dir="runs/DDPG2")
# agent = DDPGAgent(state_dim, action_dim, config, log_dir="runs/DDPG2")
agent = LNNAgent(state_dim, action_dim, log_dir="runs/DDPG2")
agent.load(
    "runs/LNN/pendulum"
)  # assumes model files are named like ddpg_pendulum_actor.pth

# Run test episodes
test_episodes = 1000
steps_per_episode = 50

for episode in range(test_episodes):
    state = env.reset()[0]
    episode_reward = 0
    for step in range(steps_per_episode):
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action.numpy())
        episode_reward += reward
        env.render()

    print(f"[Test] Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

env.close()
