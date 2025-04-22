import gym
import torch
import numpy as np
from modules.ddpg_agent import DDPGAgent

# Load environment
env = gym.make("Pendulum-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Load trained agent
agent = DDPGAgent(state_dim, action_dim, max_action)
agent.load(
    "ddpg_pendulum"
)  # assumes model files are named like ddpg_pendulum_actor.pth

# Run test episodes
test_episodes = 5
steps_per_episode = 200

for episode in range(test_episodes):
    state = env.reset()[0]
    episode_reward = 0
    for step in range(steps_per_episode):
        action = agent.select_action(state, noise=0.0)  # No exploration noise
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        env.render()

    print(f"[Test] Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

env.close()
