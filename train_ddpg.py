import gym
import numpy as np
import matplotlib.pyplot as plt
from modules.ddpg_agent import DDPGAgent

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent(state_dim, action_dim, max_action)

episodes = 200
steps_per_episode = 200
exploration_noise = 0.1
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    episode_reward = 0
    for step in range(steps_per_episode):
        action = agent.select_action(state, noise=exploration_noise)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        episode_reward += reward

    reward_history.append(episode_reward)
    agent.writer.add_scalar("Reward/Episode", episode_reward, episode)
    print(f"Episode {episode} | Reward: {episode_reward:.2f}")

agent.save("ddpg_pendulum")

# Plot
plt.plot(reward_history)
plt.title("DDPG Training Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.show()
