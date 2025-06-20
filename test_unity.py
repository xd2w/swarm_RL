import os
import shutil
import time
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter

from modules.ddpg_agent import DDPGAgent
from modules.sac_agent import SACAgent
from modules.gymlike_wrapper import UnityGymWrapper


def main(args):
    run_id = args.run_id
    unity_env = UnityEnvironment(
        file_name="../../../../../Maze gen/Builds/Maze_small_seed.app"
    )
    env = UnityGymWrapper(unity_env)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    # Load trained agent
    agent = DDPGAgent(state_dim, action_dim, max_action)
    agent.load(
        f"runs/{run_id}/{run_id}"
    )  # assumes model files are named like ddpg_pendulum_actor.pth

    # Run test episodes
    test_episodes = 50
    steps_per_episode = 200

    action = np.random.uniform(-max_action, max_action, (env.n_agents, action_dim))
    state, reward, done = env.step(action)

    for episode in range(test_episodes):
        episode_reward = 0
        for step in range(steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            # env.render()

        state = env.reset()

        # print(f"[Test] Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("run_id", type=str, default="run01")
    args = argparse.parse_args()
    main(args)
