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


def main():
    config_file = ""
    run_id = "run-id"
    force = False
    no_graphics = False
    # indivisual_rewards = args.indivisual_rewards

    # with open(config_file, "r") as f:
    #     config = yaml.safe_load(f)

    # epochs = config["hyperparameters"]["epochs"]
    epochs = 100
    # lr_actor = config["hyperparameters"]["lr_actor"]
    # lr_critic = config["hyperparameters"]["lr_critic"]
    # gamma = config["hyperparameters"]["gamma"]
    # tau = config["hyperparameters"]["tau"]
    # noise_sigma = config["hyperparameters"]["noise_sigma"]

    # hyperparam = config["hyperparameters"]

    # batch_size = config["memory"]["batch_size"]
    # buffer_size = config["memory"]["buffer_size"]
    # save_freq = config["checkpoint"]["save_freq"]

    unifrom_until = 10

    run_dir = f"runs/{run_id}"
    if os.path.exists(run_dir):
        if force:
            shutil.rmtree(run_dir)
        else:
            print("\nRun already exists, do --force to overwrite\n")
            exit()

    unity_env = UnityEnvironment(
        file_name="../../../../../Maze gen/Builds/Maze_small_no_penalty_no_com.app",
        no_graphics=no_graphics,
    )
    env = UnityGymWrapper(unity_env)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    # agent = DDPGAgent(state_dim, action_dim, config, log_dir=f"runs/{run_id}/logs")
    # agent = SACAgent(state_dim, action_dim, config, log_dir=f"runs/{run_id}/logs")
    # os.makedirs(f"./runs/{run_id}/models/", exist_ok=True)
    # shutil.copy2(config_file, f"./runs/{run_id}/config.yaml")

    run_id = "random_no_penalty_no_com"

    # writer = SummaryWriter(f"./runs/{run_id}/logs")

    try:
        reward_history = []
        start = time.time()

        rewards_hist = np.zeros((env.n_agents, epochs))

        for episode in range(1, epochs):
            state = env.reset()
            episode_reward = 0
            done = [False] * env.n_agents
            step = 0

            # if episode % save_freq == 0:
            # agent.save(f"./runs/{run_id}/models/{run_id}_{episode*200}")

            # random intial step for deferentiation
            # action = np.random.uniform(
            #     -max_action, max_action, (env.n_agents, action_dim)
            # )
            action = np.random.randn(env.n_agents, action_dim)
            next_state, reward, done = env.step(action)

            while not any(done):
                # 200 steps

                # unifrom initialization to promot
                # if episode < unifrom_until:
                #     action = np.random.uniform(
                #         -max_action, max_action, (env.n_agents, action_dim)
                #     )
                # else:
                # for k in range(env.n_agents):
                #     action[k] = agent.select_action(state[k])
                action = np.random.randn(env.n_agents, action_dim)

                next_state, reward, done = env.step(action)
                # for k in range(env.n_agents):
                #     agent.remember(
                #         state[k], action[k], reward[k], next_state[k], done[k]
                #     )
                # agent.train()
                state = next_state
                episode_reward += reward
                step += 1

            rewards_hist[:, episode] = episode_reward

            # print(step)
            reward_history.append(sum(episode_reward))

            # writer.add_scalar(
            #     "Reward/Episode",
            #     np.sum(episode_reward) / env.n_agents,
            #     episode,
            #     time.time(),
            # )
            # writer.add_histogram("Reward", rewards_hist, episode, walltime=time.time())
            # if indivisual_rewards:
            #     agent.writer.add_scalars(
            #         "Inidividual Reward ",
            #         {str(k): reward[k] for k in range(env.n_agents)},
            #         episode,
            #         time.time(),
            #     )
            print(
                f"{episode}\tep:\t{(time.time() - start):.0f}s"
                f":\tReward = {np.sum(episode_reward) / env.n_agents:.2f}\tvariance = {np.var(episode_reward):.2f}"
            )

        # agent.save(f"./runs/{run_id}")
        env.close()

        print(reward_history)
        print()
        print(np.sum(reward_history, axis=0))
        print()
        print()

        print(np.sum(rewards_hist, axis=1))
        print()
        print(sum(rewards_hist))
        print()

    except KeyboardInterrupt:
        print("""\n\nTraining stopped by user: terminating and saving model.""")
        # agent.save(f"./runs/{run_id}/models/{run_id}_{episode*200}")
        env.close()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, default="config.yaml")
    # parser.add_argument("--run-id", type=str, default="run01")
    # parser.add_argument("--no-graphics", action="store_true", default=False)
    # parser.add_argument("--indivisual-rewards", action="store_true", default=False)
    # parser.add_argument("--force", "-f", action="store_true", default=False)
    # args = parser.parse_args()

    main()
