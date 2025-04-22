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


class Server:
    def __init__(self, actor):
        self.actor = actor
        self.model_state = {}

    def average_models(self, clients):
        avg_model = {}
        n_client = len(clients)
        client_models = [client.state_dict() for client in clients]
        for key in client_models[0].keys():
            avg_model[key] = sum(model[key] for model in client_models) / n_client
        self.actor.load_state_dict(avg_model)
        return avg_model

    def distribute_model(self, clients):
        self.model_state = self.actor.state_dict()
        for client in clients:
            client.load_state_dict(self.model_state)

    def state_dict(self):
        return self.actor.state_dict()


def main(args):
    config_file = args.config
    run_id = args.run_id
    force = args.force
    no_graphics = args.no_graphics
    indivisual_rewards = args.indivisual_rewards

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    save_freq = config["checkpoint"]["save_freq"]

    run_dir = f"runs/{run_id}"
    if os.path.exists(run_dir):
        if force:
            shutil.rmtree(run_dir)
        else:
            print("\nRun already exists, do --force to overwrite\n")
            exit()

    epochs = config["hyperparameters"]["epochs"]
    lr_actor = config["hyperparameters"]["lr_actor"]
    lr_critic = config["hyperparameters"]["lr_critic"]
    gamma = config["hyperparameters"]["gamma"]
    tau = config["hyperparameters"]["tau"]
    noise_sigma = config["hyperparameters"]["noise_sigma"]

    hyperparam = config["hyperparameters"]

    batch_size = config["memory"]["batch_size"]
    buffer_size = config["memory"]["buffer_size"]

    unity_env = UnityEnvironment(
        file_name="../../../../../Maze gen/Builds/Maze_small_seed.app",
        no_graphics=no_graphics,
    )
    env = UnityGymWrapper(unity_env)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1
    agents = []

    # agent = DDPGAgent(state_dim, action_dim, max_action, log_dir=f"runs/{run_id}/"
    server = Server(
        SACAgent(state_dim, action_dim, max_action, log_dir=f"runs/{run_id}/logs"),
    )

    for k in range(env.n_agents):
        agents.append(
            SACAgent(
                state_dim, action_dim, max_action, log_dir=f"runs/{run_id}/{k}/logs"
            )
        )

    server.distribute_model(agents)

    try:
        ave_episodes = 5
        reward_history = []
        start = time.time()

        for episode in range(epochs):
            states = env.reset()
            episode_reward = 0
            rewards_hist = np.zeros(200)
            done = [False] * env.n_agents
            actions = np.zeros((env.n_agents, 2))
            step = 0

            if episode % ave_episodes == 0:
                server.average_models(agents)
                server.distribute_model(agents)
                print("model avegeraged")

            while not any(done):
                # 200 steps
                for k in range(env.n_agents):
                    actions[k] = agents[k].select_action(states[k])
                next_states, reward, done = env.step(actions)
                for k in range(env.n_agents):
                    agents[k].remember(
                        states[k], actions[k], reward[k], next_states[k], done[k]
                    )
                    agents[k].train()
                states = next_states
                episode_reward += reward
                rewards_hist[step] = np.sum(reward)
                step += 1

            # print(step)
            reward_history.append(episode_reward)
            agents[0].writer.add_scalar(
                "Reward/Episode",
                np.sum(episode_reward) / env.n_agents,
                episode,
                time.time(),
            )
            agents[0].writer.add_histogram(
                "Reward", rewards_hist, episode, walltime=time.time()
            )
            # if indivisual_rewards:
            #     agent.writer.add_scalars(
            #         "Inidividual Reward ",
            #         {str(k): reward[k] for k in range(env.n_agents)},
            #         episode,
            #         time.time(),
            #     )
            print(
                f"{episode} epoch: {(time.time() - start):.0f}s"
                f": Reward = {np.sum(episode_reward) / env.n_agents:.2f}, variance = {np.var(episode_reward):.2f}"
            )

        server.average_models(agents)
        server.actor.save(f"./{run_id}/model")
        env.close()

    except KeyboardInterrupt:
        print("""\n\nTraining stopped by user: terminating and saving model.""")
        for k in range(env.n_agents):
            agents[k].save(f"./runs/{run_id}/models/{run_id}_{episode*200}_{k}")
        server.actor.save(f"./runs/{run_id}/models/{run_id}_{episode*200}")
        env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--run-id", type=str, default="run01")
    parser.add_argument("--no-graphics", action="store_true", default=False)
    parser.add_argument("--indivisual-rewards", action="store_true", default=False)
    parser.add_argument("--force", "-f", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
