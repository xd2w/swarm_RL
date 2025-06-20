from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class LiquidTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.liquid_step = LiquidTimeStep(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid_step(x[:, t, :], h)
        output = self.output_layer(h)
        return output


def LLM_Main():
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 2  # Output size for regression

    # Create the model
    model = LiquidNeuralNetwork(input_size, hidden_size, output_size)

    # Define Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


def Unity_main():
    env = UnityEnvironment(
        file_name="../../../../../Maze gen/Builds/Maze_small_pos.app"
    )
    env.reset()
    behaviour_names = env.behavior_specs.keys()
    behaviour_name = list(behaviour_names)[0]
    spec = env.behavior_specs[behaviour_name]
    print(spec)
    num_continuous_actions = spec.action_spec.continuous_size

    for episode in range(100):
        total_rewards = np.zeros_like
        for step in range(200):
            decision_steps, terminal_steps = env.get_steps(behaviour_name)

            # print()
            # print(len(decision_steps))

            num_agents = len(decision_steps.agent_id)

            if num_agents > 0:
                action = np.random.randn(num_agents, num_continuous_actions)
                obs = decision_steps.obs
                rewards = decision_steps.reward
                # print("observations")
                # print(obs[0])
                # print()
                # print(obs[1])
                # print()

                # print("rewards")
                # print(rewards)
                # print()

                # break
                action_tuple = ActionTuple(continuous=action)
                env.set_actions(behaviour_name, action_tuple)
            env.step()

    env.close()


if __name__ == "__main__":
    Unity_main()
