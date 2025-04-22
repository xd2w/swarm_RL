import numpy as np

from mlagents_envs.base_env import ActionTuple


class UnityGymWrapper:
    def __init__(self, unity_env, obs_dim=8):
        """
        Wrapper to convert UnityEnvironment to Gym-like environment.

        Args:
            unity_env: UnityEnvironment instance.
        """

        self.unity_env = unity_env
        self.unity_env.reset()

        behaviour_names = self.unity_env.behavior_specs.keys()
        self.behaviour_name = list(behaviour_names)[0]

        spec = self.unity_env.behavior_specs[self.behaviour_name]
        self.action_dim = spec.action_spec.continuous_size
        self.state_dim = obs_dim

        # getting first observation to set params
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behaviour_name)
        self.n_agents = len(decision_steps.agent_id)

    def reset(self):
        """
        Resets the Unity environment and returns the initial observation.
        """
        self.unity_env.reset()
        decision_info, terminal_info = self.unity_env.get_steps(self.behaviour_name)
        obs = self._extract_observation(decision_info)
        return obs

    def step(self, action):
        """
        Steps through the Unity environment with the given action.

        Args:
            action: Action to take in the environment. (n_agnets, action_dim)

        Returns:
            A tuple (observation, reward, done, info).
        """

        # set actions (for all agents)
        action_tuple = ActionTuple(continuous=action)
        self.unity_env.set_actions(self.behaviour_name, action_tuple)

        # step
        self.unity_env.step()

        # get states and rewards
        d_info, t_info = self.unity_env.get_steps(self.behaviour_name)
        obs = self._extract_observation(d_info)
        reward = d_info.reward

        # if terminated need to get from t_info
        if any(t_info.interrupted) or t_info.interrupted:
            obs = self._extract_observation(t_info)
            reward = t_info.reward

        done = self._extract_done(t_info)

        return obs, reward, done

    def close(self):
        """
        Closes the Unity environment.
        """
        self.unity_env.close()

    def _extract_observation(self, info):
        """
        Extracts observation from Unity environment info.
        """
        obs = info.obs
        obs_in_use = np.zeros((self.n_agents, self.state_dim), dtype=np.float32)
        # fist 4 for raycast (5th is duplicate from obs)
        # 4+ for other observatons (any length works)

        for k in range(self.n_agents):
            obs_in_use[k, :4] = obs[0][k][2:-1:3]
            obs_in_use[k, 4:] = obs[1][k]

        return obs_in_use

    def _extract_done(self, t_info):
        if any(t_info.interrupted) or t_info.interrupted:
            return t_info.interrupted
        else:
            return [False] * self.n_agents

    def get_spec(self):
        return self.unity_env.behavior_specs[self.behaviour_name]
