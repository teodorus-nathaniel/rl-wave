import gym
import numpy as np
from gym import spaces
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, path, timescale=40):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-10., 10., (64,), np.float32)
        channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=path, seed=1, side_channels=[channel])
        channel.set_configuration_parameters(time_scale=timescale)
        print("WAVE environment created.")

    @staticmethod
    def preprocess_input(steps):
        obs1 = steps.obs[1]
        obs1 /= 4
        print(obs1)
        return np.append(obs1, steps.obs[0], axis=1).squeeze()

    def get_current_step(self):
        decision_steps, terminal_steps = self.env.get_steps(self.get_behavior_name())
        is_done = len(terminal_steps) > 0
        return decision_steps if not is_done else terminal_steps, is_done

    def get_behavior_name(self):
        behavior_name = list(self.env.behavior_specs)[0]
        return behavior_name

    def get_current_state(self):
        current_step, is_done = self.get_current_step()
        state = self.preprocess_input(current_step)
        reward = float(current_step.reward[0])
        return state, reward, is_done

    def reset(self):
        self.env.reset()
        state, _, _, _ = self.step(1)

        return state

    def step(self, action):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]]))
        self.env.set_actions(self.get_behavior_name(), action_tuple)
        self.env.step()

        state, reward, is_done = self.get_current_state()

        return state, reward, is_done, {}

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        raise Exception('not implemented')
