import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

import env_interface


class WaveEnv(env_interface.EnvInterface):
    env: UnityEnvironment

    def __init__(self, path, timescale=30, no_graphics=False, worker_id=0):
        channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=path, no_graphics=no_graphics, seed=1, side_channels=[channel], worker_id=worker_id)
        channel.set_configuration_parameters(time_scale=timescale)
        print("WAVE environment created.")

    @staticmethod
    def preprocess_input(steps):
        state = np.append(steps.obs[1], steps.obs[0], axis=1)
        return state

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
        reward = current_step.reward[0]
        # if reward > 0.5:
        #     reward = 5.0
        return state, reward, is_done

    def reset(self):
        self.env.reset()
        state, _, is_done = self.step(1)
        return state, is_done

    def step(self, action):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]]))
        self.env.set_actions(self.get_behavior_name(), action_tuple)
        self.env.step()

        return self.get_current_state()

    def close(self):
        self.env.close()
