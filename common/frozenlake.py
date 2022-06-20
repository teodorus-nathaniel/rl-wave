import gym
import numpy as np

import model_interface


class FrozenLake(model_interface.ModelInterface):
    def __init__(self):
        self.env = gym.make("FrozenLake-v1")
        print("FROZENLAKE environment created.")

    def preprocess_state(self, state):
        data = np.zeros((16))
        data[state] = 1
        return data

    def reset(self):
        state = self.env.reset()
        return np.array([self.preprocess_state(state)]), False

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return np.array([self.preprocess_state(state)]), reward, done

    def close(self):
        self.env.close()
