import gym
import numpy as np

import env_interface


class CartPoleEnv(env_interface.EnvInterface):
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        print("CARTPOLE environment created.")

    def reset(self):
        state = self.env.reset()
        return np.array([state]), False

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return np.array([state]), reward, done

    def close(self):
        self.env.close()
