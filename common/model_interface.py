import os
from typing import Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

import plot


class ModelInterface:
    model = None
    save_path = ""
    train_losses = []
    train_rewards = []
    train_timesteps = []

    def reset_train_memory(self):
        self.train_losses = []
        self.train_rewards = []
        self.train_timesteps = []

    def plot_train_memory(self, smooth=10):
        sns.set(rc={'figure.figsize':(20, 8)})
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Training Data')

        ax1.set_title("Total Rewards per Episode")
        ax1.plot(plot.smooth_values(self.train_rewards, smooth))

        ax2.set_title("Timesteps per Episode")
        ax2.plot(plot.smooth_values(self.train_timesteps, smooth))

        plt.show()

    def train(self, env, epoch, reset_memory=False, show_plot=True):
        if reset_memory:
            self.reset_train_memory()

    def test(self, env) -> Tuple[float, float]:
        return 0, 0

    def test_avg(self, env_generator, count=10, time_scale=10):
        sum_rewards = 0
        sum_timesteps = 0
        for i in range(count):
            env = env_generator(time_scale)
            reward, timesteps = self.test(env)
            sum_rewards += reward
            sum_timesteps += timesteps
        return (sum_rewards / count, sum_timesteps / count)

    def set_model_save_path(self, path):
        self.save_path = path

    def get_all_paths(self, path=""):
        path = path if path != "" else self.save_path
        return (
            f"{path}/model.pth",
            f"{path}/train_rewards.csv",
            f"{path}/train_timesteps.csv",
            f"{path}/train_losses.csv",
        )

    def load_model(self, path="", custom_model=False):
        (
            model_path,
            train_rewards_path,
            train_timesteps_path,
            train_losses_path,
        ) = self.get_all_paths(path)
        try:
            if not custom_model:
                self.model.load_state_dict(torch.load(model_path))
                print("Model loaded")
        except Exception as _:
            print("No model available")

        try:
            self.train_rewards = np.loadtxt(train_rewards_path, delimiter=",").tolist()
            self.train_losses = np.loadtxt(train_losses_path, delimiter=",").tolist()
            self.train_timesteps = np.loadtxt(
                train_timesteps_path, delimiter=","
            ).tolist()
            print("Training history loaded")
        except Exception as e:
            print("Error load training history", e)

    def save_model(self, path="", custom_model=False):
        current_path = path if path != "" else self.save_path
        os.mkdir(current_path)
        (
            model_path,
            train_rewards_path,
            train_timesteps_path,
            train_losses_path,
        ) = self.get_all_paths(path)

        if not custom_model:
            torch.save(self.model.state_dict(), model_path)
            print("Model saved")

        np.savetxt(train_rewards_path, np.asarray(self.train_rewards), delimiter=",")
        np.savetxt(train_losses_path, np.asarray(self.train_losses), delimiter=",")
        np.savetxt(
            train_timesteps_path, np.asarray(self.train_timesteps), delimiter=","
        )
        print("Training history saved")
