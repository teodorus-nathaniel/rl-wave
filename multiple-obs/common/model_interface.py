from typing import Tuple
from matplotlib import pyplot as plt
import torch
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
        _, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2)
        ax1.set_title("Loss")
        ax1.plot(plot.avg_per_x_element(self.train_losses, smooth))
        ax2.set_title("Timesteps")
        ax2.plot(plot.avg_per_x_element(self.train_timesteps, smooth))
        ax3.set_title("Rewards")
        ax3.plot(plot.avg_per_x_element(self.train_rewards, smooth))
        plt.show()

    def train(self, env, epoch, reset_memory=False, show_plot=True):
        if reset_memory:
            self.reset_train_memory()

    def test(self, env) -> Tuple[float, float]:
        return 0, 0

    def test_avg(self, count=10, time_scale=10):
        sum_rewards = 0
        sum_timesteps = 0
        for _ in range(count):
            reward, timesteps = self.test(time_scale)
            sum_rewards += reward
            sum_timesteps += timesteps
        return (sum_rewards / count, sum_timesteps / count)

    def set_model_save_path(self, path):
        self.save_path = path

    def load_model(self, path=""):
        path = path if path != "" else self.save_path
        try:
            self.model.load_state_dict(torch.load(path))
            print("Model loaded")
        except:
            print("No model available")

    def save_model(self, path=""):
        path = path if path != "" else self.save_path
        torch.save(self.model.state_dict(), path)
        print("Model Saved")
