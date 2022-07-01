import copy
import random
from collections import deque

import numpy as np
import torch

import env_interface
import model_interface
import plot


class QLearning(model_interface.ModelInterface):
    target_model = None

    def __init__(self, input_layer, output_layer, hidden_layer=256, lr=1e-4):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, output_layer),
        )
        self.sync_target_model()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.set_train_params()
        self.reset_train_memory()

    def reset_train_memory(self):
        super().reset_train_memory()
        self.experiences = deque(maxlen=self.mem_size)

    @staticmethod
    def loss_fn(pred, target):
        return torch.mean(0.5 * (pred - target) ** 2)

    @staticmethod
    def copy_model(model):
        model2 = copy.deepcopy(model)
        model2.load_state_dict(model.state_dict())
        return model2

    def sync_target_model(self):
        self.target_model = self.copy_model(self.model)

    def set_train_params(
        self,
        mem_size=100000,
        start_epsilon=1,
        min_epsilon=0.05,
        max_step=500,
        batch_size=512,
        gamma=0.99,
        sync_interval=1000,
        plot_smooth=50,
        epsilon_decay=0.999,
        save_interval=500,
        lr_decay_interval=500
    ):
        self.mem_size = mem_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.max_step = max_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_interval = sync_interval
        self.plot_smooth = plot_smooth
        self.epsilon_decay = epsilon_decay
        self.save_interval = save_interval
        self.lr_decay_interval = lr_decay_interval

    def update_weights(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.loss_fn(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.item())

    def update_model(self):
        batch = random.sample(self.experiences, self.batch_size)
        batch_t = list(map(list, zip(*batch)))
        states = batch_t[0]
        actions = batch_t[1]
        rewards = batch_t[2]
        next_states = batch_t[3]
        is_dones = batch_t[4]

        states = torch.Tensor(states)
        actions_tensor = torch.Tensor(actions).long()
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        is_dones_tensor = torch.Tensor(is_dones)

        is_dones_indices = torch.where(is_dones_tensor == True)[0]

        qvals = self.model(states)
        qvals_next = self.target_model(next_states).detach()

        qvals[range(len(qvals)), actions] = (
            rewards + self.gamma * torch.max(qvals_next, axis=1).values
        )
        qvals[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()] = rewards[
            is_dones_indices.tolist()
        ]

        self.update_weights(states.tolist(), qvals.tolist())

    def train(
        self,
        env: env_interface.EnvInterface,
        epoch=1000,
        reset_memory=False,
        show_plot=True,
    ):
        super().train(env, epoch, reset_memory)

        start_episode = len(self.train_rewards)
        interval = 0
        for i in range(start_episode, start_episode + epoch):
            timestep = 0
            state, is_done = env.reset()
            episode_rewards = []
            while timestep < self.max_step and not is_done:
                timestep += 1
                interval += 1

                state_tensor = torch.Tensor(state)

                qval = self.model(state_tensor)
                if np.random.rand() > self.epsilon:
                    action = np.argmax(qval.detach().numpy())
                else:
                    action = np.random.randint(0, self.output_layer)

                next_state, reward, is_done = env.step(action)
                episode_rewards.append(reward)

                current_exp = (state[0], action, reward, next_state[0], is_done)
                self.experiences.append(current_exp)

                state = next_state

                if len(self.experiences) >= self.batch_size:
                    self.update_model()

                if interval % self.sync_interval == 0:
                    self.sync_target_model()

            current_reward = np.sum(episode_rewards)
            self.train_rewards.append(current_reward)
            self.train_timesteps.append(timestep)
            if show_plot and (i + 1) % self.plot_smooth == 0:
                plot.plot_res(
                    self.train_rewards,
                    f"Q-Learning with Exp Replay and Target ({i + 1})",
                    self.plot_smooth,
                    self.train_losses
                )

            if (i + 1) % self.save_interval == 0:
                path = self.save_path
                if i + 1 < epoch:
                    path = f"{self.save_path}-{i + 1}"
                self.save_model(path)
                print(f"saved to {path}")

            print(
                f"EPOCH: {i}, total reward: {current_reward}, timestep: {timestep}, epsilon: {self.epsilon}, lr: {self.optimizer.param_groups[0]['lr']}"
            )
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            if (i + 1) % self.lr_decay_interval == 0:
                self.scheduler.step()

        env.close()

    def test(self, env: env_interface.EnvInterface):
        state, is_done = env.reset()
        timestep = 0
        total_reward = 0
        while not is_done:
            timestep += 1
            preds = self.model(torch.Tensor(state)).detach().numpy()

            action = np.argmax(preds)
            state, reward, is_done = env.step(action)
            total_reward += reward

        env.close()
        return total_reward, timestep
