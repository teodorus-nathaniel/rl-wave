import random
from collections import deque

import numpy as np
import torch

import env_interface
import model_interface


class QLearning(model_interface.ModelInterface):
    def __init__(self, input_layer, output_layer, hidden_layer=256, lr=1e-4):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, output_layer),
        )
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.set_train_params()
        self.reset_train_memory()

    def reset_train_memory(self):
        super().reset_train_memory()
        self.experiences = deque(maxlen=self.mem_size)

    @staticmethod
    def loss_fn(pred, target):
        return torch.mean(0.5 * (pred - target) ** 2)

    def set_train_params(self, mem_size=2500, start_epsilon=1, min_epsilon=.05, max_step=500, batch_size=512, gamma=.9):
        self.mem_size = mem_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.max_step = max_step
        self.batch_size = batch_size
        self.gamma = gamma

    def update_model(self):
        batch = random.sample(self.experiences, self.batch_size)

        states = torch.cat([s for (s, a, r, s2, done) in batch])
        actions = torch.Tensor([a for (s, a, r, s2, done) in batch])
        states2 = torch.cat([s2 for (s, a, r, s2, done) in batch])
        done_data = torch.Tensor([done for (s, a, r, s2, done) in batch])
        rewards = torch.Tensor([r for (s, a, r, s2, done) in batch])

        qvals = self.model(states)

        with torch.no_grad():
            qvals_2 = self.model(states2)

        target = rewards + self.gamma * ((1 - done_data) * torch.max(qvals_2, dim=1)[0])
        action_qval_pred = qvals.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze()
        err = self.loss_fn(action_qval_pred, target.detach())
        self.train_losses.append(err.item())

        self.optimizer.zero_grad()
        err.backward()
        self.optimizer.step()

    def train(self, env: env_interface.EnvInterface, epoch=1000, reset_memory=False):
        super().train(env, epoch, reset_memory)

        for i in range(epoch):
            timestep = 0
            state, is_done = env.reset()
            episode_rewards = []

            while timestep < self.max_step and not is_done:
                timestep += 1

                state = torch.Tensor(state)

                qval = self.model(state)
                if np.random.rand() > self.epsilon:
                    action = np.argmax(qval.detach().numpy())
                else:
                    action = np.random.randint(0, self.output_layer)

                next_state, reward, is_done = env.step(action)
                episode_rewards.append(reward)

                state2 = torch.Tensor(next_state)
                current_exp = (state, action, reward, state2, is_done)
                self.experiences.append(current_exp)

                state = next_state

                if len(self.experiences) >= self.batch_size:
                    self.update_model()

            current_reward = np.sum(episode_rewards)
            self.train_rewards.append(current_reward)
            self.train_timesteps.append(timestep)
            print(f'EPOCH: {i}, total reward: {current_reward}, timestep: {timestep}, epsilon: {self.epsilon}')
            if self.epsilon > self.min_epsilon:
                self.epsilon -= 1 / epoch

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
