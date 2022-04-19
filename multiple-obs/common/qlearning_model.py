import random
from collections import deque

import numpy as np
import torch

import env_interface
import model_interface
import plot


class QLearning(model_interface.ModelInterface):
    def __init__(self, input_layer, output_layer, hidden_layer=256, lr=1e-4):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, output_layer),
        )
        self.loss_fn = torch.nn.MSELoss()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.epsilon = 1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.set_train_params()
        self.reset_train_memory()

    def reset_train_memory(self):
        super().reset_train_memory()
        self.experiences = deque(maxlen=self.mem_size)

    def set_train_params(self, mem_size=2500, start_epsilon=1, min_epsilon=.05, max_step=500, batch_size=512, gamma=.9):
        self.mem_size = mem_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.max_step = max_step
        self.batch_size = batch_size
        self.gamma = gamma

    def update_model(self):
        batch = random.sample(self.experiences, self.batch_size)
        batch_t = list(map(list, zip(*batch)))
        states = batch_t[0]
        actions = batch_t[1]
        rewards = batch_t[2]
        next_states = batch_t[3]
        is_dones = batch_t[4]

        states = torch.Tensor(states)
        actions_tensor = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        is_dones_tensor = torch.Tensor(is_dones)

        is_dones_indices = torch.where(is_dones_tensor == True)[0]

        qvals = self.model(states)
        qvals_next = self.model(next_states)

        qvals[range(len(qvals)),actions] = rewards + self.gamma * torch.max(qvals_next, axis=1).values
        qvals[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()] = rewards[is_dones_indices.tolist()]

        self.update_weights(states.tolist(), qvals.tolist())

    def update_weights(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.loss_fn(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.item())

    def train(self, env: env_interface.EnvInterface, epoch=1000, reset_memory=False):
        super().train(env, epoch, reset_memory)

        for i in range(epoch):
            timestep = 0
            state, is_done = env.reset()
            episode_reward = 0

            while timestep < self.max_step and not is_done:
                timestep += 1

                state_tensor = torch.Tensor(state)

                qval = self.model(state_tensor)
                if np.random.rand() > self.epsilon:
                    action = np.argmax(qval.detach().numpy())
                else:
                    action = np.random.randint(0, self.output_layer)

                next_state, reward, is_done = env.step(action)
                episode_reward += reward

                current_exp = (state[0], action, reward, next_state[0], is_done)
                self.experiences.append(current_exp)

                state = next_state

                if len(self.experiences) >= self.batch_size:
                    self.update_model()

            self.train_rewards.append(episode_reward)
            self.train_timesteps.append(timestep)
            plot.plot_res(self.train_rewards, f'Q-Learning with Exp Replay ({i + 1})')
            print(f'EPOCH: {i}, total reward: {episode_reward}, timestep: {timestep}, epsilon: {self.epsilon}')
            self.epsilon = max(self.epsilon - 1 / epoch, self.min_epsilon)

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
