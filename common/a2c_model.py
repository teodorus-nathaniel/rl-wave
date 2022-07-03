import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

import env_interface
import model_interface
import plot


class ActorCritic(nn.Module):
    def __init__(self, input_layer, output_layer, hidden_layer=256):
        super(ActorCritic, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(input_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, output_layer),
            torch.nn.Softmax(dim=-1),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer, 1),
        )

    def forward(self, inp):
        actor_out = self.actor(inp)
        actor_out = Categorical(actor_out)

        critic_out = self.critic(inp)

        return actor_out, critic_out

class A2CMemory:
    def __init__(self):
        self.states = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store_memory(self, state, action, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_batch(self):
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class A2C(model_interface.ModelInterface):
    def __init__(self, input_layer, output_layer, hidden_layer=256, lr=1e-4):
        self.model = ActorCritic(input_layer, output_layer, hidden_layer)

        self.loss_fn = torch.nn.MSELoss()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.set_train_params()
        self.reset_train_memory()

        self.buffer = A2CMemory()

    def set_train_params(self, max_step=1000, gamma=0.9, plot_smooth=50, gae_lambda=0.95):
        self.gae_lambda = gae_lambda
        self.plot_smooth = plot_smooth
        self.max_step = max_step
        self.gamma = gamma

    @staticmethod
    def normalize(data):
        return (data - data.mean()) / data.std()

    def get_advantages(self, rewards, values, is_dones):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - int(is_dones[i])) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(is_dones[i])) * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return np.array(returns), self.normalize(adv)

    def update_model(self, last_val):
        states, actions, vals, rewards, is_dones = self.buffer.get_batch()
        appended_vals = np.append(vals, last_val)
        returns, advantages = self.get_advantages(rewards, appended_vals, is_dones)

        states = torch.Tensor(states)
        actions = torch.Tensor(actions).long()
        returns = torch.Tensor(returns)
        is_dones = torch.Tensor(is_dones)
        advantages = torch.Tensor(advantages)

        dist, values = self.model(states)
        log_probs = dist.log_prob(actions)

        actor_loss = (advantages * -log_probs).mean()
        critic_loss = self.loss_fn(values, returns.reshape(-1, 1))

        total_loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.train_losses.append(total_loss.item())
        self.buffer.clear_memory()

    def train(
        self,
        env: env_interface.EnvInterface,
        epoch=1000,
        reset_memory=False,
        show_plot=True,
        save_interval=500,
        lr_decay_interval=False
    ):
        super().train(env, epoch, reset_memory)

        timestep = 0
        for i in range(epoch):
            eps_timestep = 0
            state, is_done = env.reset()
            episode_reward = 0

            while eps_timestep < self.max_step and not is_done:
                eps_timestep += 1
                timestep += 1

                state_tensor = torch.Tensor(state)

                dist, values = self.model(state_tensor)
                action = dist.sample()
                action_item = action.squeeze().item()
                value = values.squeeze().item()

                next_state, reward, is_done = env.step(action_item)
                episode_reward += reward

                self.buffer.store_memory(state[0], action_item, value, reward, is_done)
                if timestep % self.max_step == 0:
                    val = 0
                    if not is_done:
                        _, val = self.model(torch.Tensor(next_state))
                        val = val.squeeze().item()
                    self.update_model(val)

                state = next_state

            self.train_rewards.append(episode_reward)
            self.train_timesteps.append(eps_timestep)

            if show_plot and (i + 1) % self.plot_smooth == 0:
                plot.plot_res(self.train_rewards, f"A2C ({i + 1})", self.plot_smooth)

            if (i + 1) % save_interval == 0:
                path = self.save_path
                if i + 1 < epoch:
                    path = f"{self.save_path}-{i + 1}"

                self.save_model(path)
                print(f"saved to {path}")

            if lr_decay_interval and (i + 1) % lr_decay_interval == 0:
                self.scheduler.step()

            print(f"EPOCH: {i}, total reward: {episode_reward}, timestep: {eps_timestep}")

        env.close()

    def test(self, env: env_interface.EnvInterface):
        state, is_done = env.reset()
        timestep = 0
        total_reward = 0
        while not is_done:
            timestep += 1
            predictions, _ = self.model(torch.Tensor(state))
            predictions = predictions.probs.detach().numpy()

            action = np.argmax(predictions)
            state, reward, is_done = env.step(action)
            total_reward += reward

        return total_reward, timestep
