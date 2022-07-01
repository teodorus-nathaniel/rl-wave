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
            torch.nn.Linear(hidden_layer, 1),
        )

    def forward(self, inp):
        actor_out = self.actor(inp)
        actor_out = Categorical(actor_out)

        critic_out = self.critic(inp)

        return actor_out, critic_out

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

    def set_train_params(self, max_step=1000, gamma=0.9, plot_smooth=50):
        self.plot_smooth = plot_smooth
        self.max_step = max_step
        self.gamma = gamma

    @staticmethod
    def normalize(data):
        return (data - data.mean()) / data.std()

    def discount_rewards(self, rewards: np.ndarray):
        reversed_rewards = np.copy(rewards)[::-1]
        discounted_rewards = []
        for i, reward in enumerate(reversed_rewards):
            discounted_rewards.append(
                reward + (0 if i == 0 else reversed_rewards[i - 1])
            )
            reversed_rewards[i] = reward * self.gamma
            if i > 0:
                reversed_rewards[i] += reversed_rewards[i - 1] * self.gamma
        discounted_rewards = np.array(discounted_rewards[::-1])
        return discounted_rewards

    def get_advantages(self, values, rewards):
        adv = rewards - values
        adv = np.array(adv)
        return self.normalize(adv)

    def update_model(self, states, actions, rewards, is_done, newest_state):
        states_tensor = torch.Tensor(states)
        dist, values = self.model(states_tensor)
        if not is_done:
            _, last_value_pred = self.model(torch.Tensor(newest_state))
            rewards = np.append(rewards, last_value_pred.detach().numpy()[0, 0])
            discounted_rewards = self.discount_rewards(rewards)[:-1]
        else:
            discounted_rewards = self.discount_rewards(rewards)

        entropy = dist.entropy()
        log_probs = dist.log_prob(torch.Tensor(actions))

        discounted_rewards = torch.Tensor(discounted_rewards)
        detached_values = values.detach().numpy()
        advantages = torch.Tensor(
            self.get_advantages(detached_values.flatten(), discounted_rewards)
        )

        entropy_loss = -entropy.mean()
        actor_loss = (advantages * -log_probs).mean()
        critic_loss = self.loss_fn(values, discounted_rewards.reshape(-1, 1))

        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.train_losses.append(total_loss.item())

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

        for i in range(epoch):
            timestep = 0
            state, is_done = env.reset()
            episode_reward = 0

            states = []
            actions = []
            rewards = []

            while timestep < self.max_step and not is_done:
                timestep += 1

                state_tensor = torch.Tensor(state)

                dist, _ = self.model(state_tensor)
                action = dist.sample()
                action_item = action.squeeze().item()

                next_state, reward, is_done = env.step(action_item)
                episode_reward += reward

                states.append(state[0])
                actions.append(action_item)
                rewards.append(reward)

                state = next_state

            states, actions, rewards = (
                np.array(states),
                np.array(actions),
                np.array(rewards),
            )
            self.update_model(states, actions, rewards, is_done, state)
            self.train_rewards.append(episode_reward)
            self.train_timesteps.append(timestep)

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

            print(f"EPOCH: {i}, total reward: {episode_reward}, timestep: {timestep}")

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
