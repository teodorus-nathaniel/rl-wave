from calendar import firstweekday

import numpy as np
import torch
from tabulate import tabulate
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

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


class PPO(model_interface.ModelInterface):
    def __init__(
        self,
        input_layer,
        output_layer,
        hidden_layer=256,
        lr=1e-4,
        ppo_epochs=5,
        clip=0.2,
        minibatch_size=128,
    ):
        self.model = ActorCritic(input_layer, output_layer, hidden_layer)

        self.loss_fn = torch.nn.MSELoss()
        self.input_layer = input_layer
        self.output_layer = output_layer

        self.ppo_epochs = ppo_epochs
        self.clip = clip
        self.minibatch_size = minibatch_size

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
        return (data - data.mean()) / (data.std())

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

    def get_advantages_from_state(self, states, rewards, is_done, newest_state):
        if not is_done:
            _, last_value_pred = self.model(torch.Tensor(newest_state))
            appended_rewards = np.append(
                rewards, last_value_pred.detach().numpy()[0, 0]
            )
            discounted_rewards = self.discount_rewards(appended_rewards)[:-1]
        else:
            discounted_rewards = self.discount_rewards(rewards)

        states_tensor = torch.Tensor(states)
        _, values = self.model(states_tensor)
        detached_values = values.detach().numpy()

        advantages = self.get_advantages(detached_values.flatten(), discounted_rewards)
        return discounted_rewards, advantages

    @staticmethod
    def random_sample(inds, minibatch_size):
        inds = np.random.permutation(inds)
        batches = inds[: len(inds) // minibatch_size * minibatch_size].reshape(
            -1, minibatch_size
        )
        for batch in batches:
            yield torch.from_numpy(batch).long()
        remainder = len(inds) % minibatch_size
        if remainder:
            yield torch.from_numpy(np.array(inds[-remainder:]).reshape(-1,)).long()

    def update_model(self, states, actions, old_log_probs, returns, advantages):
        dist, values = self.model(states)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs.detach())
        first_term = ratio * advantages
        second_term = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages

        entropy = dist.entropy()

        actor_loss = -torch.mean(torch.min(first_term, second_term))
        critic_loss = self.loss_fn(values, returns.reshape(-1, 1))
        entropy_loss = -entropy.mean()

        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.train_losses.append(total_loss.item())

    def ppo_optimization(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.ppo_epochs):
            sampler = self.random_sample(states.shape[0], self.minibatch_size)
            for inds in sampler:
                mb_states = states[inds]
                mb_actions = actions[inds]
                mb_returns = returns[inds]
                mb_advantages = advantages[inds]
                mb_old_log_probs = old_log_probs[inds]
                self.update_model(
                    mb_states,
                    mb_actions,
                    mb_old_log_probs,
                    mb_returns,
                    mb_advantages,
                )

    def generate_trajectories(self, env):
        states = []
        actions = []
        rewards = []
        log_probs = []

        timestep = 0
        state, is_done = env.reset()
        episode_reward = 0

        while timestep < self.max_step and not is_done:
            timestep += 1

            state_tensor = torch.Tensor(state)

            dist, _ = self.model(state_tensor)
            action = dist.sample()
            action_item = action.squeeze().item()

            log_probs.append(torch.squeeze(dist.log_prob(action)).item())

            next_state, reward, is_done = env.step(action_item)
            episode_reward += reward

            states.append(state[0])
            actions.append(action_item)
            rewards.append(reward)

            state = next_state

        states, actions, rewards, log_probs = (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(log_probs),
        )
        returns, advantages = self.get_advantages_from_state(
            states,
            rewards,
            is_done,
            state,
        )

        return (
            torch.Tensor(states),
            torch.Tensor(actions).long(),
            torch.Tensor(log_probs),
            torch.Tensor(returns),
            torch.Tensor(advantages),
            episode_reward,
            timestep,
        )

    def train(
        self,
        env: env_interface.EnvInterface,
        epoch=1000,
        reset_memory=False,
        show_plot=True,
        save_interval=500,
        lr_decay_interval=500
    ):
        super().train(env, epoch, reset_memory)

        for i in range(epoch):
            (
                states,
                actions,
                log_probs,
                returns,
                advantages,
                episode_reward,
                timestep,
            ) = self.generate_trajectories(env)

            self.ppo_optimization(states, actions, log_probs, returns, advantages)
            self.train_rewards.append(episode_reward)
            self.train_timesteps.append(timestep)

            if show_plot and (i + 1) % self.plot_smooth == 0:
                plot.plot_res(self.train_rewards, f"PPO ({i + 1})", self.plot_smooth)

            if (i + 1) % save_interval == 0:
                path = self.save_path
                if i + 1 < epoch:
                    path = f"{self.save_path}-{i + 1}"
                self.save_model(path)
                print(f"saved to {path}")

            if lr_decay_interval and (i + 1) % lr_decay_interval == 0:
                self.scheduler.step()

            print(f"EPOCH: {i}, total reward: {episode_reward}, timestep: {timestep}, lr: {self.optimizer.param_groups[0]['lr']}")

        env.close()

    def test(self, env: env_interface.EnvInterface):
        state, is_done = env.reset()
        timestep = 0
        total_reward = 0
        while not is_done:
            timestep += 1
            predictions, values = self.model(torch.Tensor(state))
            predictions = predictions.probs.detach().numpy()

            action = np.argmax(predictions)
            state, reward, is_done = env.step(action)
            total_reward += reward

        return total_reward, timestep
