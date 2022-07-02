import numpy as np
import torch
from tabulate import tabulate
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

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

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
        self.gae_lambda = 0.95

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.set_train_params()
        self.reset_train_memory()
        self.buffer = PPOMemory(minibatch_size)

        self.train_nstep_rewards = []
        self.train_nstep_rewards_smoothed = []

    def set_train_params(self, max_step=1000, gamma=0.9, plot_smooth=50):
        self.plot_smooth = plot_smooth
        self.max_step = max_step
        self.gamma = gamma
        self.timestep = 0

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

    def get_advantages(self, rewards, values, is_dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)

        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma*values[k+1] * (1 - int(is_dones[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t
        advantages = torch.tensor(advantages)
        return advantages

    def update_model(self, states, actions, old_log_probs, old_vals, advantages):
        dist, values = self.model(states)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs.detach())
        first_term = ratio * advantages
        second_term = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages

        # entropy = dist.entropy()

        actor_loss = -torch.mean(torch.min(first_term, second_term))

        returns = advantages + old_vals
        critic_loss = self.loss_fn(values, returns.reshape(-1, 1))
        # entropy_loss = -entropy.mean()

        total_loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.train_losses.append(total_loss.item())

    def ppo_optimization(self):
        for _ in range(self.ppo_epochs):
            states, actions, old_log_probs, old_vals, rewards, is_dones, batches = self.buffer.generate_batches()
            advantages = self.get_advantages(rewards, old_vals, is_dones)
            for inds in batches:
                mb_states = torch.Tensor(states[inds])
                mb_actions = torch.Tensor(actions[inds]).long()
                mb_old_log_probs = torch.Tensor(old_log_probs[inds])
                mb_old_vals = torch.Tensor(old_vals[inds])
                mb_advantages = torch.Tensor(advantages[inds])
                self.update_model(
                    mb_states,
                    mb_actions,
                    mb_old_log_probs,
                    mb_old_vals,
                    mb_advantages,
                )

        self.buffer.clear_memory()

    def generate_trajectories(self, env):
        state, is_done = env.reset()
        episode_reward = 0
        eps_timestep = 0

        while not is_done and eps_timestep < self.max_step:
            self.timestep += 1
            eps_timestep += 1

            state_tensor = torch.Tensor(state)

            dist, value = self.model(state_tensor)
            action = dist.sample()
            action_item = action.squeeze().item()

            next_state, reward, is_done = env.step(action_item)
            episode_reward += reward
            self.train_nstep_rewards.append(reward)
            if len(self.train_nstep_rewards) >= self.max_step:
                self.train_nstep_rewards_smoothed.append(np.array(self.train_nstep_rewards).sum())
                self.train_nstep_rewards.clear()

            self.buffer.store_memory(state[0], action_item, \
                torch.squeeze(dist.log_prob(action)).item(),
                value.squeeze().item(), reward, is_done)

            if self.timestep % self.max_step == 0:
                self.ppo_optimization()

            state = next_state

        return (
            episode_reward,
            eps_timestep,
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

        start_episode = len(self.train_rewards)
        for i in range(start_episode, start_episode + epoch):
            episode_reward, timestep = self.generate_trajectories(env)

            self.train_rewards.append(episode_reward)
            self.train_timesteps.append(timestep)

            if show_plot and (i + 1) % self.plot_smooth == 0:
                plot.plot_res(self.train_rewards, f"PPO ({i + 1})", self.plot_smooth, self.train_nstep_rewards_smoothed)

            if (i + 1) % save_interval == 0:
                path = self.save_path
                if i + 1 < epoch:
                    path = f"{self.save_path}-{i + 1}"
                self.save_model(path)
                print(f"saved to {path}")

            if lr_decay_interval and (i + 1) % lr_decay_interval == 0:
                self.scheduler.step()

            print(f"epoch: {i}, total timestep: {self.timestep}, total reward: {episode_reward}, timestep: {timestep}, lr: {self.optimizer.param_groups[0]['lr']}")

        env.close()

    def test(self, env: env_interface.EnvInterface):
        state, is_done = env.reset()
        timestep = 0
        total_reward = 0
        while not is_done:
            timestep += 1
            predictions, values = self.model(torch.Tensor(state))
            predictions = predictions.probs.detach().numpy()
            print(predictions, values)

            action = np.argmax(predictions)
            state, reward, is_done = env.step(action)
            total_reward += reward

        return total_reward, timestep
