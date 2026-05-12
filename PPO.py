import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def orthogonal_init(layer: nn.Module, gain=1.0):
    """
    Apply orthogonal initialization to a neural network layer.

    Args:
        layer (nn.Module): Layer to initialize.
        gain (float): Initialization gain.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    """
    Actor network for continuous actions based on a Beta distribution.
    The actor uses its own fully connected feature extractor.
    """

    def __init__(self, args):
        """
        Initialize the actor network.

        Args:
            args: Argument namespace containing network configuration.
        """
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.alpha_layer = nn.Linear(128, args.action_dim)
        self.beta_layer = nn.Linear(128, args.action_dim)
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, x):
        """Compute the alpha and beta parameters from input states."""
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        alpha = F.softplus(self.alpha_layer(x)) + 1.0
        beta = F.softplus(self.beta_layer(x)) + 1.0
        return alpha, beta

    def get_dist(self, x):
        """Build a Beta distribution for the input states."""
        alpha, beta = self.forward(x)
        dist = Beta(alpha, beta)
        return dist


class Critic(nn.Module):
    """
    Critic network for state-value estimation.
    The critic uses its own fully connected feature extractor.
    """

    def __init__(self, args):
        """
        Initialize the critic network.

        Args:
            args: Argument namespace containing network configuration.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        """Compute state values."""
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class PPO(object):
    """
    Proximal Policy Optimization agent.
    PPO improves training stability with a clipped objective and multiple update epochs.
    """

    def __init__(self, args):
        """Initialize the PPO agent."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor_Beta(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        self.lr_a = getattr(args, 'ppo_lr_a', 3e-4)
        self.lr_c = getattr(args, 'ppo_lr_c', 3e-4)
        self.eps_clip = getattr(args, 'ppo_eps_clip', 0.2)
        self.k_epochs = getattr(args, 'ppo_k_epochs', 4)
        self.gamma = getattr(args, 'ppo_gamma', 0.99)
        self.gae_lambda = getattr(args, 'ppo_gae_lambda', 0.95)
        self.max_step = args.max_step

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5, weight_decay=5e-4)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5, weight_decay=5e-4)

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def action(self, state):
        """
        Sample an action for the current state without gradient tracking.
        The state, action, log probability, and value are stored for later PPO updates.
        """
        with torch.no_grad():
            state = state.to(self.device)
            dist = self.actor.get_dist(state)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            value = self.critic(state)

            self.states.append(state.cpu())
            self.actions.append(a.cpu())
            self.log_probs.append(log_prob.cpu())
            self.values.append(value.cpu())

            return a

    def store_reward(self, reward, done=False):
        """Store rewards and done flags."""
        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32)
        elif not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value=0):
        """Compute generalized advantage estimation (GAE)."""
        advantages = []

        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_i = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_i = self.values[i + 1]

            reward_i = self.rewards[i]
            if isinstance(reward_i, np.ndarray):
                reward_i = torch.tensor(reward_i, dtype=torch.float32)
            elif not isinstance(reward_i, torch.Tensor):
                reward_i = torch.tensor(reward_i, dtype=torch.float32)

            value_i = self.values[i]
            if isinstance(next_value_i, (int, float)):
                next_value_i = torch.tensor(next_value_i, dtype=torch.float32)

            if value_i.dim() == 2 and value_i.shape[-1] == 1:
                value_i = value_i.squeeze(-1)
            if isinstance(next_value_i, torch.Tensor) and next_value_i.dim() == 2 and next_value_i.shape[-1] == 1:
                next_value_i = next_value_i.squeeze(-1)
            if isinstance(reward_i, torch.Tensor) and reward_i.dim() > 1:
                reward_i = reward_i.squeeze()

            batch_size = reward_i.shape[0] if isinstance(reward_i, torch.Tensor) and reward_i.dim() > 0 else 1

            if isinstance(next_value_i, torch.Tensor):
                if next_value_i.dim() > 0 and next_value_i.shape[0] != batch_size:
                    next_value_i = torch.full((batch_size,), next_value_i.mean().item(), dtype=torch.float32)
                elif next_value_i.dim() == 0:
                    next_value_i = torch.full((batch_size,), next_value_i.item(), dtype=torch.float32)
            else:
                next_value_i = torch.full((batch_size,), next_value_i, dtype=torch.float32)

            if i == len(self.rewards) - 1:
                gae = torch.zeros(batch_size, dtype=torch.float32)
            else:
                if gae.shape[0] != batch_size:
                    gae = torch.zeros(batch_size, dtype=torch.float32)

            delta = reward_i + self.gamma * next_value_i * next_non_terminal - value_i
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        flattened_advantages = []
        for adv in advantages:
            if isinstance(adv, torch.Tensor):
                flattened_advantages.append(adv.view(-1))
            else:
                flattened_advantages.append(torch.tensor(adv).view(-1))

        return flattened_advantages

    def update(self, state=None, action=None, reward=None):
        """
        Update actor and critic networks using collected experiences.
        PPO uses multiple update epochs and a clipped surrogate objective.
        """
        if len(self.states) == 0:
            return

        if reward is not None:
            if len(reward.shape) > 1:
                reward = reward.mean(dim=1)
            self.store_reward(reward.cpu().numpy())

        advantages = self.compute_gae()

        states = torch.cat(self.states).to(self.device)
        actions = torch.cat(self.actions).to(self.device)
        old_log_probs = torch.cat(self.log_probs).to(self.device)
        values = torch.cat(self.values).to(self.device)

        if len(advantages) > 0 and isinstance(advantages[0], torch.Tensor):
            advantages = torch.cat(advantages).to(self.device)
        else:
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        if advantages.dim() == 2 and advantages.shape[-1] == 1:
            advantages = advantages.squeeze(-1)
        if values.dim() == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        if advantages.shape != values.shape:
            print(f"Debug: advantages shape: {advantages.shape}, values shape: {values.shape}")
            if advantages.numel() == values.numel():
                advantages = advantages.view(-1)
                values = values.view(-1)
            else:
                raise ValueError(f"Cannot match advantages {advantages.shape} with values {values.shape}")

        returns = advantages + values

        # Scale advantages without subtracting the mean to avoid gradient cancellation when ratio is 1.
        advantages = advantages / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0

        for i_epoch in range(self.k_epochs):
            dist = self.actor.get_dist(states)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

            current_values = self.critic(states)
            if current_values.dim() == 2 and current_values.shape[-1] == 1:
                current_values = current_values.squeeze(-1)
            if returns.dim() == 2 and returns.shape[-1] == 1:
                returns = returns.squeeze(-1)

            if current_values.shape != returns.shape:
                if current_values.numel() == returns.numel():
                    current_values = current_values.view(-1)
                    returns = returns.view(-1)
                else:
                    raise ValueError(f"Cannot match shapes: current_values {current_values.shape} vs returns {returns.shape}")

            critic_loss = F.mse_loss(current_values, returns)

            self.optim_critic.zero_grad()
            critic_loss.backward()
            self.optim_critic.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        self.clear_buffer()

        return total_actor_loss / self.k_epochs, total_critic_loss / self.k_epochs

    def clear_buffer(self):
        """Clear the experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def lr_decay(self, steps):
        """Linearly decay the actor and critic learning rates."""
        if self.max_step <= 1:
            lr_a_now = self.lr_a
            lr_c_now = self.lr_c
        else:
            discount = 1 - 0.99 / (self.max_step - 1) * steps
            lr_a_now = self.lr_a * discount
            lr_c_now = self.lr_c * discount

        for p in self.optim_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optim_critic.param_groups:
            p['lr'] = lr_c_now
