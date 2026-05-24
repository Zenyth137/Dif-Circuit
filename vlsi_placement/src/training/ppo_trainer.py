"""
PPO (Proximal Policy Optimization) Trainer for MDP placement agent.

Trains the PolicyNet to sequentially place macro modules on a grid,
maximizing the terminal reward (negative HPWL + congestion).

Architecture:
  1. Collect rollouts using current policy
  2. Compute advantages via GAE (Generalized Advantage Estimation)
  3. Update policy + value with PPO clipping objective
  4. Repeat for N epochs

Reference: VLSI.tex Section 3.1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time
from tqdm import tqdm

from ..models.policy_net import PolicyNet
from ..environment.mdp_env import MacroPlacementEnv, EnvConfig
from ..netlist.generator import NetlistGenerator, NetlistConfig


class PPOBuffer:
    """Stores trajectory data for PPO updates."""

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def store(self,
              state: torch.Tensor,
              action: torch.Tensor,
              log_prob: torch.Tensor,
              reward: float,
              value: torch.Tensor,
              done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """
    PPO trainer for macro placement policy.

    Key hyperparameters:
      - lr: learning rate
      - gamma: discount factor
      - lam: GAE lambda
      - clip_eps: PPO clipping epsilon
      - value_coef: value loss coefficient
      - entropy_coef: entropy bonus coefficient
      - epochs: number of PPO update epochs per batch
      - batch_size: number of episodes per batch
    """

    def __init__(self,
                 policy_net: PolicyNet,
                 env_config: EnvConfig,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 epochs: int = 10,
                 batch_size: int = 64,
                 max_grad_norm: float = 1.0,
                 device: str = "cpu"):
        self.policy = policy_net.to(device)
        self.env_config = env_config
        self.env = MacroPlacementEnv(env_config)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = PPOBuffer()

        # Metrics tracking
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_hpwls: deque = deque(maxlen=100)

    def collect_rollout(self,
                        generator: NetlistGenerator,
                        deterministic: bool = False) -> float:
        """
        Collect one episode of experience.

        Returns:
            episode_reward: total terminal reward
        """
        # Generate a netlist
        nodes, nets = generator.generate()
        adj = generator.build_adjacency()
        edge_index, edge_attr = generator.build_edge_index()

        # Convert to tensors
        module_features = torch.tensor(nodes[:, 1:], dtype=torch.float32, device=self.device)
        net_features = torch.ones(len(nets), 1, dtype=torch.float32, device=self.device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)

        # Reset environment
        self.env.reset(nodes, nets)

        episode_reward = 0.0
        total_hpwl = 0.0

        for step in range(self.env.num_modules):
            current_module = nodes[step]
            w, h = float(current_module[1]), float(current_module[2])

            # Get action
            result = self.policy.get_action(
                module_features, net_features, edge_index_t, edge_attr_t,
                w, h, deterministic=deterministic
            )

            action_idx = result["action_idx"]
            log_prob = result["log_prob"]
            value = result["value"]

            # Convert action_idx to grid position
            action_idx_int = int(action_idx.item())
            gy = action_idx_int % self.env.grid_size
            gx = action_idx_int // self.env.grid_size

            # Step environment
            state, reward, done, info = self.env.step((gx, gy))
            episode_reward += reward

            # Store in buffer
            state_feat = self.policy.forward_state(
                module_features, net_features, edge_index_t, edge_attr_t,
                *([float(nodes[min(step+1, len(nodes)-1)][1]), 
                   float(nodes[min(step+1, len(nodes)-1)][2])]
                  if step + 1 < len(nodes) else [0.0, 0.0])
            )

            self.buffer.store(state_feat, action_idx, log_prob, reward, value, done)

        # Compute final HPWL for logging
        final_positions = self.env.get_placement()
        from ..environment.reward import compute_hpwl
        total_hpwl = compute_hpwl(final_positions, nets)

        self.episode_rewards.append(episode_reward)
        self.episode_hpwls.append(total_hpwl)

        return episode_reward

    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        values = torch.stack([v.detach() for v in self.buffer.values])

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Perform one PPO update epoch."""
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self.compute_gae()

        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            # Shuffle
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Forward
                log_probs, values = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy = -log_probs.mean()

                # Total loss
                loss = (policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        num_updates = self.epochs * max(1, len(states) // self.batch_size)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def train(self,
              generator: NetlistGenerator,
              num_iterations: int = 1000,
              episodes_per_iter: int = 16,
              log_interval: int = 10,
              save_path: Optional[str] = None,
              early_stop_patience: int = 50) -> List[Dict]:
        """
        Main training loop.

        Args:
            generator: NetlistGenerator instance for sampling netlists
            num_iterations: number of PPO update iterations
            episodes_per_iter: episodes to collect per iteration
            log_interval: logging frequency
            save_path: path to save best model
            early_stop_patience: stop if no improvement for N iterations

        Returns:
            training_log: list of metrics per iteration
        """
        self.policy.train()
        log = []
        best_reward = -float('inf')
        patience_counter = 0

        pbar = tqdm(range(num_iterations), desc="PPO Training")
        for iteration in pbar:
            self.buffer.clear()

            # Collect episodes
            episode_rewards = []
            for _ in range(episodes_per_iter):
                r = self.collect_rollout(generator)
                episode_rewards.append(r)

            # PPO update
            metrics = self.update()
            avg_reward = np.mean(episode_rewards)
            avg_hpwl = np.mean(self.episode_hpwls) if self.episode_hpwls else 0.0

            metrics.update({
                "iteration": iteration,
                "avg_reward": avg_reward,
                "avg_hpwl": avg_hpwl,
                "running_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            })
            log.append(metrics)

            if iteration % log_interval == 0:
                pbar.set_postfix({
                    "reward": f"{avg_reward:.1f}",
                    "hpwl": f"{avg_hpwl:.0f}",
                })

            # Save best
            if avg_reward > best_reward:
                best_reward = avg_reward
                patience_counter = 0
                if save_path:
                    torch.save({
                        "policy_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "iteration": iteration,
                        "best_reward": best_reward,
                    }, save_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at iteration {iteration}")
                break

        return log

    def evaluate(self, generator: NetlistGenerator, num_episodes: int = 10) -> Dict:
        """Evaluate the trained policy."""
        self.policy.eval()
        rewards = []
        hpwls = []

        with torch.no_grad():
            for _ in range(num_episodes):
                r = self.collect_rollout(generator, deterministic=True)
                rewards.append(r)
                if self.episode_hpwls:
                    hpwls.append(self.episode_hpwls[-1])

        self.policy.train()
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_hpwl": np.mean(hpwls) if hpwls else 0.0,
            "std_hpwl": np.std(hpwls) if hpwls else 0.0,
        }
