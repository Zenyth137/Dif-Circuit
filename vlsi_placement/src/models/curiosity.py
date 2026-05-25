"""
Intrinsic Curiosity Module (ICM) for MDP exploration.

Architecture (Pathak et al., 2017):
  State encoder φ(s) → shared features
    ├── Forward dynamics: f(φ(s_t), a_t) → φ̂(s_{t+1})
    └── Inverse dynamics: g(φ(s_t), φ(s_{t+1})) → â_t

The forward prediction error ||φ̂(s_{t+1}) - φ(s_{t+1})||² is the
intrinsic reward — states whose consequences are hard to predict
are novel and worth exploring.

Inverse dynamics regularizes the state encoder to only capture
features that are *controllable* by the agent (not noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module.

    Args:
        state_dim: dimension of state feature vectors (e.g. policy hidden_dim)
        action_dim: number of discrete actions (grid_size × grid_size)
        hidden_dim: internal hidden dimension for both forward and inverse models
        action_embed_dim: dimension of learned action embeddings
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 action_embed_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim

        # Learned action embeddings (avoids huge one-hot vectors)
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)

        # Forward dynamics: f(φ(s_t), embed(a_t)) → φ̂(s_{t+1})
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Inverse dynamics: g(φ(s_t), φ(s_{t+1})) → â_t
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self,
                state_t: torch.Tensor,
                action_t: torch.Tensor,
                state_next: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ICM.

        Args:
            state_t: (B, state_dim) state features at time t
            action_t: (B,) long tensor of action indices
            state_next: (B, state_dim) state features at time t+1

        Returns:
            intr_reward: (B,) intrinsic reward per sample
            pred_next: (B, state_dim) predicted next state features
            pred_action_logits: (B, action_dim) predicted action logits
        """
        B = state_t.size(0)

        # Embed actions
        action_embed = self.action_embedding(action_t.long())  # (B, action_embed_dim)

        # Forward dynamics: predict next state
        forward_input = torch.cat([state_t, action_embed], dim=-1)
        pred_next = self.forward_model(forward_input)

        # Inverse dynamics: predict action from state pair
        inverse_input = torch.cat([state_t, state_next], dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)

        # Intrinsic reward = 0.5 * ||pred_next - state_next||² per sample
        # Sum over state dimensions, keep per-sample
        intr_reward = 0.5 * ((pred_next - state_next.detach()) ** 2).sum(dim=-1)

        return intr_reward, pred_next, pred_action_logits

    def compute_loss(self,
                     state_t: torch.Tensor,
                     action_t: torch.Tensor,
                     state_next: torch.Tensor,
                     forward_weight: float = 1.0,
                     inverse_weight: float = 0.2) -> Tuple[torch.Tensor, dict]:
        """
        Compute ICM training loss.

        Args:
            state_t, action_t, state_next: as in forward()
            forward_weight: weight for forward dynamics loss
            inverse_weight: weight for inverse dynamics loss

        Returns:
            total_loss: scalar loss for backprop
            metrics: dict with individual loss components
        """
        intr_reward, pred_next, pred_action_logits = self.forward(
            state_t, action_t, state_next
        )

        # Forward loss: MSE between predicted and actual next state
        forward_loss = F.mse_loss(pred_next, state_next.detach())

        # Inverse loss: cross-entropy for action prediction
        inverse_loss = F.cross_entropy(pred_action_logits, action_t.long())

        total_loss = forward_weight * forward_loss + inverse_weight * inverse_loss

        metrics = {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "mean_intr_reward": intr_reward.mean().item(),
        }

        return total_loss, metrics


class RunningNormalizer:
    """
    Running mean/std normalizer for intrinsic rewards.

    Keeps intrinsic reward scale stable across training, so the
    intrinsic-to-extrinsic balance doesn't drift.
    """

    def __init__(self, momentum: float = 0.01):
        self.momentum = momentum
        self.mean = 0.0
        self.std = 1.0
        self.initialized = False

    def update(self, values: torch.Tensor):
        """Update running statistics with a batch of values."""
        batch_mean = values.mean().item()
        # unbiased=False avoids NaN for single-element tensors
        batch_std = values.std(unbiased=False).item()
        # Fallback: use absolute mean if std is zero (all values identical)
        if batch_std < 1e-12:
            batch_std = abs(batch_mean) + 1e-8

        if not self.initialized:
            self.mean = batch_mean
            self.std = batch_std + 1e-8
            self.initialized = True
        else:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.std = (1 - self.momentum) * self.std + self.momentum * (batch_std + 1e-8)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values to zero mean, unit variance (approx)."""
        if not self.initialized:
            return values
        result = (values - self.mean) / (self.std + 1e-8)
        # NaN guard — fall back to raw values if normalization explodes
        if torch.isnan(result).any() or torch.isinf(result).any():
            return values
        return result
