"""
Actor-Critic Policy Network for the MDP placement agent.

Architecture:
  GNN Encoder → shared features
    ├── Policy Head: outputs action logits over grid positions
    └── Value Head: outputs scalar state value

The policy head decomposes the action space (grid_size × grid_size)
into two independent marginal distributions (row + column),
or uses a fully-connected output layer for small grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .gnn import EdgeGNNEncoder


class PolicyHead(nn.Module):
    """Predicts action probabilities over grid positions."""

    def __init__(self, hidden_dim: int, grid_size: int, use_factorized: bool = True):
        super().__init__()
        self.grid_size = grid_size
        self.use_factorized = use_factorized

        if use_factorized:
            # Factorized: predict row and column independently
            self.row_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, grid_size),
            )
            self.col_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, grid_size),
            )
        else:
            # Flat: predict over all grid_size^2 positions
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, grid_size * grid_size),
            )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, hidden_dim) or (hidden_dim,) state features

        Returns:
            logits: (batch, grid_size, grid_size) action logits
            log_probs: same shape, log-softmax
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if self.use_factorized:
            row_logits = self.row_head(features)    # (B, G)
            col_logits = self.col_head(features)    # (B, G)
            # Outer product
            logits = row_logits.unsqueeze(-1) + col_logits.unsqueeze(-2)  # (B, G, G)
        else:
            logits = self.fc(features).view(-1, self.grid_size, self.grid_size)

        return logits

    def sample(self, features: torch.Tensor,
               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action_idx, log_prob)."""
        logits = self.forward(features)  # (B, G, G)
        flat_logits = logits.view(logits.size(0), -1)

        if deterministic:
            action_idx = flat_logits.argmax(dim=-1)
        else:
            probs = F.softmax(flat_logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).squeeze(-1)

        log_probs = F.log_softmax(flat_logits, dim=-1)
        action_log_prob = log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)

        return action_idx, action_log_prob

    def evaluate(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate log-probability of given actions."""
        logits = self.forward(features)
        flat_logits = logits.view(logits.size(0), -1)
        log_probs = F.log_softmax(flat_logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)


class ValueHead(nn.Module):
    """Predicts scalar state value V(s)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns (batch,) or scalar value."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.net(features).squeeze(-1)


class PolicyNet(nn.Module):
    """
    Full Actor-Critic Policy Network.

    Encodes netlist topology via Edge-GNN, then produces:
      - Action distribution over grid positions
      - State value estimate
    """

    def __init__(self,
                 grid_size: int = 64,
                 hidden_dim: int = 128,
                 gnn_layers: int = 3,
                 gnn_heads: int = 4,
                 use_factorized_policy: bool = True):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim

        # Shared GNN encoder
        self.encoder = EdgeGNNEncoder(
            node_in_dim=2,
            net_in_dim=1,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
        )

        # State feature projector (combines current module + global graph embedding)
        state_dim = 2 + hidden_dim  # [w, h] + global_graph_emb
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Heads
        self.policy_head = PolicyHead(hidden_dim, grid_size, use_factorized_policy)
        self.value_head = ValueHead(hidden_dim)

    def encode_graph(self,
                     module_features: torch.Tensor,
                     net_features: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode netlist graph once per episode."""
        return self.encoder(module_features, net_features, edge_index, edge_attr)

    def get_action(self,
                   module_features: torch.Tensor,
                   net_features: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: torch.Tensor,
                   current_module_w: float,
                   current_module_h: float,
                   deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Get action for the current step.

        Args:
            module_features: (N, 2) all module sizes
            net_features: (M, 1) net pin counts
            edge_index: (2, E) graph edges
            edge_attr: (E, 1) edge weights
            current_module_w, current_module_h: size of module to place

        Returns:
            dict with 'action_idx', 'log_prob', 'value'
        """
        _, global_emb = self.encode_graph(
            module_features, net_features, edge_index, edge_attr
        )
        # global_emb: (hidden_dim,)

        # Current module features
        curr_feat = torch.tensor(
            [current_module_w, current_module_h],
            device=global_emb.device,
            dtype=global_emb.dtype
        )

        # Combine state features
        state_feat = torch.cat([curr_feat, global_emb], dim=-1)
        state_feat = self.state_proj(state_feat)

        # Get action and value
        action_idx, log_prob = self.policy_head.sample(
            state_feat, deterministic=deterministic
        )
        value = self.value_head(state_feat)

        return {
            "action_idx": action_idx,
            "log_prob": log_prob,
            "value": value,
        }

    def evaluate_actions(self,
                         state_features: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_probs and values for given states and actions."""
        # state_features: (B, hidden_dim) already processed
        log_probs = self.policy_head.evaluate(state_features, actions)
        values = self.value_head(state_features)
        return log_probs, values

    def forward_state(self,
                      module_features: torch.Tensor,
                      net_features: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_attr: torch.Tensor,
                      current_w: float,
                      current_h: float) -> torch.Tensor:
        """Forward pass to get processed state features for storage."""
        _, global_emb = self.encode_graph(
            module_features, net_features, edge_index, edge_attr
        )
        curr_feat = torch.tensor(
            [current_w, current_h],
            device=global_emb.device,
            dtype=global_emb.dtype
        )
        state_feat = torch.cat([curr_feat, global_emb], dim=-1)
        return self.state_proj(state_feat)
