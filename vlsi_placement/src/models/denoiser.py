"""
Denoising Network (Edge-GNN based noise predictor) for the diffusion model.

Architecture:
  Takes noisy coordinates X_t, time embedding t, and static conditions C
  (module sizes + netlist adjacency) as input.
  Uses Edge-GNN to predict the noise ε that was added to X_0.

Reference: VLSI.tex Section 3.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .gnn import EdgeGATConv


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embedding (from DDPM / Transformer)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps

        Returns:
            (batch, dim) time embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DenoiserNet(nn.Module):
    """
    Edge-GNN based noise predictor for diffusion placement.

    Inputs:
      - X_t: (N, 2) noisy module center coordinates at timestep t
      - t: scalar timestep
      - conditions C: module sizes (N, 2) + netlist edge_index + edge_attr

    Output:
      - ε_pred: (N, 2) predicted noise added to coordinates

    The network builds a graph where nodes are modules, and edges are
    defined by netlist connectivity. Each node has features:
      [x, y, w, h, time_emb]
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 time_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node feature projection: [x, y, w, h] → hidden_dim
        self.node_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge feature projection: [weight] → hidden_dim
        self.edge_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
        )

        # Message-passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.time_biases = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                EdgeGATConv(hidden_dim, hidden_dim,
                            edge_dim=hidden_dim // 4, heads=4, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.time_biases.append(nn.Linear(hidden_dim, hidden_dim))

        # Output: predict 2D noise per module
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                module_sizes: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (N, 2) noisy coordinates at timestep t
            t: scalar or (batch,) timestep
            module_sizes: (N, 2) width, height of each module
            edge_index: (2, E) module-module edges from netlist
            edge_attr: (E, 1) edge weights

        Returns:
            epsilon_pred: (N, 2) predicted noise
        """
        N = x_t.size(0)

        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        if t_emb.size(0) == 1:
            t_emb = t_emb.squeeze(0)  # (hidden_dim,)
        else:
            t_emb = t_emb.mean(0)     # average over batch

        # Node features: concat coordinates + sizes
        node_feat = torch.cat([x_t, module_sizes], dim=-1)  # (N, 4)
        h = self.node_proj(node_feat)  # (N, hidden_dim)

        # Edge features
        e = self.edge_proj(edge_attr)  # (E, hidden_dim//4)

        # Message passing with time conditioning
        for conv, norm, time_bias in zip(self.convs, self.norms, self.time_biases):
            h_res = h
            h = conv(h, edge_index, e)
            # Time conditioning
            h = h + time_bias(t_emb).unsqueeze(0)
            h = norm(h + h_res)
            h = F.relu(h)

        # Predict noise
        eps_pred = self.out_proj(h)  # (N, 2)
        return eps_pred
