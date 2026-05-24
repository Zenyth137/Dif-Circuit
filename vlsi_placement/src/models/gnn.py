"""
Edge Graph Neural Network (Edge-GNN / Edge-GAT) for circuit netlist topology embedding.

Models the circuit as a bipartite graph:
- Module nodes (with features: width, height)
- Net nodes (with feature: pin count)
- Edges: module <-> net, with edge weight (1/pin_count)

Uses Edge-GAT convolution to pass messages with edge features.

Reference:
  - Circuit GNN / Edge-GAT for netlist modeling (VLSI.tex Section 2.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EdgeGATConv(nn.Module):
    """
    Edge-featured Graph Attention Convolution layer.

    Computes attention coefficients that depend on both source node features
    and edge attributes.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 1,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.edge_dim = edge_dim

        assert out_dim % heads == 0, "out_dim must be divisible by heads"

        # Linear transformations
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.empty(1, heads, self.head_dim))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) source -> target indices
            edge_attr: (E, edge_dim) edge features

        Returns:
            out: (N, out_dim) updated node features
        """
        N = x.size(0)
        H = self.heads
        D = self.head_dim

        # Linear transform
        x_src = self.lin_src(x).view(N, H, D)  # (N, H, D)
        x_dst = self.lin_dst(x).view(N, H, D)
        edge_feat = self.lin_edge(edge_attr).view(-1, H, D)  # (E, H, D)

        src, dst = edge_index[0], edge_index[1]  # (E,), (E,)

        # Compute attention scores
        alpha_src = (x_src[src] * self.att_src).sum(dim=-1)  # (E, H)
        alpha_dst = (x_dst[dst] * self.att_dst).sum(dim=-1)
        alpha_edge = (edge_feat * self.att_edge).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst + alpha_edge, 0.2)

        # Softmax per destination node
        alpha = self._softmax_per_dst(alpha, dst, N)

        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted message passing
        msg = x_src[src] + edge_feat  # (E, H, D)
        out = torch.zeros(N, H, D, device=x.device)
        out.index_add_(0, dst, msg * alpha.unsqueeze(-1))

        return out.view(N, -1)

    @staticmethod
    def _softmax_per_dst(alpha: torch.Tensor, dst: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """Compute softmax over attention scores per destination node."""
        # Subtract max for numerical stability
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.index_reduce_(0, dst, alpha, 'amax', include_self=False)
        alpha = alpha - alpha_max[dst]

        alpha_exp = alpha.exp()
        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.index_add_(0, dst, alpha_exp)
        return alpha_exp / (alpha_sum[dst] + 1e-8)


class EdgeGNN(nn.Module):
    """
    Edge Graph Neural Network for netlist topology embedding.

    Uses bipartite graph representation:
    - Module nodes (index 0..N-1): features = [width, height]
    - Net nodes (index N..N+M-1): features = [pin_count]

    Architecture:
      Input → Linear → EdgeGATConv × L → Global Pool → Output
    """

    def __init__(self,
                 node_in_dim: int = 2,       # w, h for modules
                 net_in_dim: int = 1,         # pin_count for nets
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 edge_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections
        self.module_proj = nn.Linear(node_in_dim, hidden_dim)
        self.net_proj = nn.Linear(net_in_dim, hidden_dim)

        # Edge-GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EdgeGATConv(hidden_dim, hidden_dim, edge_dim, heads, dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                module_features: torch.Tensor,
                net_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            module_features: (N, 2) module width, height
            net_features: (M, 1) pin count per net
            edge_index: (2, E) bipartite edges (modules <-> nets)
            edge_attr: (E, 1) edge weights

        Returns:
            module_embeddings: (N, hidden_dim) per-module topology embeddings
        """
        # Project to hidden dim
        x_modules = self.module_proj(module_features)
        x_nets = self.net_proj(net_features)
        x = torch.cat([x_modules, x_nets], dim=0)  # (N+M, hidden_dim)

        # Edge-GAT layers
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x + x_res)
            x = F.relu(x)
            x = self.dropout(x)

        # Return only module embeddings
        num_modules = module_features.size(0)
        return x[:num_modules]


class EdgeGNNEncoder(nn.Module):
    """
    Full Edge-GNN encoder that produces both per-module embeddings
    and a global graph embedding (for use in policy/value networks).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.gnn = EdgeGNN(**kwargs)
        hidden_dim = kwargs.get('hidden_dim', 128)
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            module_embeddings: (N, hidden_dim)
            global_embedding: (hidden_dim,) pooled graph representation
        """
        module_embs = self.gnn(*args, **kwargs)
        global_emb = module_embs.mean(dim=0)  # mean pooling
        global_emb = self.global_pool(global_emb)
        return module_embs, global_emb
