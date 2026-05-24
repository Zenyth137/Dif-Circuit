"""
Synthetic netlist generator for VLSI placement benchmarks.

Generates random netlists with configurable:
- Number of modules (2000 - 10000)
- Module dimensions (width, height sampled from realistic distributions)
- Random hyperedges (nets connecting multiple modules)
- Export to CSV format
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NetlistConfig:
    """Configuration for synthetic netlist generation."""
    num_modules: int = 2000
    num_nets: int = 2000
    canvas_width: float = 1000.0
    canvas_height: float = 1000.0
    # Module size distribution parameters (log-normal)
    min_width: float = 2.0
    max_width: float = 50.0
    min_height: float = 2.0
    max_height: float = 50.0
    size_mean: float = 3.0       # mean of log-normal
    size_std: float = 0.8        # std of log-normal
    # Net connectivity
    min_pins_per_net: int = 2
    max_pins_per_net: int = 10
    avg_pins_per_net: float = 4.0
    # Random seed
    seed: Optional[int] = None


class NetlistGenerator:
    """Generate synthetic circuit netlists for VLSI placement."""

    def __init__(self, config: NetlistConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        self._nodes: Optional[np.ndarray] = None
        self._nets: Optional[list] = None

    def generate(self) -> Tuple[np.ndarray, list]:
        """
        Generate a synthetic netlist.

        Returns:
            nodes: np.ndarray of shape (N, 3) with columns [id, width, height]
            nets: list of lists, each inner list contains module indices in that net
        """
        cfg = self.config
        N = cfg.num_modules

        # Generate module sizes from log-normal distribution
        widths = np.random.lognormal(
            mean=np.log(cfg.size_mean),
            sigma=cfg.size_std,
            size=N
        )
        heights = np.random.lognormal(
            mean=np.log(cfg.size_mean),
            sigma=cfg.size_std,
            size=N
        )

        # Clip to valid ranges
        widths = np.clip(widths, cfg.min_width, cfg.max_width)
        heights = np.clip(heights, cfg.min_height, cfg.max_height)

        # Build node array
        node_ids = np.arange(N)
        nodes = np.stack([node_ids, widths, heights], axis=1)

        # Generate random nets (hyperedges)
        nets = self._generate_nets(N)

        self._nodes = nodes
        self._nets = nets
        return nodes, nets

    def truncate_to_max_modules(self, max_modules: int) -> Tuple[np.ndarray, list]:
        """
        Keep the first ``max_modules`` modules and nets whose pins lie in that range.

        Updates internal state so ``build_edge_index()`` matches the truncated netlist.
        """
        if self._nodes is None or self._nets is None:
            raise ValueError("Must call generate() first.")

        from .prepare import truncate_netlist, attach_netlist_to_generator

        nodes, nets = truncate_netlist(
            self._nodes, self._nets, max_modules, self.config.min_pins_per_net
        )
        attach_netlist_to_generator(self, nodes, nets)
        return nodes, nets

    def _generate_nets(self, num_modules: int) -> list:
        """Generate random hyperedges connecting modules."""
        cfg = self.config
        nets = []
        # Use a power-law-like distribution for pins per net
        for net_id in range(cfg.num_nets):
            # Number of pins: clamp between min and max
            n_pins = np.random.randint(cfg.min_pins_per_net,
                                        cfg.max_pins_per_net + 1)
            n_pins = min(n_pins, num_modules)
            # Sample modules without replacement
            modules = np.random.choice(num_modules, size=n_pins, replace=False)
            nets.append(sorted(modules.tolist()))
        return nets

    def build_adjacency(self) -> np.ndarray:
        """Build adjacency matrix from nets (module-module co-occurrence in nets)."""
        if self._nodes is None:
            raise ValueError("Must call generate() first.")
        N = self.config.num_modules
        adj = np.zeros((N, N), dtype=np.float32)
        for net in self._nets:
            for i in range(len(net)):
                for j in range(i + 1, len(net)):
                    u, v = net[i], net[j]
                    adj[u, v] += 1
                    adj[v, u] += 1
        # Normalize
        adj = np.minimum(adj, 1.0)  # binary adjacency
        return adj

    def build_edge_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build PyTorch Geometric-style edge_index for bipartite graph
        (modules <-> nets) with edge features.

        Returns:
            edge_index: (2, E) array of module-net edges
            edge_attr: (E, 1) array of edge features (pin count weight)
        """
        if self._nets is None:
            raise ValueError("Must call generate() first.")

        sources = []
        targets = []
        weights = []

        for net_id, net in enumerate(self._nets):
            net_node_id = self.config.num_modules + net_id  # offset for bipartite
            for module_id in net:
                sources.append(module_id)
                targets.append(net_node_id)
                weights.append(1.0 / len(net))  # inverse pin count as weight

                sources.append(net_node_id)
                targets.append(module_id)
                weights.append(1.0 / len(net))

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_attr = np.array(weights, dtype=np.float32).reshape(-1, 1)
        return edge_index, edge_attr

    def build_module_edge_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Module↔module edges from net co-occurrence (for DenoiserNet / module-only GNN).

        Returns:
            edge_index: (2, E) with indices in [0, num_modules)
            edge_attr: (E, 1) weights (1 / pin_count per net)
        """
        if self._nets is None:
            raise ValueError("Must call generate() first.")

        sources = []
        targets = []
        weights = []

        for net in self._nets:
            if len(net) < 2:
                continue
            w = 1.0 / len(net)
            for i in range(len(net)):
                for j in range(i + 1, len(net)):
                    u, v = net[i], net[j]
                    sources.extend([u, v])
                    targets.extend([v, u])
                    weights.extend([w, w])

        if not sources:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 1), dtype=np.float32)
        else:
            edge_index = np.array([sources, targets], dtype=np.int64)
            edge_attr = np.array(weights, dtype=np.float32).reshape(-1, 1)
        return edge_index, edge_attr

    def save(self, output_dir: str, prefix: str = "netlist"):
        """Save generated netlist to CSV files."""
        if self._nodes is None:
            raise ValueError("Must call generate() first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save nodes
        nodes_df = pd.DataFrame(
            self._nodes,
            columns=["id", "width", "height"]
        )
        nodes_df.to_csv(out / f"{prefix}_nodes.csv", index=False)

        # Save nets
        nets_df = pd.DataFrame({
            "net_id": range(len(self._nets)),
            "modules": [",".join(map(str, net)) for net in self._nets]
        })
        nets_df.to_csv(out / f"{prefix}_nets.csv", index=False)

        print(f"Saved {len(self._nodes)} nodes, {len(self._nets)} nets "
              f"to {output_dir}/")

    def get_total_area(self) -> float:
        """Total area of all modules."""
        if self._nodes is None:
            raise ValueError("Must call generate() first.")
        return np.sum(self._nodes[:, 1] * self._nodes[:, 2])

    @property
    def nodes(self) -> Optional[np.ndarray]:
        return self._nodes

    @property
    def nets(self) -> Optional[list]:
        return self._nets
