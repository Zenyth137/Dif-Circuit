"""
MDP-based Macro Placement Environment.

Models placement as a sequential decision process:
- State: grid density + module features + GNN topology embeddings
- Action: place current module at a grid position (x, y)
- Reward: terminal HPWL + congestion (delayed reward)

References:
  - AlphaChip (Mirhoseini et al., Nature 2021)
  - VLSI.tex Section 3.1
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the MDP placement environment."""
    grid_size: int = 64            # Grid resolution (grid_size x grid_size)
    canvas_width: float = 1000.0
    canvas_height: float = 1000.0
    max_modules: int = 100         # Max modules to place per episode
    # Reward weights
    w_hpwl: float = 1.0
    w_congestion: float = 0.5
    w_overlap: float = 0.0
    # Congestion grid
    congestion_bins: int = 16

    def __post_init__(self):
        self.cell_w = self.canvas_width / self.grid_size
        self.cell_h = self.canvas_height / self.grid_size


class MacroPlacementEnv:
    """
    Sequential macro placement environment.

    At each step t, the agent selects a grid position (gx, gy)
    to place module t. State includes:
      - density_grid: (grid_size, grid_size) current placement density
      - module_features: features of the current module (w, h)
      - topology_embedding: optional GNN embedding of the netlist
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.reset()

    def reset(self,
              nodes: Optional[np.ndarray] = None,
              nets: Optional[list] = None,
              topology_embedding: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment with a new netlist.

        Args:
            nodes: (N, 3) array [id, width, height]
            nets: list of lists, each inner list = module indices in that net
            topology_embedding: (N, D) pre-computed GNN embeddings per module

        Returns:
            state: dictionary with environment state
        """
        self.nodes = nodes
        self.nets = nets
        self.topology_embedding = topology_embedding

        num_modules = len(nodes) if nodes is not None else self.cfg.max_modules
        self.num_modules = min(num_modules, self.cfg.max_modules)

        # Initialize placement state
        self.placed_positions = {}        # module_id -> (cx, cy)
        self.placed_modules = set()
        self.current_step = 0

        # Density grid (continuous-valued)
        self.density_grid = np.zeros(
            (self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32
        )

        # Congestion tracking (for terminal reward)
        self._route_demand = np.zeros(
            (self.cfg.congestion_bins, self.cfg.congestion_bins),
            dtype=np.float32
        )

        self._done = False
        return self._get_state()

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one placement step.

        Args:
            action: (gx, gy) grid position indices

        Returns:
            state, reward, done, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        gx, gy = action
        module_id = self.current_step

        # Convert grid to continuous coordinates (center of cell)
        cx = (gx + 0.5) * self.cfg.cell_w
        cy = (gy + 0.5) * self.cfg.cell_h

        # Record placement
        self.placed_positions[module_id] = (cx, cy)
        self.placed_modules.add(module_id)

        # Update density grid
        self._update_density(module_id, gx, gy)

        # Advance
        self.current_step += 1

        # Check termination
        done = (self.current_step >= self.num_modules)

        if done:
            reward = self._compute_terminal_reward()
            self._done = True
        else:
            reward = 0.0  # delayed reward

        state = self._get_state()
        info = {
            "step": self.current_step,
            "module_id": module_id,
            "position": (cx, cy),
        }

        return state, reward, done, info

    def _get_state(self) -> Dict[str, np.ndarray]:
        """Build the current state dictionary."""
        state = {
            "density_grid": self.density_grid.copy(),
        }

        if self.nodes is not None and self.current_step < self.num_modules:
            next_module = self.nodes[self.current_step]
            state["module_features"] = next_module[1:].astype(np.float32)  # [w, h]

        if self.topology_embedding is not None:
            state["topology_embedding"] = self.topology_embedding

        # Mask of available grid positions (simplified: all positions available)
        state["action_mask"] = np.ones(
            (self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32
        )

        return state

    def _update_density(self, module_id: int, gx: int, gy: int):
        """Update density grid for placed module."""
        if self.nodes is None:
            return
        w = self.nodes[module_id][1]
        h = self.nodes[module_id][2]

        # Calculate how many grid cells this module covers
        gw = max(1, int(np.ceil(w / self.cfg.cell_w)))
        gh = max(1, int(np.ceil(h / self.cfg.cell_h)))

        # Mark occupied cells
        gx_start = max(0, gx - gw // 2)
        gy_start = max(0, gy - gh // 2)
        gx_end = min(self.cfg.grid_size, gx_start + gw)
        gy_end = min(self.cfg.grid_size, gy_start + gh)

        self.density_grid[gx_start:gx_end, gy_start:gy_end] += 1.0

    def _compute_terminal_reward(self) -> float:
        """Compute terminal reward: negative weighted sum of HPWL and congestion."""
        from .reward import compute_hpwl, compute_congestion, compute_overlap_union

        if self.nodes is None or self.nets is None:
            return 0.0

        positions = np.array([
            [self.placed_positions[i][0], self.placed_positions[i][1]]
            for i in range(self.num_modules)
        ])

        hpwl = compute_hpwl(positions, self.nets)
        congestion = compute_congestion(
            positions, self.nodes,
            bins=self.cfg.congestion_bins,
            canvas_width=self.cfg.canvas_width,
            canvas_height=self.cfg.canvas_height,
            nets=self.nets,
        )
        overlap = compute_overlap_union(
            positions, self.nodes,
            canvas_width=self.cfg.canvas_width,
            canvas_height=self.cfg.canvas_height,
        )

        reward = -(
            self.cfg.w_hpwl * hpwl
            + self.cfg.w_congestion * congestion
            + self.cfg.w_overlap * overlap
        )
        return float(reward)

    def get_placement(self) -> np.ndarray:
        """Get final placement as (N, 2) array of (cx, cy) coordinates."""
        positions = np.zeros((self.num_modules, 2), dtype=np.float32)
        for i in range(self.num_modules):
            if i in self.placed_positions:
                positions[i] = self.placed_positions[i]
        return positions

    @property
    def action_space_size(self) -> int:
        return self.cfg.grid_size * self.cfg.grid_size

    @property
    def grid_size(self) -> int:
        return self.cfg.grid_size
