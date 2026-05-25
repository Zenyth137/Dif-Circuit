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
    w_overlap: float = 500.0      # CRITICAL: heavy penalty for overlap
    # Congestion grid
    congestion_bins: int = 16
    # Action masking
    max_density_per_cell: float = 1.0  # Max allowed module count per grid cell
    use_action_masking: bool = True     # Enable action masking

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
        self._partial_hpwl = 0.0    # cumulative per-step HPWL for logging
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

        # ---- Per-step dense reward ----
        step_reward = self._compute_step_reward(module_id, cx, cy)
        self._partial_hpwl += step_reward  # track cumulative for logging

        # Check termination
        done = (self.current_step >= self.num_modules)

        if done:
            reward = self._compute_terminal_reward()
            self._done = True
        else:
            reward = step_reward  # dense per-step reward (HPWL + overlap)

        state = self._get_state()
        info = {
            "step": self.current_step,
            "module_id": module_id,
            "position": (cx, cy),
            "step_reward": step_reward,
        }

        return state, reward, done, info

    def _get_state(self) -> Dict[str, np.ndarray]:
        """Build the current state dictionary with action masking."""
        state = {
            "density_grid": self.density_grid.copy(),
        }

        if self.nodes is not None and self.current_step < self.num_modules:
            next_module = self.nodes[self.current_step]
            state["module_features"] = next_module[1:].astype(np.float32)  # [w, h]

        if self.topology_embedding is not None:
            state["topology_embedding"] = self.topology_embedding

        # ---- Action Masking: block grid cells that are over capacity ----
        if self.cfg.use_action_masking and self.nodes is not None:
            # Compute per-cell capacity needed for current module
            if self.current_step < self.num_modules:
                next_mod = self.nodes[self.current_step]
                mod_w, mod_h = next_mod[1], next_mod[2]
                gw = max(1, int(np.ceil(mod_w / self.cfg.cell_w)))
                gh = max(1, int(np.ceil(mod_h / self.cfg.cell_h)))

                # For each grid cell, check if placing here would exceed capacity
                mask = np.ones((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)

                # Module occupies gw×gh cells. Check all cells in that footprint.
                for gx in range(self.cfg.grid_size - gw + 1):
                    for gy in range(self.cfg.grid_size - gh + 1):
                        footprint = self.density_grid[gx:gx+gw, gy:gy+gh]
                        if np.any(footprint >= self.cfg.max_density_per_cell):
                            mask[gx, gy] = 0.0  # Block this position

                state["action_mask"] = mask
            else:
                state["action_mask"] = np.ones(
                    (self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32
                )
        else:
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

    def _compute_step_reward(self, module_id: int, cx: float, cy: float) -> float:
        """
        Per-step dense reward: incremental HPWL contribution + immediate overlap.

        For each net containing the newly placed module, if all other members are
        already placed, accumulate the net's HPWL. Also penalize overlap with
        previously placed modules.
        """
        from .reward import compute_hpwl

        if self.nodes is None or self.nets is None:
            return 0.0

        # ---- Overlap: check against all previously placed modules ----
        overlap_penalty = 0.0
        w_cur, h_cur = self.nodes[module_id][1], self.nodes[module_id][2]
        left_cur = cx - w_cur / 2
        right_cur = cx + w_cur / 2
        bottom_cur = cy - h_cur / 2
        top_cur = cy + h_cur / 2

        for other_id in self.placed_modules:
            if other_id == module_id:
                continue
            ox, oy = self.placed_positions[other_id]
            ow, oh = self.nodes[other_id][1], self.nodes[other_id][2]
            left_o = ox - ow / 2
            right_o = ox + ow / 2
            bottom_o = oy - oh / 2
            top_o = oy + oh / 2

            dx = max(0.0, min(right_cur, right_o) - max(left_cur, left_o))
            dy = max(0.0, min(top_cur, top_o) - max(bottom_cur, bottom_o))
            overlap_penalty += dx * dy

        # ---- HPWL: nets that are now "complete" ----
        hpwl_contrib = 0.0
        for net in self.nets:
            if module_id not in net:
                continue
            if not all(m in self.placed_modules for m in net):
                continue  # not all pins placed yet — wait
            # All members placed — compute this net's HPWL
            positions = np.array([
                [self.placed_positions[m][0], self.placed_positions[m][1]]
                for m in net
            ])
            x_min, y_min = positions.min(axis=0)
            x_max, y_max = positions.max(axis=0)
            hpwl_contrib += (x_max - x_min) + (y_max - y_min)

        reward = -(
            self.cfg.w_hpwl * hpwl_contrib
            + self.cfg.w_overlap * overlap_penalty
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
