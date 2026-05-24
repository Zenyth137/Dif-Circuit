"""
Analytical Placement baseline: Quadratic Placement + Tetris-like Legalization.

This is a simplified analytical approach:
  1. Solve quadratic placement (minimize squared net length)
  2. Legalize via greedy Tetris-like packing (simplified Abacus-style)

Reference: VLSI.tex Section 2.3 (traditional analytical methods)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
import time


class AnalyticalPlacer:
    """
    Simple analytical placer.

    1. Quadratic placement: minimizes Σ (x_i - x_j)² for all nets
       → solves Ax = b_x, Ay = b_y
    2. Greedy legalization: sort by x, pack modules left-to-right
    """

    def __init__(self,
                 canvas_width: float = 1000.0,
                 canvas_height: float = 1000.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def place(self,
              nodes: np.ndarray,
              nets: list,
              verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run analytical placement.

        Args:
            nodes: (N, 3) [id, width, height]
            nets: list of lists of module indices

        Returns:
            positions: (N, 2) legalized center positions
            stats: runtime info
        """
        start = time.time()

        # Step 1: Quadratic placement
        qp_positions = self._quadratic_placement(nodes, nets)

        # Step 2: Legalization
        legal_positions = self._legalize(qp_positions, nodes)

        elapsed = time.time() - start

        from ..environment.reward import compute_hpwl, compute_overlap
        hpwl = compute_hpwl(legal_positions, nets)
        overlap = compute_overlap(legal_positions, nodes)

        stats = {
            "hpwl": hpwl,
            "overlap": overlap,
            "runtime": elapsed,
        }

        if verbose:
            print(f"Analytical: hpwl={hpwl:.0f}, "
                  f"overlap={overlap:.0f}, time={elapsed:.1f}s")

        return legal_positions, stats

    def _quadratic_placement(self,
                             nodes: np.ndarray,
                             nets: list) -> np.ndarray:
        """
        Solve quadratic placement.

        Minimizes: Σ_{e∈nets} Σ_{i,j∈e} (x_i - x_j)² + (y_i - y_j)²

        Equivalent to solving: L·x = 0, L·y = 0
        where L is the graph Laplacian (with fixed pins to break symmetry).
        """
        N = len(nodes)

        # Build adjacency matrix (unweighted module-module)
        row, col, data = [], [], []
        for net in nets:
            for i in range(len(net)):
                for j in range(i + 1, len(net)):
                    u, v = net[i], net[j]
                    row.extend([u, v])
                    col.extend([v, u])
                    data.extend([1, 1])

        if len(data) == 0:
            return np.random.rand(N, 2) * min(self.canvas_width, self.canvas_height)

        A = csr_matrix((data, (row, col)), shape=(N, N))

        # Degree matrix
        deg = np.array(A.sum(axis=1)).flatten()

        # Laplacian
        L = csr_matrix((N, N))
        L.setdiag(deg)
        L = L - A

        # Fix some nodes to break translational symmetry
        # Fix the first few modules to anchor positions
        num_fixed = min(4, N)
        fixed_indices = list(range(num_fixed))

        # Remove fixed nodes from system
        free_indices = [i for i in range(N) if i not in fixed_indices]

        # Anchor positions for fixed nodes
        spacing = min(self.canvas_width, self.canvas_height) / (num_fixed + 1)
        fixed_positions = np.zeros((num_fixed, 2))
        for k, idx in enumerate(fixed_indices):
            fixed_positions[k] = [(k + 1) * spacing, self.canvas_height / 2]

        # Solve for free nodes
        L_free = L[free_indices][:, free_indices]
        L_fixed = L[free_indices][:, fixed_indices]

        # Solve L_free * x_free = -L_fixed * x_fixed
        x_fixed_val = fixed_positions[:, 0]
        y_fixed_val = fixed_positions[:, 1]

        b_x = -L_fixed.dot(x_fixed_val)
        b_y = -L_fixed.dot(y_fixed_val)

        x_free = spsolve(L_free, b_x)
        y_free = spsolve(L_free, b_y)

        # Assemble
        positions = np.zeros((N, 2))
        positions[fixed_indices] = fixed_positions
        positions[free_indices, 0] = x_free
        positions[free_indices, 1] = y_free

        # Center and scale to canvas
        positions = self._normalize_positions(positions, nodes)

        return positions

    def _normalize_positions(self,
                              positions: np.ndarray,
                              nodes: np.ndarray) -> np.ndarray:
        """Normalize positions to fit within canvas."""
        # Shift to positive quadrant
        min_xy = positions.min(axis=0)
        positions = positions - min_xy

        # Scale to fit canvas
        max_xy = positions.max(axis=0)
        if max_xy[0] > 0:
            positions[:, 0] *= (self.canvas_width * 0.9) / max_xy[0]
        if max_xy[1] > 0:
            positions[:, 1] *= (self.canvas_height * 0.9) / max_xy[1]

        # Add margin
        positions += np.array([self.canvas_width * 0.05, self.canvas_height * 0.05])

        return positions

    def _legalize(self,
                  positions: np.ndarray,
                  nodes: np.ndarray) -> np.ndarray:
        """
        Greedy Tetris-like legalization.

        Sort modules by x-coordinate, then pack left-to-right,
        shifting up when overlap is detected.
        """
        N = len(nodes)
        legal = positions.copy()

        # Sort modules by x
        order = np.argsort(legal[:, 0])

        placed = []  # list of (x_left, x_right, y_top, idx)

        for idx in order:
            cx, cy = legal[idx]
            w = nodes[idx, 1]
            h = nodes[idx, 2]

            left = cx - w / 2
            right = cx + w / 2
            bottom = cy - h / 2

            # Find overlap with placed modules
            max_y = 0.0
            for (pl, pr, pt, _) in placed:
                if left < pr and right > pl:
                    max_y = max(max_y, pt)

            # Adjust y
            new_bottom = max(bottom, max_y)
            legal[idx, 1] = new_bottom + h / 2
            legal[idx, 0] = left + w / 2

            placed.append((left, left + w, new_bottom + h, idx))

        return legal
