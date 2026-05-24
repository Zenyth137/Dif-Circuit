"""
Reward functions for VLSI placement.

- HPWL: Half-Perimeter Wire Length (bounding box of each net)
- Congestion: routing congestion estimated from grid-bin wire density
- Overlap: total overlapping area between modules
"""

import numpy as np
from typing import Optional


def compute_hpwl(positions: np.ndarray, nets: list) -> float:
    """
    Compute Half-Perimeter Wire Length.

    For each net, computes the bounding box of all connected modules
    and sums the half-perimeter (width + height).

    Args:
        positions: (N, 2) array of (cx, cy) module center positions
        nets: list of lists, each inner list = module indices in that net

    Returns:
        total_hpwl: sum of bounding-box half-perimeters
    """
    total = 0.0
    for net in nets:
        if len(net) == 0:
            continue
        net_positions = positions[net]
        x_min, y_min = net_positions.min(axis=0)
        x_max, y_max = net_positions.max(axis=0)
        total += (x_max - x_min) + (y_max - y_min)
    return total


def compute_congestion(
    positions: np.ndarray,
    nodes: np.ndarray,
    bins: int = 16,
    canvas_width: float = 1000.0,
    canvas_height: float = 1000.0,
    nets: Optional[list] = None,
) -> float:
    """
    Estimate routing congestion by counting modules per grid bin.

    Simplified congestion metric: standard deviation of bin densities.
    Higher std → more uneven distribution → higher congestion.

    Args:
        positions: (N, 2) array of module centers
        nodes: (N, 3) array [id, width, height]
        bins: number of bins per dimension
        canvas_width, canvas_height: canvas dimensions
        nets: optional nets for wire-based congestion

    Returns:
        congestion score (higher = worse)
    """
    N = len(positions)
    if N == 0:
        return 0.0

    bin_w = canvas_width / bins
    bin_h = canvas_height / bins

    density = np.zeros((bins, bins), dtype=np.float32)

    for i in range(N):
        cx, cy = positions[i]
        w, h = nodes[i, 1], nodes[i, 2]

        # Which bins does this module overlap?
        bx_start = max(0, int((cx - w / 2) / bin_w))
        by_start = max(0, int((cy - h / 2) / bin_h))
        bx_end = min(bins - 1, int((cx + w / 2) / bin_w))
        by_end = min(bins - 1, int((cy + h / 2) / bin_h))

        density[bx_start:bx_end + 1, by_start:by_end + 1] += 1.0

    # Congestion = std of bin densities (higher variance = worse congestion)
    congestion = float(np.std(density))
    return congestion


def compute_overlap(
    positions: np.ndarray,
    nodes: np.ndarray,
) -> float:
    """
    Compute total overlapping area between all module pairs.

    Args:
        positions: (N, 2) array of (cx, cy) module center positions
        nodes: (N, 3) array [id, width, height]

    Returns:
        total_overlap_area: sum of pairwise intersection areas
    """
    N = len(positions)
    total_overlap = 0.0

    for i in range(N):
        xi, yi = positions[i]
        wi, hi = nodes[i, 1], nodes[i, 2]
        left_i = xi - wi / 2
        right_i = xi + wi / 2
        bottom_i = yi - hi / 2
        top_i = yi + hi / 2

        for j in range(i + 1, N):
            xj, yj = positions[j]
            wj, hj = nodes[j, 1], nodes[j, 2]
            left_j = xj - wj / 2
            right_j = xj + wj / 2
            bottom_j = yj - hj / 2
            top_j = yj + hj / 2

            # Overlap in x
            dx = max(0.0, min(right_i, right_j) - max(left_i, left_j))
            # Overlap in y
            dy = max(0.0, min(top_i, top_j) - max(bottom_i, bottom_j))

            total_overlap += dx * dy

    return total_overlap


def compute_overlap_union(
    positions: np.ndarray,
    nodes: np.ndarray,
    canvas_width: float = 1000.0,
    canvas_height: float = 1000.0,
    bins: int = 256,
) -> float:
    """
    Geometric overlap area via rasterization (each pixel over-count is physical).

    For every grid cell, if ``coverage`` modules cover it, count
    ``(coverage - 1) * cell_area``. Summed over cells this is at most
    ``sum(module_areas)``, so overlap_pct = overlap / total_area is in [0, 100].
    """
    N = len(positions)
    if N == 0:
        return 0.0

    grid = np.zeros((bins, bins), dtype=np.float32)
    cell_w = canvas_width / bins
    cell_h = canvas_height / bins
    cell_area = cell_w * cell_h

    for i in range(N):
        cx, cy = positions[i]
        w, h = nodes[i, 1], nodes[i, 2]
        left = cx - w / 2
        right = cx + w / 2
        bottom = cy - h / 2
        top = cy + h / 2

        bx0 = max(0, int(np.floor(left / cell_w)))
        bx1 = min(bins - 1, int(np.floor((right - 1e-9) / cell_w)))
        by0 = max(0, int(np.floor(bottom / cell_h)))
        by1 = min(bins - 1, int(np.floor((top - 1e-9) / cell_h)))

        if bx0 <= bx1 and by0 <= by1:
            grid[by0:by1 + 1, bx0:bx1 + 1] += 1.0

    excess = np.maximum(grid - 1.0, 0.0).sum() * cell_area
    return float(excess)
