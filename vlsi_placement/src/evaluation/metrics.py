"""
Evaluation metrics for VLSI placement quality.

Metrics:
  - HPWL (Half-Perimeter Wire Length)
  - Overlap area (absolute and percentage)
  - Congestion (routing density variance)
  - Runtime

All metrics compute from module positions and netlist data.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PlacementMetrics:
    """Container for placement evaluation metrics."""
    hpwl: float = 0.0
    overlap_area: float = 0.0
    overlap_pct: float = 0.0       # overlap / total_module_area * 100
    congestion: float = 0.0
    total_area: float = 0.0
    canvas_area: float = 0.0
    density: float = 0.0           # total_module_area / canvas_area
    runtime: float = 0.0
    num_modules: int = 0
    num_nets: int = 0

    def to_dict(self) -> dict:
        return {
            "hpwl": self.hpwl,
            "overlap_area": self.overlap_area,
            "overlap_pct": self.overlap_pct,
            "congestion": self.congestion,
            "density": self.density,
            "runtime": self.runtime,
            "num_modules": self.num_modules,
            "num_nets": self.num_nets,
        }

    def __repr__(self):
        return (f"Metrics(HPWL={self.hpwl:.0f}, "
                f"Overlap={self.overlap_pct:.2f}%, "
                f"Congestion={self.congestion:.4f}, "
                f"Time={self.runtime:.1f}s)")


def compute_all_metrics(
    positions: np.ndarray,
    nodes: np.ndarray,
    nets: list,
    canvas_width: float = 1000.0,
    canvas_height: float = 1000.0,
    runtime: float = 0.0,
) -> PlacementMetrics:
    """
    Compute all placement quality metrics.

    Args:
        positions: (N, 2) module center coordinates
        nodes: (N, 3) [id, width, height]
        nets: list of lists, module indices per net
        canvas_width, canvas_height: canvas dimensions
        runtime: wall-clock time for placement

    Returns:
        PlacementMetrics dataclass
    """
    from ..environment.reward import compute_hpwl, compute_congestion, compute_overlap

    hpwl = compute_hpwl(positions, nets)
    overlap = compute_overlap(positions, nodes)
    total_module_area = float(np.sum(nodes[:, 1] * nodes[:, 2]))
    canvas_area = canvas_width * canvas_height
    overlap_pct = (overlap / total_module_area * 100) if total_module_area > 0 else 0.0
    congestion = compute_congestion(positions, nodes,
                                     canvas_width=canvas_width,
                                     canvas_height=canvas_height,
                                     nets=nets)

    return PlacementMetrics(
        hpwl=hpwl,
        overlap_area=overlap,
        overlap_pct=overlap_pct,
        congestion=congestion,
        total_area=total_module_area,
        canvas_area=canvas_area,
        density=total_module_area / canvas_area if canvas_area > 0 else 0.0,
        runtime=runtime,
        num_modules=len(nodes),
        num_nets=len(nets),
    )


# Convenience aliases — import at module level for external use
from ..environment.reward import compute_hpwl as _compute_hpwl
from ..environment.reward import compute_overlap as _compute_overlap
from ..environment.reward import compute_congestion as _compute_congestion

HPWL = _compute_hpwl
overlap_area = _compute_overlap
congestion = _compute_congestion
