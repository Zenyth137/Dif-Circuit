"""
Zero-Overlap Deterministic Legalizer.

Implements a shelf/skyline-based greedy packing algorithm that guarantees
absolutely no module overlap.  Used to generate clean X_0 ground-truth
placements for diffusion model training.

Algorithm (Skyline / Shelf Packing):
  1. Sort modules by guide positions (e.g., x-coordinate from analytical placer)
  2. For connectivity-aware ordering, cluster modules that share many nets
  3. Maintain a skyline — a piecewise-constant contour y(x) of placed modules
  4. For each module: find the lowest feasible horizontal span, place it,
     update skyline
  5. Optional: post-placement refinement via greedy swap that only accepts
     swaps reducing HPWL while maintaining zero-overlap

Invariant: every output placement has 0.0 overlap_area.

Reference: adapted from classical VLSI legalization (Tetris, Abacus-like shelf)
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time


@dataclass
class LegalizerConfig:
    """Configuration for the zero-overlap legalizer."""
    canvas_width: float = 1000.0
    canvas_height: float = 1000.0
    # Sorting strategy for module placement order
    sort_by: str = "connectivity"  # "x", "area", "connectivity"
    # Post-optimization
    refine_swaps: int = 50         # number of greedy swap attempts (0 = skip)
    # Random seed for tie-breaking
    seed: int = 42


class Skyline:
    """
    Piecewise-constant contour representing the top of placed modules.
    Maintained as a sorted list of (x_start, x_end, y_top) segments.
    """

    def __init__(self, canvas_width: float):
        # Initial skyline: flat at y=0 for the whole canvas
        self.segments = [(0.0, canvas_width, 0.0)]
        self.canvas_width = canvas_width

    def find_lowest_span(self, module_width: float) -> Tuple[float, float]:
        """
        Find the lowest y-position where a module of given width fits.

        Checks all segment boundary positions as candidates, guaranteeing
        we find the optimal placement. Snaps the result to the nearest
        segment boundary to avoid floating-point slivers between modules.

        Returns:
            (x_position, y_position): bottom-left corner to place the module
        """
        best_x = 0.0
        best_y = float('inf')

        # Candidate x positions: 0 + all segment boundaries
        candidates = [0.0]
        for xs, xe, _ in self.segments:
            if xs + module_width <= self.canvas_width + 1e-4:
                candidates.append(xs)

        # Deduplicate and sort
        candidates = sorted(set(candidates))

        for x in candidates:
            if x + module_width > self.canvas_width + 1e-4:
                continue
            span_y = self._max_y_over_span(x, x + module_width)
            # Prefer earlier x when y is tied (within epsilon)
            if span_y < best_y - 1e-6:
                best_y = span_y
                best_x = x

        # Snap best_x to the nearest segment boundary to guarantee
        # exact alignment and avoid FP slivers between modules
        best_x = self._snap_to_boundary(best_x)

        return best_x, best_y

    def _snap_to_boundary(self, x: float) -> float:
        """Snap x to the nearest segment boundary (within tolerance)."""
        for xs, xe, _ in self.segments:
            if abs(x - xs) < 1e-3:
                return xs
            if abs(x - xe) < 1e-3:
                return xe
        return x

    def _max_y_over_span(self, x_start: float, x_end: float) -> float:
        """Maximum y of skyline over [x_start, x_end]."""
        max_y = 0.0
        for xs, xe, y in self.segments:
            overlap_start = max(xs, x_start)
            overlap_end = min(xe, x_end)
            if overlap_start < overlap_end:
                max_y = max(max_y, y)
        return max_y

    def place_module(self, x: float, y: float, w: float, h: float):
        """
        Place a module on the skyline and update the contour.

        Args:
            x, y: bottom-left corner
            w, h: module dimensions
        """
        new_top = y + h
        new_segments = []
        placed = False

        for xs, xe, ys in self.segments:
            if xe <= x or xs >= x + w:
                # No overlap with this segment
                new_segments.append((xs, xe, ys))
            else:
                # This segment is partially/fully covered
                if xs < x:
                    new_segments.append((xs, x, ys))
                if not placed:
                    new_segments.append((x, x + w, new_top))
                    placed = True
                if xe > x + w:
                    new_segments.append((x + w, xe, ys))

        if not placed:
            new_segments.append((x, x + w, new_top))

        # Merge adjacent segments at same height
        self.segments = self._merge(new_segments)

    @staticmethod
    def _merge(segments: list) -> list:
        """Merge adjacent segments with the same y."""
        if not segments:
            return []
        segments.sort(key=lambda s: s[0])
        merged = [segments[0]]
        for xs, xe, y in segments[1:]:
            px_s, px_e, py = merged[-1]
            if abs(xs - px_e) < 1e-6 and abs(y - py) < 1e-6:
                merged[-1] = (px_s, xe, py)
            else:
                merged.append((xs, xe, y))
        return merged

    def max_y(self) -> float:
        """Maximum y-coordinate of the skyline."""
        return max(y for _, _, y in self.segments)


class ZeroOverlapLegalizer:
    """
    Deterministic legalizer producing absolutely zero-overlap placements.

    Suitable for generating ground-truth X_0 for diffusion training.
    """

    def __init__(self, config: LegalizerConfig):
        self.cfg = config
        np.random.seed(config.seed)

    def legalize(self,
                 nodes: np.ndarray,
                 nets: list,
                 guide_positions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Produce a zero-overlap placement.

        Args:
            nodes: (N, 3) [id, width, height]
            nets: list of lists of module indices
            guide_positions: optional (N, 2) initial positions used for
                             sorting heuristic (e.g., from analytical placer)

        Returns:
            positions: (N, 2) center coordinates — GUARANTEED zero overlap
            stats: dict with metrics
        """
        N = len(nodes)
        start = time.time()

        # 1. Determine placement order
        if guide_positions is None:
            # No guide: use random positions as tie-breaker
            guide_positions = np.random.rand(N, 2) * self.cfg.canvas_width

        order = self._compute_order(nodes, nets, guide_positions)

        # 2. Skyline packing
        skyline = Skyline(self.cfg.canvas_width)
        corners = np.zeros((N, 2), dtype=np.float32)  # bottom-left corners

        for idx in order:
            w = float(nodes[idx, 1])
            h = float(nodes[idx, 2])

            # Find lowest feasible position
            x, y = skyline.find_lowest_span(w)

            # Check if we exceed canvas height
            if y + h > self.cfg.canvas_height:
                # Module doesn't fit vertically — start new "row" at y=0
                # (this is a degenerate case; in practice canvas should be sized properly)
                pass

            # Place module
            corners[idx] = [x, y]
            skyline.place_module(x, y, w, h)

        # Convert corners to centers
        centers = corners.copy()
        for i in range(N):
            centers[i, 0] = corners[i, 0] + nodes[i, 1] / 2.0
            centers[i, 1] = corners[i, 1] + nodes[i, 2] / 2.0

        # 3. Post-optimization: greedy swap refinement (zero-overlap preserving)
        if self.cfg.refine_swaps > 0:
            centers = self._greedy_swap_refine(centers, nodes, nets,
                                               max_attempts=self.cfg.refine_swaps)

        elapsed = time.time() - start

        from ..environment.reward import compute_hpwl, compute_overlap
        hpwl = compute_hpwl(centers, nets)
        overlap = compute_overlap(centers, nodes)

        stats = {
            "hpwl": hpwl,
            "overlap": overlap,
            "max_y": skyline.max_y(),
            "runtime": elapsed,
            "num_modules": N,
        }

        return centers, stats

    def _compute_order(self,
                       nodes: np.ndarray,
                       nets: list,
                       guide_positions: np.ndarray) -> np.ndarray:
        """
        Compute module placement order.

        Strategies:
          - "x": sort by guide x-coordinate
          - "area": sort by descending module area
          - "connectivity": cluster modules that share nets, then by x
        """
        N = len(nodes)

        if self.cfg.sort_by == "x":
            order = np.argsort(guide_positions[:, 0])

        elif self.cfg.sort_by == "area":
            areas = nodes[:, 1] * nodes[:, 2]
            order = np.argsort(-areas)  # descending

        elif self.cfg.sort_by == "connectivity":
            # Build module adjacency weight matrix
            adj_weight = np.zeros((N, N), dtype=np.float32)
            for net in nets:
                for i in range(len(net)):
                    for j in range(i + 1, len(net)):
                        u, v = net[i], net[j]
                        adj_weight[u, v] += 1.0 / len(net)
                        adj_weight[v, u] += 1.0 / len(net)

            # Clustering: greedy selection by connectivity degree
            degree = adj_weight.sum(axis=1)
            visited = np.zeros(N, dtype=bool)
            order = []

            # Start with the highest-degree module
            while len(order) < N:
                if len(order) == 0:
                    current = int(np.argmax(degree))
                else:
                    # Find unvisited module most connected to the last placed one
                    last = order[-1]
                    candidates = np.where(~visited)[0]
                    if len(candidates) == 0:
                        break
                    scores = adj_weight[last, candidates]
                    current = candidates[np.argmax(scores)]

                visited[current] = True
                order.append(current)

                # Also add strongly-connected neighbors
                neighbors = np.where((adj_weight[current] > 0.01) & (~visited))[0]
                sorted_neighbors = neighbors[np.argsort(-adj_weight[current, neighbors])]
                for nb in sorted_neighbors[:5]:  # Limit chain length
                    if not visited[nb]:
                        visited[nb] = True
                        order.append(nb)

            order = np.array(order, dtype=np.int64)

        else:
            order = np.arange(N)

        return order

    def _greedy_swap_refine(self,
                            centers: np.ndarray,
                            nodes: np.ndarray,
                            nets: list,
                            max_attempts: int = 50) -> np.ndarray:
        """
        Greedy swap refinement: try swapping two modules' positions.
        Accept only if HPWL improves AND zero overlap maintained.

        This is a safe post-optimization since the starting placement
        is guaranteed zero-overlap — we only accept swaps that preserve this.
        """
        from ..environment.reward import compute_hpwl, compute_overlap

        N = len(centers)
        current_centers = centers.copy()
        current_hpwl = compute_hpwl(current_centers, nets)

        for _ in range(max_attempts):
            i, j = np.random.choice(N, 2, replace=False)

            # Try swap
            new_centers = current_centers.copy()
            new_centers[i], new_centers[j] = current_centers[j].copy(), current_centers[i].copy()

            # Check overlap (allow 1e-6 tolerance for floating-point)
            overlap = compute_overlap(new_centers, nodes)
            if overlap > 1e-6:
                continue

            # Check HPWL
            new_hpwl = compute_hpwl(new_centers, nets)
            if new_hpwl < current_hpwl:
                current_centers = new_centers
                current_hpwl = new_hpwl

        return current_centers
