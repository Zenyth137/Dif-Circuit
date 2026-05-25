"""
Simulated Annealing (SA) placement baseline using B*-tree representation.

B*-tree is a compact binary tree representation for non-slicing floorplans.
Each node in the tree represents a module; the tree encodes relative positions.

SA perturbs the B*-tree via:
  - Node swap: exchange two modules
  - Rotation: rotate a module 90°
  - Move: move a node within the tree

Cost function: HPWL + overlap penalty.

Reference: VLSI.tex — 传统算法的智能引导 (Section 2.3)
"""

import numpy as np
import random
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
import time


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing."""
    # Cooling schedule
    T_initial: float = 1000.0
    T_final: float = 0.01
    cooling_rate: float = 0.95        # geometric: T_k = α^k · T_0
    # Perturbation probabilities
    p_swap: float = 0.4
    p_rotate: float = 0.3
    p_move: float = 0.3
    # Cost weights
    w_hpwl: float = 1.0
    w_overlap: float = 100.0
    # Termination
    max_iterations: int = 10000
    max_no_improve: int = 1000
    # Random seed
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


class BStarTree:
    """
    B*-tree representation for non-slicing floorplan.

    Each node has:
      - module_id: index into the module array
      - left, right: child pointers
      - parent: parent pointer
      - rotated: whether the module is rotated 90°
    """

    class Node:
        __slots__ = ('module_id', 'left', 'right', 'parent', 'rotated')

        def __init__(self, module_id: int):
            self.module_id = module_id
            self.left: Optional['BStarTree.Node'] = None
            self.right: Optional['BStarTree.Node'] = None
            self.parent: Optional['BStarTree.Node'] = None
            self.rotated: bool = False

    def __init__(self, num_modules: int):
        self.num_modules = num_modules
        self.root: Optional[BStarTree.Node] = None
        self.node_map: dict = {}  # module_id -> Node

    def build_random(self) -> 'BStarTree':
        """Build a random B*-tree by inserting modules sequentially."""
        module_ids = list(range(self.num_modules))
        random.shuffle(module_ids)

        self.root = self.Node(module_ids[0])
        self.node_map[module_ids[0]] = self.root

        for mid in module_ids[1:]:
            self._insert_random(mid)

        return self

    def _insert_random(self, module_id: int):
        """Insert a new module at a random free slot in the tree."""
        node = self.Node(module_id)

        # Find nodes with at least one free child slot (EXCLUDING the new node)
        candidates = [n for n in self.node_map.values()
                      if n.left is None or n.right is None]
        if not candidates:
            candidates = list(self.node_map.values())
        parent = random.choice(candidates)

        # Attach to free slot
        if parent.left is None:
            parent.left = node
        elif parent.right is None:
            parent.right = node
        else:
            parent.left = node  # fallback (shouldn't happen)
        node.parent = parent

        # Add to map AFTER attaching (so node can't be its own parent)
        self.node_map[module_id] = node

    def to_positions(self,
                     modules: np.ndarray,
                     canvas_width: float = 1000.0,
                     canvas_height: float = 1000.0) -> np.ndarray:
        """
        Convert B*-tree to module positions via contour-based packing.

        modules: (N, 3) array [id, width, height]

        Returns:
            positions: (N, 2) array of (x, y) bottom-left coordinates
        """
        N = self.num_modules
        positions = np.zeros((N, 2), dtype=np.float32)

        if self.root is None:
            return positions

        # In-order traversal with contour tracking
        contour = [(0.0, canvas_width, 0.0)]  # (x_start, x_end, y)

        def place_node(node: 'BStarTree.Node', x_offset: float):
            if node is None:
                return

            mid = node.module_id
            w = modules[mid, 1]
            h = modules[mid, 2]
            if node.rotated:
                w, h = h, w

            # Find lowest y at which module fits: max contour height over
            # all contour segments that overlap with the module's x-range.
            place_y = 0.0
            for (xs, xe, yc) in contour:
                if x_offset + w > xs and x_offset < xe:
                    place_y = max(place_y, yc)

            # Place module
            positions[mid] = [x_offset, place_y]

            # Update contour: replace ALL overlapping segments with a single raised
            # segment, preserving non-overlapping left/right edges of each segment.
            new_contour = []
            inserted = False
            for (xs, xe, yc) in contour:
                if xe <= x_offset or xs >= x_offset + w:
                    # Non-overlapping — keep as-is
                    new_contour.append((xs, xe, yc))
                elif not inserted:
                    # First overlapping segment: emit raised segment
                    if xs < x_offset:
                        new_contour.append((xs, x_offset, yc))
                    new_contour.append((x_offset, x_offset + w, place_y + h))
                    if xe > x_offset + w:
                        new_contour.append((x_offset + w, xe, yc))
                    inserted = True
                else:
                    # Subsequent overlapping segment: only keep the right
                    # part that extends beyond the module (if any)
                    if xe > x_offset + w:
                        new_contour.append((x_offset + w, xe, yc))
            if not inserted:
                new_contour.append((x_offset, x_offset + w, place_y + h))

            # Merge adjacent segments at same height
            contour[:] = new_contour

            # Place left child
            if node.left:
                place_node(node.left, x_offset + w)

            # Place right child
            if node.right:
                place_node(node.right, x_offset)

        place_node(self.root, 0.0)
        return positions

    def perturb_swap(self):
        """Swap two random modules."""
        keys = list(self.node_map.keys())
        if len(keys) < 2:
            return
        a, b = random.sample(keys, 2)
        node_a, node_b = self.node_map[a], self.node_map[b]
        node_a.module_id, node_b.module_id = b, a
        self.node_map[a], self.node_map[b] = node_b, node_a

    def perturb_rotate(self):
        """Rotate a random module."""
        keys = list(self.node_map.keys())
        if not keys:
            return
        mid = random.choice(keys)
        self.node_map[mid].rotated = not self.node_map[mid].rotated

    def perturb_move(self):
        """Move a node within the tree (detach and re-insert at a free slot)."""
        keys = list(self.node_map.keys())
        if len(keys) < 2:
            return
        mid = random.choice(keys)
        node = self.node_map[mid]

        # Don't move root (simplification)
        if node == self.root:
            return

        # Detach
        parent = node.parent
        if parent is None:
            return
        if parent.left == node:
            parent.left = None
        else:
            parent.right = None

        # Find a target with at least one free child slot
        targets = [n for n in self.node_map.values()
                   if n != node and (n.left is None or n.right is None)]
        if not targets:
            # No free slot — re-attach to original parent
            if parent.left is None:
                parent.left = node
            elif parent.right is None:
                parent.right = node
            else:
                # Original parent is full too, just put as left
                parent.left = node
            node.parent = parent
            return

        target = random.choice(targets)
        if target.left is None:
            target.left = node
        else:
            target.right = node
        node.parent = target

    def copy(self) -> 'BStarTree':
        """Deep copy the tree."""
        new_tree = BStarTree(self.num_modules)
        if self.root is None:
            return new_tree

        def copy_node(node: 'BStarTree.Node',
                      parent: Optional['BStarTree.Node'] = None):
            if node is None:
                return None
            new_node = BStarTree.Node(node.module_id)
            new_node.rotated = node.rotated
            new_node.parent = parent
            new_tree.node_map[node.module_id] = new_node
            new_node.left = copy_node(node.left, new_node)
            new_node.right = copy_node(node.right, new_node)
            return new_node

        new_tree.root = copy_node(self.root)
        return new_tree


class SimulatedAnnealing:
    """
    Simulated Annealing placement using B*-tree representation.

    Iteratively perturbs the B*-tree, evaluates cost (HPWL + overlap),
    and accepts/rejects based on the Metropolis criterion.
    """

    def __init__(self, config: SAConfig):
        self.cfg = config
        self.tree: Optional[BStarTree] = None
        self.best_tree: Optional[BStarTree] = None
        self.best_cost: float = float('inf')
        self.best_positions: Optional[np.ndarray] = None

    def place(self,
              nodes: np.ndarray,
              nets: list,
              canvas_width: float = 1000.0,
              canvas_height: float = 1000.0,
              verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Run Simulated Annealing placement.

        Args:
            nodes: (N, 3) array [id, width, height]
            nets: list of lists of module indices
            canvas_width, canvas_height: canvas dimensions

        Returns:
            positions: (N, 2) module center positions
            stats: dict with cost history, runtime, etc.
        """
        N = len(nodes)
        self.tree = BStarTree(N)
        self.tree.build_random()

        self.best_tree = self.tree.copy()
        self.best_cost = float('inf')
        self.best_positions = None

        T = self.cfg.T_initial
        no_improve = 0
        costs = []

        start_time = time.time()

        for iteration in range(self.cfg.max_iterations):
            # Save current state
            old_tree = self.tree.copy()
            old_positions = self.tree.to_positions(nodes, canvas_width, canvas_height)
            old_cost = self._cost(old_positions, nodes, nets)

            # Perturb
            r = random.random()
            if r < self.cfg.p_swap:
                self.tree.perturb_swap()
            elif r < self.cfg.p_swap + self.cfg.p_rotate:
                self.tree.perturb_rotate()
            else:
                self.tree.perturb_move()

            # Evaluate new state
            new_positions = self.tree.to_positions(nodes, canvas_width, canvas_height)
            new_cost = self._cost(new_positions, nodes, nets)

            # Metropolis criterion
            delta = new_cost - old_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                costs.append(new_cost)
                if new_cost < self.best_cost:
                    self.best_cost = new_cost
                    self.best_tree = self.tree.copy()
                    self.best_positions = new_positions.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reject: revert
                self.tree = old_tree
                costs.append(old_cost)
                no_improve += 1

            # Cool down
            T *= self.cfg.cooling_rate

            # Check termination
            if T < self.cfg.T_final:
                break
            if no_improve >= self.cfg.max_no_improve:
                break

        elapsed = time.time() - start_time

        # Use best found
        if self.best_positions is None:
            self.best_positions = self.tree.to_positions(
                nodes, canvas_width, canvas_height
            )
            self.best_cost = self._cost(self.best_positions, nodes, nets)

        # Convert bottom-left corners to centers
        centers = self._corners_to_centers(self.best_positions, nodes)

        stats = {
            "best_cost": self.best_cost,
            "iterations": iteration + 1,
            "runtime": elapsed,
            "cost_history": costs,
        }

        if verbose:
            print(f"SA: cost={self.best_cost:.1f}, "
                  f"iters={iteration+1}, time={elapsed:.1f}s")

        return centers, stats

    def _cost(self, positions: np.ndarray,
              nodes: np.ndarray, nets: list) -> float:
        """Compute total cost = HPWL + overlap penalty."""
        from ..environment.reward import compute_hpwl, compute_overlap_union
        centers = self._corners_to_centers(positions, nodes)
        hpwl = compute_hpwl(centers, nets)
        overlap = compute_overlap_union(centers, nodes)
        return self.cfg.w_hpwl * hpwl + self.cfg.w_overlap * overlap

    def _corners_to_centers(self, corners: np.ndarray,
                            nodes: np.ndarray) -> np.ndarray:
        """Convert bottom-left corner coordinates to center coordinates."""
        centers = np.zeros_like(corners)
        for i in range(len(nodes)):
            w = nodes[i, 1]
            h = nodes[i, 2]
            # Check if rotated in tree
            if (self.best_tree and i in self.best_tree.node_map and
                    self.best_tree.node_map[i].rotated):
                w, h = h, w
            centers[i, 0] = corners[i, 0] + w / 2
            centers[i, 1] = corners[i, 1] + h / 2
        return centers
