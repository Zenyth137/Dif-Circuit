#!/usr/bin/env python3
"""
Generate SA expert trajectories for imitation learning.

For each netlist, runs SA to get a high-quality placement, then converts
it into a sequential trajectory (module order + target grid positions)
that the MDP policy can learn to imitate.

Usage:
    python scripts/generate_sa_trajectories.py \
        --num-trajectories 2000 \
        --output data/sa_trajectories/
"""

import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

from src.baselines.simulated_annealing import SimulatedAnnealing, SAConfig
from src.netlist.generator import NetlistGenerator, NetlistConfig
from src.environment.mdp_env import EnvConfig


def compute_connectivity(nets: list, num_modules: int) -> np.ndarray:
    """Compute connectivity score per module: total pins across all its nets."""
    scores = np.zeros(num_modules, dtype=np.float32)
    for net in nets:
        for m in net:
            scores[m] += len(net)
    return scores


def sa_position_to_grid(cx: float, cy: float,
                        grid_size: int,
                        cell_w: float, cell_h: float) -> tuple:
    """Convert SA center position to grid cell indices."""
    gx = int(np.clip(cx / cell_w, 0, grid_size - 1))
    gy = int(np.clip(cy / cell_h, 0, grid_size - 1))
    return gx, gy


def main():
    parser = argparse.ArgumentParser(description="Generate SA expert trajectories")
    parser.add_argument("--num-trajectories", type=int, default=2000)
    parser.add_argument("--output", type=str, default="data/sa_trajectories")
    parser.add_argument("--num-modules", type=int, default=200)
    parser.add_argument("--num-nets", type=int, default=800)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--canvas-width", type=float, default=1000.0)
    parser.add_argument("--canvas-height", type=float, default=1000.0)
    parser.add_argument("--sa-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Netlist generator
    netlist_cfg = NetlistConfig(
        num_modules=args.num_modules,
        num_nets=args.num_nets,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        seed=args.seed,
    )
    generator = NetlistGenerator(netlist_cfg)

    # B*-tree placer (direct contour packing — fast, guaranteed non-overlapping)
    from src.baselines.simulated_annealing import BStarTree

    # Grid conversion
    cell_w = args.canvas_width / args.grid_size
    cell_h = args.canvas_height / args.grid_size

    all_trajectories = []
    total_saved = 0

    for idx in tqdm(range(args.num_trajectories), desc="Generating SA trajectories"):
        # Generate netlist
        nodes, nets = generator.generate()
        nodes, nets = generator.truncate_to_max_modules(args.num_modules)

        # Pack with B*-tree (deterministic random tree + contour packing)
        try:
            tree = BStarTree(len(nodes))
            tree.build_random()
            bl_positions = tree.to_positions(nodes, args.canvas_width, args.canvas_height)
            # Convert bottom-left corners to centers
            sa_positions = np.zeros_like(bl_positions)
            for i in range(len(nodes)):
                w, h = nodes[i, 1], nodes[i, 2]
                sa_positions[i, 0] = bl_positions[i, 0] + w / 2
                sa_positions[i, 1] = bl_positions[i, 1] + h / 2
            sa_hpwl = 0.0  # will compute later
        except Exception as e:
            print(f"  Trajectory {idx}: B*-tree failed ({e}), skipping")
            continue

        # Determine module placement order: by connectivity score (descending)
        # Tie-break by area (larger first)
        connectivity = compute_connectivity(nets, len(nodes))
        areas = nodes[:, 1] * nodes[:, 2]
        order_score = connectivity * areas  # composite score
        module_order = np.argsort(-order_score)  # descending

        # Build sequential trajectory
        # For each step: (module_id, w, h, target_gx, target_gy)
        steps = []
        for rank, mid in enumerate(module_order):
            cx, cy = sa_positions[mid]
            w, h = float(nodes[mid, 1]), float(nodes[mid, 2])
            gx, gy = sa_position_to_grid(cx, cy, args.grid_size, cell_w, cell_h)
            action_idx = gx * args.grid_size + gy
            steps.append({
                "module_id": int(mid),
                "w": w,
                "h": h,
                "gx": gx,
                "gy": gy,
                "action_idx": action_idx,
            })

        trajectory = {
            "nodes": nodes.astype(np.float32),
            "nets": nets,
            "module_order": module_order.astype(np.int32),
            "sa_positions": sa_positions.astype(np.float32),
            "sa_hpwl": float(sa_hpwl),
            "steps": steps,
        }
        all_trajectories.append(trajectory)
        total_saved += 1

        # Save in chunks of 500 to limit memory
        if len(all_trajectories) >= 500:
            chunk_path = os.path.join(
                args.output, f"trajectories_{total_saved - len(all_trajectories):05d}.pkl"
            )
            with open(chunk_path, 'wb') as f:
                pickle.dump(all_trajectories, f)
            all_trajectories = []

    # Save remaining
    if all_trajectories:
        chunk_path = os.path.join(
            args.output, f"trajectories_{total_saved - len(all_trajectories):05d}.pkl"
        )
        with open(chunk_path, 'wb') as f:
            pickle.dump(all_trajectories, f)

    print(f"\nSaved {total_saved} trajectories to {args.output}/")


if __name__ == "__main__":
    main()
