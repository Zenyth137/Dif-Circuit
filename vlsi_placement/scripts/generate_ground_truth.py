#!/usr/bin/env python3
"""
Generate (netlist, zero-overlap placement) pairs for diffusion model training.

Workflow:
  1. Generate synthetic netlists (2000-10000 modules)
  2. Run analytical placer to get a rough guide placement
  3. Run zero-overlap legalizer using the guide for module ordering
  4. Save both netlist and legal placement (X_0 ground truth)

This avoids the "pseudo-ground-truth" trap: every saved X_0 has
ABSOLUTELY ZERO overlap (guaranteed by the skyline legalizer).

Usage:
    # Match MDP/diffusion configs (200 modules for ~8 GB GPU)
    python scripts/generate_ground_truth.py --num-modules 200 --num-nets 800 \\
        --num-samples 100 --output data/ground_truth
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.netlist.generator import NetlistGenerator, NetlistConfig
from src.netlist.legalizer import ZeroOverlapLegalizer, LegalizerConfig
from src.baselines.analytical import AnalyticalPlacer
from src.environment.reward import compute_hpwl, compute_overlap


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth (netlist, X_0) pairs for diffusion training"
    )
    # Netlist params
    parser.add_argument("--num-modules", type=int, default=200,
                        help="Modules per netlist (use same as mdp_train / diffusion_train)")
    parser.add_argument("--num-nets", type=int, default=800)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--canvas-width", type=float, default=1000.0)
    parser.add_argument("--canvas-height", type=float, default=1000.0)
    parser.add_argument("--min-width", type=float, default=2.0)
    parser.add_argument("--max-width", type=float, default=50.0)
    parser.add_argument("--min-height", type=float, default=2.0)
    parser.add_argument("--max-height", type=float, default=50.0)
    parser.add_argument("--min-pins", type=int, default=2)
    parser.add_argument("--max-pins", type=int, default=10)
    # Legalizer params
    parser.add_argument("--sort-by", type=str, default="connectivity",
                        choices=["x", "area", "connectivity"])
    parser.add_argument("--refine-swaps", type=int, default=200)
    # Output
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/ground_truth")
    parser.add_argument("--prefix", type=str, default="gt")
    # Skip analytical guide (use random order)
    parser.add_argument("--no-guide", action="store_true",
                        help="Skip analytical placer guide; use random ordering")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize components
    netlist_cfg = NetlistConfig(
        num_modules=args.num_modules,
        num_nets=args.num_nets,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        min_pins_per_net=args.min_pins,
        max_pins_per_net=args.max_pins,
        seed=args.seed,
    )
    generator = NetlistGenerator(netlist_cfg)

    legalizer_cfg = LegalizerConfig(
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        sort_by=args.sort_by,
        refine_swaps=args.refine_swaps,
        seed=args.seed,
    )
    legalizer = ZeroOverlapLegalizer(legalizer_cfg)

    if not args.no_guide:
        analytical = AnalyticalPlacer(args.canvas_width, args.canvas_height)

    all_hpwls = []
    all_overlaps = []

    print(f"Generating {args.num_samples} ground-truth pairs...")
    print(f"  Modules: {args.num_modules}, Nets: {args.num_nets}")
    print(f"  Legalizer: sort_by={args.sort_by}, refine_swaps={args.refine_swaps}")
    print()

    for i in range(args.num_samples):
        # Unique seed per sample
        netlist_cfg.seed = args.seed + i
        generator.config = netlist_cfg
        nodes, nets = generator.generate()

        # Step 1: Get guide positions
        if args.no_guide:
            guide = None
        else:
            guide, _ = analytical.place(nodes, nets, verbose=False)

        # Step 2: Zero-overlap legalization
        centers, stats = legalizer.legalize(nodes, nets, guide)

        overlap = stats["overlap"]
        hpwl = stats["hpwl"]

        all_hpwls.append(hpwl)
        all_overlaps.append(overlap)

        # Verify zero overlap
        if overlap > 1e-6:
            print(f"  WARNING: Sample {i} has overlap={overlap:.6f}! "
                  f"(should be 0.0)")

        # Save netlist (CSV)
        generator.save(args.output, f"{args.prefix}_{i:04d}")

        # Save placement (NumPy .npy for efficiency)
        np.save(
            os.path.join(args.output, f"{args.prefix}_{i:04d}_placement.npy"),
            centers
        )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:4d}/{args.num_samples}] "
                  f"HPWL={hpwl:.0f}, Overlap={overlap:.6f}, "
                  f"MaxY={stats['max_y']:.0f}, Time={stats['runtime']:.1f}s")

    # Print summary
    print()
    print("=" * 60)
    print("GROUND TRUTH GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Samples:       {args.num_samples}")
    print(f"  HPWL (mean):   {np.mean(all_hpwls):.0f} ± {np.std(all_hpwls):.0f}")
    print(f"  Overlap (max): {np.max(np.abs(all_overlaps)):.6f}")
    print(f"  All zero-overlap: {np.all(np.array(all_overlaps) < 1e-6)}")
    print(f"  Saved to:      {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
