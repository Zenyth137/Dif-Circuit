#!/usr/bin/env python3
"""
Run baseline placement methods on test netlists.

Usage:
    python scripts/run_baselines.py --data-dir data/test_netlists --output results/baselines
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time

from src.baselines.simulated_annealing import SimulatedAnnealing, SAConfig
from src.baselines.analytical import AnalyticalPlacer
from src.netlist.parser import NetlistParser
from src.evaluation.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline placement methods")
    parser.add_argument("--data-dir", type=str, default="data/test_netlists")
    parser.add_argument("--num-netlists", type=int, default=20)
    parser.add_argument("--prefix", type=str, default="netlist")
    parser.add_argument("--output", type=str, default="results/baselines")
    parser.add_argument("--canvas-width", type=float, default=1000.0)
    parser.add_argument("--canvas-height", type=float, default=1000.0)
    parser.add_argument("--sa-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize methods
    sa_config = SAConfig(
        max_iterations=args.sa_iterations,
        seed=args.seed,
    )
    sa = SimulatedAnnealing(sa_config)
    analytical = AnalyticalPlacer(args.canvas_width, args.canvas_height)

    all_results = []

    for i in range(args.num_netlists):
        try:
            nodes, nets = NetlistParser.load(args.data_dir, f"{args.prefix}_{i:04d}")
        except FileNotFoundError:
            print(f"Skipping {args.prefix}_{i:04d} (not found)")
            continue

        print(f"\nNetlist {i}: {len(nodes)} modules, {len(nets)} nets")

        # Simulated Annealing
        print("  Running SA...", end=" ", flush=True)
        start = time.time()
        sa_positions, sa_stats = sa.place(
            nodes, nets, args.canvas_width, args.canvas_height, verbose=False
        )
        sa_metrics = compute_all_metrics(
            sa_positions, nodes, nets,
            args.canvas_width, args.canvas_height,
            runtime=sa_stats["runtime"],
        )
        print(f"HPWL={sa_metrics.hpwl:.0f}, Overlap={sa_metrics.overlap_pct:.2f}%")

        # Analytical
        print("  Running Analytical...", end=" ", flush=True)
        start = time.time()
        an_positions, an_stats = analytical.place(
            nodes, nets, verbose=False
        )
        an_metrics = compute_all_metrics(
            an_positions, nodes, nets,
            args.canvas_width, args.canvas_height,
            runtime=an_stats["runtime"],
        )
        print(f"HPWL={an_metrics.hpwl:.0f}, Overlap={an_metrics.overlap_pct:.2f}%")

        all_results.append({
            "netlist_idx": i,
            "num_modules": len(nodes),
            "num_nets": len(nets),
            "SA_HPWL": sa_metrics.hpwl,
            "SA_Overlap_pct": sa_metrics.overlap_pct,
            "SA_Congestion": sa_metrics.congestion,
            "SA_Runtime": sa_metrics.runtime,
            "Analytical_HPWL": an_metrics.hpwl,
            "Analytical_Overlap_pct": an_metrics.overlap_pct,
            "Analytical_Congestion": an_metrics.congestion,
            "Analytical_Runtime": an_metrics.runtime,
        })

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output, "baseline_results.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"SA      - HPWL: {df['SA_HPWL'].mean():.0f} ± {df['SA_HPWL'].std():.0f}, "
          f"Overlap: {df['SA_Overlap_pct'].mean():.2f}%, "
          f"Time: {df['SA_Runtime'].mean():.1f}s")
    print(f"Analytical - HPWL: {df['Analytical_HPWL'].mean():.0f} ± {df['Analytical_HPWL'].std():.0f}, "
          f"Overlap: {df['Analytical_Overlap_pct'].mean():.2f}%, "
          f"Time: {df['Analytical_Runtime'].mean():.1f}s")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
