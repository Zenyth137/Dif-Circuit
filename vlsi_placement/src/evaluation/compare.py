"""
Comparison framework for evaluating multiple placement methods.

Runs all methods (MDP, MDP+Diffusion, SA, Analytical) on the same
test netlists and produces comparison tables and plots.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import time
import os
from pathlib import Path

from .metrics import compute_all_metrics, PlacementMetrics
from ..netlist.generator import NetlistGenerator, NetlistConfig


@dataclass
class MethodResult:
    """Result of running a single placement method on one netlist."""
    method_name: str
    metrics: PlacementMetrics
    positions: np.ndarray


class ComparisonRunner:
    """
    Run and compare multiple placement methods on test netlists.

    Usage:
        runner = ComparisonRunner(
            methods={
                "SA": sa_placer.place,
                "MDP": mdp_pipeline,
                "MDP+Diffusion": mdp_diff_pipeline,
            },
            test_netlists=[...],
        )
        results = runner.run()
        runner.print_summary(results)
    """

    def __init__(self,
                 methods: Dict[str, Callable],
                 test_netlists: List[Tuple[np.ndarray, list]],
                 canvas_width: float = 1000.0,
                 canvas_height: float = 1000.0,
                 output_dir: Optional[str] = None):
        """
        Args:
            methods: dict of method_name -> callable(nodes, nets) -> positions
            test_netlists: list of (nodes, nets) tuples
            canvas_width, canvas_height: canvas dimensions
            output_dir: directory to save results
        """
        self.methods = methods
        self.test_netlists = test_netlists
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> Dict[str, List[MethodResult]]:
        """
        Run all methods on all test netlists.

        Returns:
            results: method_name -> list of MethodResult per netlist
        """
        all_results = {name: [] for name in self.methods}

        for i, (nodes, nets) in enumerate(self.test_netlists):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Netlist {i+1}/{len(self.test_netlists)}: "
                      f"{len(nodes)} modules, {len(nets)} nets")
                print(f"{'='*60}")

            for method_name, method_fn in self.methods.items():
                if verbose:
                    print(f"  Running {method_name}...", end=" ", flush=True)

                try:
                    start = time.time()
                    positions = method_fn(nodes, nets)
                    elapsed = time.time() - start

                    metrics = compute_all_metrics(
                        positions, nodes, nets,
                        canvas_width=self.canvas_width,
                        canvas_height=self.canvas_height,
                        runtime=elapsed,
                    )

                    result = MethodResult(
                        method_name=method_name,
                        metrics=metrics,
                        positions=positions,
                    )
                    all_results[method_name].append(result)

                    if verbose:
                        print(f"done. HPWL={metrics.hpwl:.0f}, "
                              f"Overlap={metrics.overlap_pct:.2f}%, "
                              f"Time={elapsed:.1f}s")

                except Exception as e:
                    if verbose:
                        print(f"FAILED: {e}")
                    # Record empty metrics
                    all_results[method_name].append(MethodResult(
                        method_name=method_name,
                        metrics=PlacementMetrics(runtime=0.0),
                        positions=np.zeros((len(nodes), 2)),
                    ))

        # Save results
        if self.output_dir:
            self._save_results(all_results)

        return all_results

    def summarize(self,
                  results: Dict[str, List[MethodResult]]) -> pd.DataFrame:
        """
        Compute summary statistics across all test netlists.

        Returns:
            DataFrame with mean ± std for each metric per method
        """
        rows = []
        for method_name, method_results in results.items():
            valid_results = [r for r in method_results
                             if r.metrics.num_modules > 0]

            if not valid_results:
                continue

            hpwls = [r.metrics.hpwl for r in valid_results]
            overlaps = [r.metrics.overlap_pct for r in valid_results]
            congestions = [r.metrics.congestion for r in valid_results]
            runtimes = [r.metrics.runtime for r in valid_results]

            rows.append({
                "Method": method_name,
                "HPWL (mean)": f"{np.mean(hpwls):.0f}",
                "HPWL (std)": f"{np.std(hpwls):.0f}",
                "Overlap % (mean)": f"{np.mean(overlaps):.2f}",
                "Overlap % (std)": f"{np.std(overlaps):.2f}",
                "Congestion (mean)": f"{np.mean(congestions):.4f}",
                "Runtime (mean)": f"{np.mean(runtimes):.1f}s",
                "Runtime (std)": f"{np.std(runtimes):.1f}s",
            })

        df = pd.DataFrame(rows)
        return df

    def print_mdp_vs_diffusion(self, results: Dict[str, List[MethodResult]]):
        """Per-netlist and mean deltas: MDP+Diffusion relative to MDP."""
        base_name, target_name = "MDP", "MDP+Diffusion"
        if base_name not in results or target_name not in results:
            return

        base_list = results[base_name]
        target_list = results[target_name]
        n = min(len(base_list), len(target_list))

        hpwl_deltas = []
        overlap_deltas = []
        rows = []

        print("\n" + "=" * 80)
        print("MDP+Diffusion vs MDP (per netlist)")
        print("=" * 80)
        print(f"{'Netlist':>8} {'ΔHPWL%':>10} {'MDP Ov%':>10} {'Diff Ov%':>10} {'ΔOv pp':>10}")
        print("-" * 80)

        for i in range(n):
            b, t = base_list[i], target_list[i]
            if b.metrics.num_modules == 0 or t.metrics.num_modules == 0:
                continue
            if b.metrics.hpwl <= 0:
                continue

            d_hpwl = (b.metrics.hpwl - t.metrics.hpwl) / b.metrics.hpwl * 100.0
            d_ov = b.metrics.overlap_pct - t.metrics.overlap_pct
            hpwl_deltas.append(d_hpwl)
            overlap_deltas.append(d_ov)
            rows.append((i + 1, d_hpwl, b.metrics.overlap_pct, t.metrics.overlap_pct, d_ov))
            print(
                f"{i + 1:8d} {d_hpwl:+10.1f} {b.metrics.overlap_pct:10.2f} "
                f"{t.metrics.overlap_pct:10.2f} {d_ov:+10.2f}"
            )

        if hpwl_deltas:
            print("-" * 80)
            print(
                f"{'MEAN':>8} {np.mean(hpwl_deltas):+10.1f} "
                f"{'':>10} {'':>10} {np.mean(overlap_deltas):+10.2f}"
            )
            print(
                f"\n  ΔHPWL% > 0  → diffusion lowered wirelength vs MDP\n"
                f"  ΔOv pp > 0  → diffusion reduced overlap % vs MDP"
            )
        print("=" * 80)

        self._save_mdp_vs_diffusion(rows, hpwl_deltas, overlap_deltas)

    def _save_mdp_vs_diffusion(self, rows, hpwl_deltas, overlap_deltas):
        if not self.output_dir or not rows:
            return
        df = pd.DataFrame(
            rows,
            columns=["netlist_idx", "hpwl_improve_pct", "mdp_overlap_pct",
                     "diffusion_overlap_pct", "overlap_reduction_pp"],
        )
        df.to_csv(self.output_dir / "mdp_vs_diffusion.csv", index=False)
        if hpwl_deltas:
            summary = pd.DataFrame([{
                "hpwl_improve_pct_mean": np.mean(hpwl_deltas),
                "overlap_reduction_pp_mean": np.mean(overlap_deltas),
            }])
            summary.to_csv(self.output_dir / "mdp_vs_diffusion_summary.csv", index=False)

    def print_summary(self, results: Dict[str, List[MethodResult]]):
        """Print formatted comparison table."""
        df = self.summarize(results)
        print("\n" + "=" * 80)
        print("PLACEMENT METHOD COMPARISON")
        print("=" * 80)
        print("Overlap % = union overlap / total module area (0–100%, physical)")
        print(df.to_string(index=False))
        print("=" * 80)

        self.print_mdp_vs_diffusion(results)

        # Compute relative improvements vs first method in table
        if len(df) >= 2:
            print("\nRelative to first method (HPWL only):")
            baseline_name = df.iloc[0]["Method"]
            baseline_results = results[baseline_name]
            baseline_hpwls = [r.metrics.hpwl for r in baseline_results
                              if r.metrics.num_modules > 0]
            baseline_hpwl_mean = np.mean(baseline_hpwls) if baseline_hpwls else 1.0

            for method_name, method_results in results.items():
                if method_name == baseline_name:
                    continue
                valid = [r for r in method_results if r.metrics.num_modules > 0]
                if not valid:
                    continue
                hpwl_mean = np.mean([r.metrics.hpwl for r in valid])
                improvement = (baseline_hpwl_mean - hpwl_mean) / baseline_hpwl_mean * 100
                print(f"  {method_name} vs {baseline_name}: "
                      f"HPWL improvement = {improvement:+.1f}%")

    def _save_results(self, results: Dict[str, List[MethodResult]]):
        """Save results to CSV."""
        if not self.output_dir:
            return

        # Save per-netlist results
        all_rows = []
        for method_name, method_results in results.items():
            for i, r in enumerate(method_results):
                row = {"netlist_idx": i, "method": method_name}
                row.update(r.metrics.to_dict())
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(self.output_dir / "comparison_results.csv", index=False)

        # Save summary
        summary_df = self.summarize(results)
        summary_df.to_csv(self.output_dir / "comparison_summary.csv", index=False)

        print(f"\nResults saved to {self.output_dir}/")
