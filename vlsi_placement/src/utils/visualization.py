"""
Visualization utilities for placement results.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path


def plot_placement(
    positions: np.ndarray,
    nodes: np.ndarray,
    nets: Optional[list] = None,
    canvas_width: float = 1000.0,
    canvas_height: float = 1000.0,
    title: str = "Placement",
    save_path: Optional[str] = None,
    show_nets: bool = False,
    figsize: Tuple[int, int] = (10, 10),
):
    """
    Plot module placement with optional net connections.

    Args:
        positions: (N, 2) module centers
        nodes: (N, 3) [id, width, height]
        nets: optional net connections
        canvas_width, canvas_height: canvas dimensions
        title: plot title
        save_path: path to save figure
        show_nets: if True, draw net connections as lines
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw canvas
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')

    # Draw modules
    colors = plt.cm.viridis(np.linspace(0, 1, len(nodes)))
    for i in range(len(nodes)):
        cx, cy = positions[i]
        w, h = nodes[i, 1], nodes[i, 2]

        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            linewidth=0.5, edgecolor='black',
            facecolor=colors[i], alpha=0.7
        )
        ax.add_patch(rect)

    # Draw net connections
    if show_nets and nets is not None:
        for net in nets[:100]:  # Limit to first 100 nets
            if len(net) < 2:
                continue
            net_positions = positions[net]
            for j in range(len(net) - 1):
                ax.plot(
                    [net_positions[j, 0], net_positions[j + 1, 0]],
                    [net_positions[j, 1], net_positions[j + 1, 1]],
                    'r-', linewidth=0.2, alpha=0.3
                )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_comparison(
    results: Dict[str, List['MethodResult']],
    metric: str = "hpwl",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot comparison of placement methods on a given metric.

    Args:
        results: method_name -> list of MethodResult
        metric: one of 'hpwl', 'overlap_pct', 'congestion', 'runtime'
        save_path: path to save figure
    """
    import pandas as pd

    method_names = list(results.keys())
    num_netlists = len(next(iter(results.values())))

    data = []
    for method_name, method_results in results.items():
        for i, r in enumerate(method_results):
            val = getattr(r.metrics, metric, 0.0)
            data.append({
                "Method": method_name,
                "Netlist": i,
                metric: val,
            })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot: mean ± std
    ax1 = axes[0]
    means = df.groupby("Method")[metric].mean()
    stds = df.groupby("Method")[metric].std()
    x = np.arange(len(means))
    bars = ax1.bar(x, means.values, yerr=stds.values, capsize=5,
                   color=plt.cm.Set2(np.linspace(0, 1, len(means))))
    ax1.set_xticks(x)
    ax1.set_xticklabels(means.index, rotation=45, ha='right')
    ax1.set_ylabel(metric)
    ax1.set_title(f"{metric} by Method (mean ± std)")

    # Box plot
    ax2 = axes[1]
    data_for_box = [df[df["Method"] == m][metric].values for m in method_names]
    bp = ax2.boxplot(data_for_box, labels=method_names, patch_artist=True)
    for patch, color in zip(bp['boxes'],
                            plt.cm.Set2(np.linspace(0, 1, len(method_names)))):
        patch.set_facecolor(color)
    ax2.set_ylabel(metric)
    ax2.set_title(f"{metric} Distribution")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
