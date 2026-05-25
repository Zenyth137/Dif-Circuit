"""
Visualization tools for VLSI placement results.

Generates:
  - Placement scatter plots (module positions)
  - Congestion / density heatmaps
  - Training convergence curves (HPWL, reward, loss over iterations)
  - Method comparison charts
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from pathlib import Path

# ── Color schemes ──
CONG_CMAP = LinearSegmentedColormap.from_list(
    'congestion',
    [(0, '#1a1a2e'), (0.3, '#16213e'), (0.6, '#e94560'),
     (0.85, '#ff6b35'), (1.0, '#ffd700')]
)

METHOD_COLORS = {
    'SA': '#4fc3f7',
    'Analytical': '#81c784',
    'MDP': '#ff6b35',
    'MDP+Diffusion': '#e94560',
}

# ── Placement plot ──

def plot_placement(positions, nodes, title='Placement', save_path=None,
                   canvas_width=1000, canvas_height=1000, ax=None):
    """Scatter plot of module positions with size-coded rectangles."""
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for i in range(len(nodes)):
        cx, cy = positions[i, 0], positions[i, 1]
        w, h = nodes[i, 1], nodes[i, 2]
        rect = plt.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=0.5, edgecolor='#ff6b35',
            facecolor='#ff6b3522', zorder=2
        )
        ax.add_patch(rect)

    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.15)

    if create_fig and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ── Congestion heatmap ──

def plot_congestion(positions, nodes, nets, title='Congestion Heatmap',
                    save_path=None, canvas_width=1000, canvas_height=1000,
                    bins=64, ax=None):
    """Density/congestion heatmap from module placements."""
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    density = np.zeros((bins, bins), dtype=np.float32)
    bin_w = canvas_width / bins
    bin_h = canvas_height / bins

    for i in range(len(nodes)):
        cx, cy = positions[i, 0], positions[i, 1]
        w, h = nodes[i, 1], nodes[i, 2]
        bx0 = max(0, int((cx - w/2) / bin_w))
        bx1 = min(bins-1, int((cx + w/2) / bin_w))
        by0 = max(0, int((cy - h/2) / bin_h))
        by1 = min(bins-1, int((cy + h/2) / bin_h))
        if bx0 <= bx1 and by0 <= by1:
            density[by0:by1+1, bx0:bx1+1] += 1.0

    im = ax.imshow(density, origin='lower', cmap=CONG_CMAP,
                   vmin=0, vmax=max(3.0, density.max()),
                   extent=[0, canvas_width, 0, canvas_height],
                   interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Module Density')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if create_fig and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ── Training curves ──

def plot_training_curves(log_data, title='Training Progress', save_path=None):
    """Plot HPWL, reward, and loss curves from training log."""
    if isinstance(log_data, str):
        # Parse tqdm-style log
        hpwl_vals, reward_vals, iterations = [], [], []
        for line in open(log_data):
            if 'PPO Training' in line and 'reward=' in line:
                parts = line.split('reward=')[1].split(',')[0].strip()
                hpwl_parts = line.split('hpwl=')[1].split(']')[0].strip()
                iter_parts = line.split('|')[1].split('/')[0].strip()
                try:
                    reward_vals.append(float(parts))
                    hpwl_vals.append(float(hpwl_parts))
                    iterations.append(int(iter_parts))
                except ValueError:
                    pass
    else:
        iterations = [d.get('iteration', i) for i, d in enumerate(log_data)]
        hpwl_vals = [d.get('avg_hpwl', 0) for d in log_data]
        reward_vals = [d.get('avg_reward', 0) for d in log_data]

    if not iterations:
        print("No training data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # HPWL
    ax = axes[0]
    ax.plot(iterations, hpwl_vals, color='#ff6b35', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('HPWL')
    ax.set_title('HPWL over Training')
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[1]
    ax.plot(iterations, reward_vals, color='#4fc3f7', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Training')
    ax.grid(True, alpha=0.3)

    # Smoothed HPWL
    ax = axes[2]
    if len(hpwl_vals) > 20:
        window = min(20, len(hpwl_vals) // 5)
        kernel = np.ones(window) / window
        smoothed = np.convolve(hpwl_vals, kernel, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, color='#e94560', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('HPWL (smoothed)')
    ax.set_title('HPWL Trend (Moving Avg)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig

# ── Method comparison bar chart ──

def plot_comparison(results_dict, save_path=None):
    """Bar chart comparing HPWL across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Placement Method Comparison', fontsize=14, fontweight='bold')

    methods = list(results_dict.keys())
    hpwls = [results_dict[m].get('hpwl', 0) for m in methods]
    overlaps = [results_dict[m].get('overlap_pct', 0) for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]

    # HPWL
    axes[0].bar(methods, hpwls, color=colors, alpha=0.85)
    axes[0].set_ylabel('HPWL')
    axes[0].set_title('Wirelength (HPWL)')
    for i, v in enumerate(hpwls):
        axes[0].text(i, v + max(hpwls)*0.02, f'{v:.0f}', ha='center', fontsize=9)

    # Overlap
    axes[1].bar(methods, overlaps, color=colors, alpha=0.85)
    axes[1].set_ylabel('Overlap %')
    axes[1].set_title('Overlap')
    for i, v in enumerate(overlaps):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Main: generate all visualizations ──

def generate_all_visualizations(output_dir='results/visualizations',
                                mdp_positions=None, nodes=None, nets=None,
                                sa_positions=None, analytical_positions=None,
                                training_log=None, comparison_results=None):
    """Generate a full suite of visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Placement plots for each method
    if nodes is not None:
        if sa_positions is not None:
            plot_placement(sa_positions, nodes, title='SA Placement',
                           save_path=os.path.join(output_dir, 'placement_sa.png'))
        if analytical_positions is not None:
            plot_placement(analytical_positions, nodes, title='Analytical Placement',
                           save_path=os.path.join(output_dir, 'placement_analytical.png'))
        if mdp_positions is not None:
            plot_placement(mdp_positions, nodes, title='MDP Placement',
                           save_path=os.path.join(output_dir, 'placement_mdp.png'))

            # Congestion heatmap
            if nets is not None:
                plot_congestion(mdp_positions, nodes, nets, title='MDP Density Map',
                                save_path=os.path.join(output_dir, 'congestion_mdp.png'))

    # Training curves
    if training_log and os.path.exists(training_log):
        plot_training_curves(training_log, save_path=os.path.join(output_dir, 'training_curves.png'))

    print(f"Visualizations saved to {output_dir}/")
