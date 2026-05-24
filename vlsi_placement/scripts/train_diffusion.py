#!/usr/bin/env python3
"""
Train the diffusion denoising model using pre-computed ground-truth placements.

The ground-truth X_0 placements should be generated beforehand using:
    python scripts/generate_ground_truth.py --num-modules 5000 --num-samples 100

This ensures every training sample has ABSOLUTELY ZERO overlap (guaranteed
by the zero-overlap legalizer), avoiding the "pseudo-ground-truth" trap.

Usage:
    # Step 1: Generate ground truth data
    python scripts/generate_ground_truth.py --num-modules 5000 --num-nets 5000 \\
        --num-samples 100 --output data/ground_truth

    # Step 2: Train diffusion model
    python scripts/train_diffusion.py --config configs/diffusion_train.yaml \\
        --data-dir data/ground_truth --gt-prefix gt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import yaml
from src.models.diffusion import DiffusionPlacer
from src.netlist.generator import NetlistGenerator, NetlistConfig
from src.netlist.parser import NetlistParser
from src.training.diffusion_trainer import DiffusionTrainer


class GroundTruthDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-computed (netlist, X_0) pairs.

    Each sample consists of:
      - nodes.csv + nets.csv: the netlist
      - *_placement.npy: the pre-computed zero-overlap X_0 placement

    The X_0 placements are generated offline by the zero-overlap legalizer
    and are guaranteed to have absolutely no module overlap.
    """

    def __init__(self, data_dir: str, num_samples: int, prefix: str = "gt"):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.prefix = prefix

        # Verify first sample exists
        sample_name = f"{prefix}_0000"
        nodes_path = os.path.join(data_dir, f"{sample_name}_nodes.csv")
        placement_path = os.path.join(data_dir, f"{sample_name}_placement.npy")
        if not os.path.exists(nodes_path):
            raise FileNotFoundError(
                f"Ground truth data not found at {data_dir}/{sample_name}_*. "
                f"Run 'scripts/generate_ground_truth.py' first to generate X_0 placements."
            )
        if not os.path.exists(placement_path):
            raise FileNotFoundError(
                f"Placement file not found: {placement_path}. "
                f"Run 'scripts/generate_ground_truth.py' with the same prefix."
            )

        print(f"GroundTruthDataset: {num_samples} samples from {data_dir}/{prefix}_*")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_name = f"{self.prefix}_{idx:04d}"

        # Load netlist
        nodes, nets = NetlistParser.load(self.data_dir, sample_name)

        # Load pre-computed ground-truth placement (guaranteed zero-overlap)
        placement_path = os.path.join(self.data_dir, f"{sample_name}_placement.npy")
        positions = np.load(placement_path).astype(np.float32)

        # Build graph structure
        gen = NetlistGenerator(NetlistConfig(
            num_modules=len(nodes),
            num_nets=len(nets),
        ))
        gen._nodes = nodes
        gen._nets = nets
        edge_index, edge_attr = gen.build_edge_index()

        # Convert to tensors
        x_0 = torch.tensor(positions, dtype=torch.float32)
        module_sizes = torch.tensor(nodes[:, 1:], dtype=torch.float32)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        return x_0, module_sizes, edge_index_t, edge_attr_t


def main():
    parser = argparse.ArgumentParser(
        description="Train diffusion denoising model with ground-truth placements"
    )
    parser.add_argument("--config", type=str, default="configs/diffusion_train.yaml")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="data/ground_truth",
                        help="Directory with pre-computed (netlist, X_0) pairs")
    parser.add_argument("--gt-prefix", type=str, default="gt",
                        help="Prefix used for ground-truth files")
    parser.add_argument("--save-path", type=str, default="checkpoints/diffusion.pt")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Device: {args.device}")
    print(f"Data dir: {args.data_dir}")
    print(f"GT prefix: {args.gt_prefix}")

    num_train_samples = cfg.get("num_train_samples", 100)

    # Dataset — loads pre-computed zero-overlap X_0 placements
    dataset = GroundTruthDataset(
        data_dir=args.data_dir,
        num_samples=num_train_samples,
        prefix=args.gt_prefix,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,      # Each netlist can have different sizes
        shuffle=True,
        num_workers=0,     # Avoid multiprocessing issues with .npy loading
    )

    # Initialize diffusion model
    sample_nodes, _ = NetlistParser.load(
        args.data_dir, f"{args.gt_prefix}_0000"
    )
    num_modules = len(sample_nodes)

    diffusion_cfg = cfg.get("diffusion", {})
    diffusion = DiffusionPlacer(
        num_modules=num_modules,
        **diffusion_cfg,
    )

    # Trainer
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        device=args.device,
        **cfg.get("training", {}),
    )

    print(f"Denoiser params: {sum(p.numel() for p in diffusion.denoiser.parameters()):,}")
    print(f"Canvas: {diffusion.canvas_width}×{diffusion.canvas_height}")
    print(f"Timesteps: {diffusion.timesteps}")
    print()

    # Train
    log = trainer.train(
        dataloader=dataloader,
        num_epochs=cfg.get("num_epochs", 100),
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
