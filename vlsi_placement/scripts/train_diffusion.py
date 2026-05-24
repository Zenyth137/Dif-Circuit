#!/usr/bin/env python3
"""
Train the diffusion denoising model using pre-computed ground-truth placements.

The ground-truth X_0 placements should be generated beforehand using:
    python scripts/generate_ground_truth.py --num-modules 200 --num-nets 800 \\
        --num-samples 100 --output data/ground_truth

Usage:
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
from src.netlist.parser import NetlistParser
from src.netlist.prepare import truncate_netlist, prepare_netlist_graph
from src.training.diffusion_trainer import DiffusionTrainer


class GroundTruthDataset(torch.utils.data.Dataset):
    """Loads pre-computed (netlist, X_0) pairs; optional module cap for GPU memory."""

    def __init__(self, data_dir: str, num_samples: int, prefix: str = "gt",
                 max_modules: int = None):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.prefix = prefix
        self.max_modules = max_modules

        sample_name = f"{prefix}_0000"
        nodes_path = os.path.join(data_dir, f"{sample_name}_nodes.csv")
        placement_path = os.path.join(data_dir, f"{sample_name}_placement.npy")
        if not os.path.exists(nodes_path):
            raise FileNotFoundError(
                f"Ground truth data not found at {data_dir}/{sample_name}_*. "
                f"Run 'scripts/generate_ground_truth.py' first."
            )
        if not os.path.exists(placement_path):
            raise FileNotFoundError(
                f"Placement file not found: {placement_path}."
            )

        cap = f", max_modules={max_modules}" if max_modules else ""
        print(f"GroundTruthDataset: {num_samples} samples from {data_dir}/{prefix}_*{cap}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_name = f"{self.prefix}_{idx:04d}"

        nodes, nets = NetlistParser.load(self.data_dir, sample_name)
        positions = np.load(
            os.path.join(self.data_dir, f"{sample_name}_placement.npy")
        ).astype(np.float32)

        if self.max_modules is not None:
            nodes, nets = truncate_netlist(nodes, nets, self.max_modules)
            positions = positions[:len(nodes)]

        _, _, edge_index, edge_attr = prepare_netlist_graph(
            nodes, nets, bipartite=False,
        )

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
    parser.add_argument("--data-dir", type=str, default="data/ground_truth")
    parser.add_argument("--gt-prefix", type=str, default="gt")
    parser.add_argument("--save-path", type=str, default="checkpoints/diffusion.pt")
    parser.add_argument("--max-modules", type=int, default=None,
                        help="Override config max_modules cap")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    max_modules = args.max_modules or cfg.get("max_modules")
    num_train_samples = cfg.get("num_train_samples", 100)

    print(f"Device: {args.device}")
    print(f"Data dir: {args.data_dir}")
    print(f"GT prefix: {args.gt_prefix}")
    if max_modules:
        print(f"Max modules (GPU cap): {max_modules}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    dataset = GroundTruthDataset(
        data_dir=args.data_dir,
        num_samples=num_train_samples,
        prefix=args.gt_prefix,
        max_modules=max_modules,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    sample_nodes, _ = NetlistParser.load(args.data_dir, f"{args.gt_prefix}_0000")
    num_modules = len(sample_nodes)
    if max_modules is not None:
        num_modules = min(num_modules, max_modules)

    # Keys used only at inference / in evaluate.py, not by DiffusionPlacer.__init__
    _DIFFUSION_INIT_SKIP = frozenset({"inference_num_steps"})
    diffusion_cfg = {
        k: v for k, v in cfg.get("diffusion", {}).items()
        if k not in _DIFFUSION_INIT_SKIP
    }
    diffusion = DiffusionPlacer(num_modules=num_modules, **diffusion_cfg)

    trainer = DiffusionTrainer(
        diffusion=diffusion,
        device=args.device,
        clear_cuda_cache=args.device.startswith("cuda"),
        model_config=diffusion_cfg,
        **cfg.get("training", {}),
    )

    print(f"Denoiser params: {sum(p.numel() for p in diffusion.denoiser.parameters()):,}")
    print(f"Modules: {num_modules}, timesteps: {diffusion.timesteps}")
    print()

    log = trainer.train(
        dataloader=dataloader,
        num_epochs=cfg.get("num_epochs", 100),
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
