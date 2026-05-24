#!/usr/bin/env python3
"""
Full evaluation: MDP → Diffusion pipeline vs baselines on test netlists.

Usage:
    python scripts/evaluate.py --mdp-checkpoint checkpoints/mdp_policy.pt \\
                                --diffusion-checkpoint checkpoints/diffusion.pt \\
                                --data-dir data/test_netlists \\
                                --output results/comparison
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import yaml
from pathlib import Path

from src.models.policy_net import PolicyNet
from src.models.diffusion import DiffusionPlacer
from src.environment.mdp_env import MacroPlacementEnv, EnvConfig
from src.baselines.simulated_annealing import SimulatedAnnealing, SAConfig
from src.baselines.analytical import AnalyticalPlacer
from src.netlist.parser import NetlistParser
from src.netlist.generator import NetlistGenerator, NetlistConfig
from src.evaluation.compare import ComparisonRunner


def mdp_pipeline_fn(policy, env_config, device):
    """Create a closure for the MDP pipeline."""
    def pipeline(nodes, nets):
        generator = NetlistGenerator(NetlistConfig(
            num_modules=len(nodes), num_nets=len(nets),
        ))
        generator._nodes = nodes
        generator._nets = nets
        edge_index, edge_attr = generator.build_edge_index()
        adj = generator.build_adjacency()

        module_features = torch.tensor(nodes[:, 1:], dtype=torch.float32, device=device)
        net_features = torch.ones(len(nets), 1, dtype=torch.float32, device=device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

        env = MacroPlacementEnv(env_config)
        env.reset(nodes, nets)

        with torch.no_grad():
            for step in range(env.num_modules):
                w = float(nodes[step, 1])
                h = float(nodes[step, 2])
                result = policy.get_action(
                    module_features, net_features, edge_index_t, edge_attr_t,
                    w, h, deterministic=True,
                )
                action_idx = int(result["action_idx"].item())
                gy = action_idx % env.grid_size
                gx = action_idx // env.grid_size
                env.step((gx, gy))

        return env.get_placement()
    return pipeline


def mdp_diffusion_fn(policy, diffusion, env_config, device):
    """Create a closure for the MDP+Diffusion pipeline."""
    def pipeline(nodes, nets):
        generator = NetlistGenerator(NetlistConfig(
            num_modules=len(nodes), num_nets=len(nets),
        ))
        generator._nodes = nodes
        generator._nets = nets
        edge_index, edge_attr = generator.build_edge_index()

        module_features = torch.tensor(nodes[:, 1:], dtype=torch.float32, device=device)
        module_sizes = module_features
        net_features = torch.ones(len(nets), 1, dtype=torch.float32, device=device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

        env = MacroPlacementEnv(env_config)
        env.reset(nodes, nets)

        with torch.no_grad():
            for step in range(env.num_modules):
                w = float(nodes[step, 1])
                h = float(nodes[step, 2])
                result = policy.get_action(
                    module_features, net_features, edge_index_t, edge_attr_t,
                    w, h, deterministic=True,
                )
                action_idx = int(result["action_idx"].item())
                gy = action_idx % env.grid_size
                gx = action_idx // env.grid_size
                env.step((gx, gy))

        coarse_positions = torch.tensor(env.get_placement(), dtype=torch.float32, device=device)

        # Diffusion refinement
        refined = diffusion.sample(
            coarse_positions, module_sizes, edge_index_t, edge_attr_t,
            nets=nets,
        )
        return refined.cpu().numpy()
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Full evaluation of placement methods")
    parser.add_argument("--mdp-checkpoint", type=str, default=None,
                        help="Path to trained MDP policy checkpoint")
    parser.add_argument("--diffusion-checkpoint", type=str, default=None,
                        help="Path to trained diffusion model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/test_netlists")
    parser.add_argument("--num-netlists", type=int, default=10)
    parser.add_argument("--prefix", type=str, default="netlist")
    parser.add_argument("--output", type=str, default="results/comparison")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-mdp", action="store_true",
                        help="Skip MDP-based methods (only run baselines)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    env_config = EnvConfig(**cfg.get("environment", {}))

    # Load test netlists
    test_netlists = []
    for i in range(args.num_netlists):
        try:
            nodes, nets = NetlistParser.load(args.data_dir, f"{args.prefix}_{i:04d}")
            test_netlists.append((nodes, nets))
        except FileNotFoundError:
            print(f"Warning: {args.prefix}_{i:04d} not found, skipping")
            continue

    if not test_netlists:
        print("No test netlists found!")
        return

    print(f"Loaded {len(test_netlists)} test netlists")

    # Build methods dict
    methods = {}

    # Baselines (always available)
    sa = SimulatedAnnealing(SAConfig(seed=42))
    methods["SA"] = lambda nodes, nets: sa.place(
        nodes, nets, verbose=False
    )[0]  # return only positions

    analytical = AnalyticalPlacer()
    methods["Analytical"] = lambda nodes, nets: analytical.place(
        nodes, nets, verbose=False
    )[0]

    # MDP-based methods (if checkpoints provided)
    if not args.skip_mdp and args.mdp_checkpoint:
        # Load policy
        policy = PolicyNet(**cfg.get("policy", {}))
        checkpoint = torch.load(args.mdp_checkpoint, map_location=args.device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(args.device)
        policy.eval()
        print(f"Loaded MDP policy from {args.mdp_checkpoint}")

        methods["MDP"] = mdp_pipeline_fn(policy, env_config, args.device)

        # MDP + Diffusion
        if args.diffusion_checkpoint:
            sample_nodes = test_netlists[0][0]
            diffusion = DiffusionPlacer(
                num_modules=len(sample_nodes),
                **cfg.get("diffusion", {}),
            )
            diff_checkpoint = torch.load(args.diffusion_checkpoint, map_location=args.device)
            diffusion.load_state_dict(diff_checkpoint["diffusion_state_dict"])
            diffusion.to(args.device)
            diffusion.eval()
            print(f"Loaded diffusion model from {args.diffusion_checkpoint}")

            methods["MDP+Diffusion"] = mdp_diffusion_fn(
                policy, diffusion, env_config, args.device
            )
    else:
        print("MDP checkpoint not provided. Running baselines only.")

    # Run comparison
    runner = ComparisonRunner(
        methods=methods,
        test_netlists=test_netlists,
        output_dir=args.output,
    )
    results = runner.run(verbose=True)
    runner.print_summary(results)


if __name__ == "__main__":
    main()
