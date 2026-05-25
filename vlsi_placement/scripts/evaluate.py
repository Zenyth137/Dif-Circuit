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

from src.models.policy_net import PolicyNet
from src.models.diffusion import DiffusionPlacer
from src.environment.mdp_env import MacroPlacementEnv, EnvConfig
from src.baselines.simulated_annealing import SimulatedAnnealing, SAConfig
from src.baselines.analytical import AnalyticalPlacer
from src.netlist.parser import NetlistParser
from src.netlist.prepare import prepare_netlist_graph
from src.evaluation.compare import ComparisonRunner
from src.utils.checkpoint import load_checkpoint, resolve_diffusion_model_config


def _clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_mdp_placement(policy, env_config, device, nodes, nets, max_modules):
    """MDP coarse placement with one GNN encode per netlist."""
    nodes, nets, edge_index, edge_attr = prepare_netlist_graph(
        nodes, nets, max_modules=max_modules, bipartite=True,
    )

    module_features = torch.tensor(nodes[:, 1:], dtype=torch.float32, device=device)
    net_features = torch.ones(len(nets), 1, dtype=torch.float32, device=device)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    env = MacroPlacementEnv(env_config)
    state = env.reset(nodes, nets)

    with torch.no_grad():
        _, global_emb = policy.encode_graph(
            module_features, net_features, edge_index_t, edge_attr_t,
        )
        for step in range(env.num_modules):
            w = float(nodes[step, 1])
            h = float(nodes[step, 2])
            density_tensor = torch.from_numpy(
                env.density_grid.copy()
            ).float().to(device)
            mask = state.get("action_mask")
            mask_tensor = torch.from_numpy(mask).float().to(device) if mask is not None else None
            result = policy.get_action(
                module_features, net_features, edge_index_t, edge_attr_t,
                w, h, deterministic=True, global_emb=global_emb,
                density_grid=density_tensor,
                action_mask=mask_tensor,
            )
            action_idx = int(result["action_idx"].item())
            gy = action_idx % env.grid_size
            gx = action_idx // env.grid_size
            state, _, _, _ = env.step((gx, gy))

    return env.get_placement()


def mdp_pipeline_fn(policy, env_config, device, max_modules):
    def pipeline(nodes, nets):
        return _run_mdp_placement(policy, env_config, device, nodes, nets, max_modules)
    return pipeline


def mdp_diffusion_fn(policy, diffusion, env_config, device, max_modules, inference_num_steps):
    def pipeline(nodes, nets):
        nodes, nets, edge_index, edge_attr = prepare_netlist_graph(
            nodes, nets, max_modules=max_modules, bipartite=False,
        )

        coarse = _run_mdp_placement(
            policy, env_config, device, nodes, nets, max_modules,
        )

        module_sizes = torch.tensor(nodes[:, 1:], dtype=torch.float32, device=device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        coarse_t = torch.tensor(coarse, dtype=torch.float32, device=device)

        start_t = diffusion.calibrate_start_t(env_config.cell_w, env_config.cell_h)
        refined = diffusion.sample(
            coarse_t, module_sizes, edge_index_t, edge_attr_t,
            nets=nets,
            start_t=start_t,
            num_steps=inference_num_steps,
        )
        return refined.detach().cpu().numpy()
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
    parser.add_argument("--config", type=str, default="configs/mdp_train.yaml",
                        help="MDP policy + environment settings")
    parser.add_argument("--diffusion-config", type=str, default=None,
                        help="Fallback diffusion yaml if not stored in checkpoint "
                             "(e.g. configs/diffusion_train_8gb.yaml)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-modules", type=int, default=None,
                        help="Cap modules for GPU methods (default: environment.max_modules)")
    parser.add_argument("--inference-num-steps", type=int, default=None,
                        help="Diffusion reverse steps (default: diffusion.inference_num_steps)")
    parser.add_argument("--skip-mdp", action="store_true",
                        help="Skip MDP-based methods (only run baselines)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    diffusion_yaml_cfg = {}
    if args.diffusion_config:
        with open(args.diffusion_config, 'r') as f:
            diffusion_yaml_cfg = yaml.safe_load(f).get("diffusion", {})
    else:
        diffusion_yaml_cfg = cfg.get("diffusion", {})

    env_config = EnvConfig(**cfg.get("environment", {}))
    max_modules = args.max_modules or env_config.max_modules
    inference_num_steps = (
        args.inference_num_steps
        if args.inference_num_steps is not None
        else diffusion_yaml_cfg.get("inference_num_steps", 50)
    )

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
    print(f"GPU cap: max_modules={max_modules}, diffusion steps={inference_num_steps}")

    methods = {}

    sa = SimulatedAnnealing(SAConfig(seed=42))
    methods["SA"] = lambda nodes, nets: sa.place(nodes, nets, verbose=False)[0]

    analytical = AnalyticalPlacer()
    methods["Analytical"] = lambda nodes, nets: analytical.place(nodes, nets, verbose=False)[0]

    if not args.skip_mdp and args.mdp_checkpoint:
        policy = PolicyNet(**cfg.get("policy", {}))
        checkpoint = load_checkpoint(args.mdp_checkpoint, args.device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(args.device)
        policy.eval()
        print(f"Loaded MDP policy from {args.mdp_checkpoint}")

        methods["MDP"] = mdp_pipeline_fn(
            policy, env_config, args.device, max_modules,
        )

        if args.diffusion_checkpoint:
            sample_nodes, _ = test_netlists[0]
            n_modules = min(len(sample_nodes), max_modules)
            diff_checkpoint = load_checkpoint(args.diffusion_checkpoint, args.device)
            diffusion_model_cfg = resolve_diffusion_model_config(
                diff_checkpoint, yaml_defaults=diffusion_yaml_cfg,
            )
            diffusion = DiffusionPlacer(
                num_modules=n_modules,
                **diffusion_model_cfg,
            )
            diffusion.load_state_dict(diff_checkpoint["diffusion_state_dict"])
            if "ema_params" in diff_checkpoint:
                diffusion.load_ema(diff_checkpoint["ema_params"])
                print("Loaded diffusion EMA weights from checkpoint")
            diffusion.to(args.device)
            diffusion.eval()
            print(f"Loaded diffusion model from {args.diffusion_checkpoint}")

            methods["MDP+Diffusion"] = mdp_diffusion_fn(
                policy, diffusion, env_config, args.device,
                max_modules, inference_num_steps,
            )
    else:
        print("MDP checkpoint not provided. Running baselines only.")

    runner = ComparisonRunner(
        methods=methods,
        test_netlists=test_netlists,
        output_dir=args.output,
    )
    results = runner.run(verbose=True)
    runner.print_summary(results)

    _clear_cuda_cache(args.device)


if __name__ == "__main__":
    main()
