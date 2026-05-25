#!/usr/bin/env python3
"""
Train the MDP policy network using PPO.

Usage:
    python scripts/train_mdp.py --config configs/mdp_train.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from src.models.policy_net import PolicyNet
from src.environment.mdp_env import EnvConfig
from src.netlist.generator import NetlistGenerator, NetlistConfig
from src.training.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train MDP placement policy")
    parser.add_argument("--config", type=str, default="configs/mdp_train.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/mdp_policy.pt")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint (from imitation learning)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Device: {args.device}")
    print(f"Config: {cfg}")

    # Netlist generator
    netlist_cfg = NetlistConfig(**cfg.get("netlist", {}))
    generator = NetlistGenerator(netlist_cfg)

    # Environment config
    env_cfg = EnvConfig(**cfg.get("environment", {}))

    # Policy network
    policy = PolicyNet(**cfg.get("policy", {}))

    # Load pretrained weights (from imitation learning)
    if args.pretrained:
        print(f"Loading pretrained policy from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=args.device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        print(f"  Pretrained loss: {ckpt.get('loss', 'N/A')}")

    # Trainer (with optional curiosity)
    ppo_cfg = cfg.get("ppo", {})
    curiosity_cfg = cfg.get("curiosity", {})
    trainer = PPOTrainer(
        policy_net=policy,
        env_config=env_cfg,
        device=args.device,
        **ppo_cfg,
        **curiosity_cfg,
    )

    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Train
    log = trainer.train(
        generator=generator,
        num_iterations=cfg.get("num_iterations", 1000),
        episodes_per_iter=cfg.get("episodes_per_iter", 16),
        save_path=args.save_path,
        early_stop_patience=cfg.get("early_stop_patience", 50),
    )

    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate(generator, num_episodes=20)
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
