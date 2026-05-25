#!/usr/bin/env python3
"""
Train policy via Behavior Cloning on SA expert trajectories.

Usage:
    # Step 1: Generate SA trajectories
    python scripts/generate_sa_trajectories.py --num-trajectories 2000

    # Step 2: Imitation learning
    python scripts/train_imitation.py \
        --trajectory-dir data/sa_trajectories \
        --save-path checkpoints/mdp_policy_imitation.pt

    # Step 3: Fine-tune with PPO
    python scripts/train_mdp.py \
        --config configs/mdp_train.yaml \
        --save-path checkpoints/mdp_policy_finetuned.pt \
        --pretrained checkpoints/mdp_policy_imitation.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from src.models.policy_net import PolicyNet
from src.environment.mdp_env import EnvConfig
from src.training.imitation_trainer import ImitationTrainer


def main():
    parser = argparse.ArgumentParser(description="Imitation Learning for MDP policy")
    parser.add_argument("--trajectory-dir", type=str, required=True,
                        help="Directory with SA trajectory .pkl files")
    parser.add_argument("--config", type=str, default="configs/mdp_train.yaml")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/mdp_policy_imitation.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Load config for model architecture
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Device: {args.device}")

    # Policy network
    policy_cfg = cfg.get("policy", {})
    env_cfg_dict = cfg.get("environment", {})
    env_cfg = EnvConfig(**env_cfg_dict)

    policy = PolicyNet(**policy_cfg)

    # Trainer
    trainer = ImitationTrainer(
        policy_net=policy,
        env_config=env_cfg,
        lr=args.lr,
        device=args.device,
    )

    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Trajectory dir: {args.trajectory_dir}")

    # Train
    log = trainer.train(
        trajectory_dir=args.trajectory_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        log_interval=5,
    )

    # Final stats
    final_loss = log[-1]["loss"] if log else float('inf')
    final_acc = log[-1]["accuracy"] if log else 0.0
    print(f"\nImitation training complete.")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    main()
