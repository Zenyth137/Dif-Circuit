"""
Imitation Learning (Behavior Cloning) Trainer for MDP placement policy.

Trains PolicyNet to imitate SA expert placements via cross-entropy loss
on per-step action predictions. After pre-training, the policy can be
fine-tuned with PPO.

Architecture:
  1. Load SA trajectories (module order + target grid positions)
  2. For each step, run GNN encoder + state projection → action logits
  3. Compute cross-entropy loss against expert action
  4. Update policy weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Tuple
from collections import deque
from tqdm import tqdm

from ..models.policy_net import PolicyNet
from ..environment.mdp_env import MacroPlacementEnv, EnvConfig
from ..netlist.generator import NetlistGenerator


class ImitationTrainer:
    """
    Behavior cloning trainer — learns to predict expert (SA) actions.

    Trains the full PolicyNet (GNN encoder + state projector + policy head)
    by minimizing cross-entropy between predicted action distributions
    and expert grid positions.
    """

    def __init__(self,
                 policy_net: PolicyNet,
                 env_config: EnvConfig,
                 lr: float = 1e-3,
                 device: str = "cpu"):
        self.policy = policy_net.to(device)
        self.env_config = env_config
        self.env = MacroPlacementEnv(env_config)
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.losses: deque = deque(maxlen=100)
        self.accuracies: deque = deque(maxlen=100)

    def prepare_dataset(self, trajectories: List[dict]) -> List[Tuple[torch.Tensor, int]]:
        """
        Pre-extract all (state_feature, expert_action) pairs from trajectories.
        Called once before training — the GNN encoder runs only once per trajectory.
        """
        self.policy.eval()
        all_pairs = []
        print(f"  Extracting features from {len(trajectories)} trajectories...")
        for traj in tqdm(trajectories, desc="  Pre-extracting"):
            pairs = self._extract_pairs(traj)
            all_pairs.extend(pairs)
        print(f"  Total training pairs: {len(all_pairs)}")
        return all_pairs

    def train_epoch(self,
                    all_pairs: List[Tuple[torch.Tensor, int]],
                    batch_size: int = 16) -> Dict[str, float]:
        """
        One epoch over pre-extracted (state_feature, expert_action) pairs.
        """
        self.policy.train()

        if not all_pairs:
            return {"loss": 0.0, "accuracy": 0.0}

        # Shuffle and batch
        indices = np.random.permutation(len(all_pairs))
        total_loss = 0.0
        total_correct = 0
        num_batches = 0

        for start in range(0, len(all_pairs), batch_size):
            end = min(start + batch_size, len(all_pairs))
            batch_idx = indices[start:end]

            batch_states = torch.stack([all_pairs[i][0] for i in batch_idx]).to(self.device)
            batch_actions = torch.tensor(
                [all_pairs[i][1] for i in batch_idx],
                dtype=torch.long, device=self.device
            )

            # Forward: policy head only (state features already computed)
            logits = self.policy.policy_head.forward(batch_states)
            flat_logits = logits.view(logits.size(0), -1)

            loss = self.criterion(flat_logits, batch_actions)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            # Accuracy
            pred = flat_logits.argmax(dim=-1)
            correct = (pred == batch_actions).sum().item()

            total_loss += loss.item()
            total_correct += correct
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = total_correct / len(all_pairs)

        self.losses.append(avg_loss)
        self.accuracies.append(accuracy)

        return {"loss": avg_loss, "accuracy": accuracy}

    def _extract_pairs(self, trajectory: dict) -> List[Tuple[torch.Tensor, int]]:
        """
        Extract (state_feature, expert_action) pairs from one trajectory.

        Replays the sequential placement process: at each step, we compute
        the state features (using the modules placed so far) and record
        the expert's target grid action.
        """
        nodes = trajectory["nodes"]
        nets = trajectory["nets"]
        module_order = trajectory["module_order"]
        steps = trajectory["steps"]

        # Build module features tensor
        module_features = torch.tensor(
            nodes[:, 1:], dtype=torch.float32, device=self.device
        )  # (N, 2) — [w, h]
        net_features = torch.ones(len(nets), 1, dtype=torch.float32, device=self.device)

        # Build edge index (from nets)
        edge_index, edge_attr = self._build_edge_index(nodes, nets)

        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)

        # GNN encode once per trajectory
        with torch.no_grad():
            _, global_emb = self.policy.encode_graph(
                module_features, net_features, edge_index_t, edge_attr_t
            )

        # Reset a temporary environment to track density grid
        self.env.reset(nodes, nets)

        pairs = []

        for step_idx, step_info in enumerate(steps):
            mid = step_info["module_id"]
            w, h = step_info["w"], step_info["h"]
            expert_action = step_info["action_idx"]

            # Build density grid tensor for current state
            density_tensor = torch.from_numpy(
                self.env.density_grid.copy()
            ).float().to(self.device)

            # Compute state features (no grad — we just need the feature vector)
            with torch.no_grad():
                state_feat = self.policy.build_state_features(
                    global_emb, w, h, density_tensor
                )

            pairs.append((state_feat.detach().cpu(), expert_action))

            # Step the environment to update density grid for next step
            gx, gy = step_info["gx"], step_info["gy"]
            self.env.step((gx, gy))

        return pairs

    def _build_edge_index(self, nodes: np.ndarray, nets: list):
        """Build bipartite edge index (module <-> net)."""
        num_modules = len(nodes)
        edges_src = []
        edges_dst = []
        edges_attr = []

        for net_idx, net in enumerate(nets):
            for module_idx in net:
                if module_idx < num_modules:
                    edges_src.append(module_idx)
                    edges_dst.append(num_modules + net_idx)
                    edges_attr.append(1.0)
                    edges_src.append(num_modules + net_idx)
                    edges_dst.append(module_idx)
                    edges_attr.append(1.0)

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_attr = np.array([edges_attr], dtype=np.float32).T
        return edge_index, edge_attr

    def train(self,
              trajectory_dir: str,
              num_epochs: int = 50,
              batch_size: int = 64,
              save_path: Optional[str] = None,
              log_interval: int = 5) -> List[Dict]:
        """
        Full training loop over saved trajectory files.

        Args:
            trajectory_dir: directory containing trajectory .pkl files
            num_epochs: number of passes over the dataset
            batch_size: training batch size
            save_path: path to save best model
            log_interval: logging frequency (epochs)
        """
        # Load all trajectories
        print(f"Loading trajectories from {trajectory_dir}...")
        all_trajectories = []
        for fname in sorted(os.listdir(trajectory_dir)):
            if fname.endswith('.pkl'):
                with open(os.path.join(trajectory_dir, fname), 'rb') as f:
                    all_trajectories.extend(pickle.load(f))
        print(f"Loaded {len(all_trajectories)} trajectories")

        # Pre-extract all training pairs once (GNN runs once per trajectory)
        all_pairs = self.prepare_dataset(all_trajectories)

        log = []
        best_loss = float('inf')

        pbar = tqdm(range(num_epochs), desc="Imitation Training")
        for epoch in pbar:
            metrics = self.train_epoch(all_pairs, batch_size=batch_size)
            metrics["epoch"] = epoch
            log.append(metrics)

            if epoch % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.3f}",
                })

            # Save best
            if metrics["loss"] < best_loss and save_path:
                best_loss = metrics["loss"]
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    "policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                }, save_path)

        return log
