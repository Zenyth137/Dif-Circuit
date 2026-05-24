"""
Diffusion Model Trainer.

Trains the Edge-GNN denoiser to predict noise added to clean placements.
Uses standard DDPM training objective: E[||ε - ε_θ(X_t, t, C)||²]

Training data: clean placements (could be from analytical placer or
hand-crafted layouts) → add noise → predict and denoise.

Reference: VLSI.tex Section 3.2
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from tqdm import tqdm

from ..models.diffusion import DiffusionPlacer


def _unwrap_batch(
    x_0: torch.Tensor,
    module_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove DataLoader batch dim (batch_size=1 adds a leading size-1 axis)."""
    if x_0.dim() == 3 and x_0.size(0) == 1:
        x_0 = x_0.squeeze(0)
    if module_sizes.dim() == 3 and module_sizes.size(0) == 1:
        module_sizes = module_sizes.squeeze(0)
    if edge_index.dim() == 3 and edge_index.size(0) == 1:
        edge_index = edge_index.squeeze(0)
    if edge_attr.dim() == 3 and edge_attr.size(0) == 1:
        edge_attr = edge_attr.squeeze(0)
    return x_0, module_sizes, edge_index, edge_attr


class DiffusionTrainer:
    """
    Trainer for the diffusion placement model.

    Trains the denoiser to predict noise from corrupted coordinates.
    """

    def __init__(self,
                 diffusion: DiffusionPlacer,
                 lr: float = 1e-4,
                 ema_decay: float = 0.9999,
                 max_grad_norm: float = 1.0,
                 device: str = "cpu",
                 clear_cuda_cache: bool = False,
                 model_config: Optional[dict] = None):
        self.diffusion = diffusion.to(device)
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.clear_cuda_cache = clear_cuda_cache
        self.model_config = model_config or {}

        self.optimizer = optim.AdamW(
            diffusion.denoiser.parameters(), lr=lr, weight_decay=1e-5
        )

        # EMA of denoiser parameters
        self.ema_params: Optional[dict] = None

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=1e-6
        )

        # Metrics
        self.losses: deque = deque(maxlen=1000)

    def _update_ema(self):
        """Update exponential moving average of denoiser parameters."""
        if self.ema_params is None:
            self.ema_params = {
                name: param.data.clone()
                for name, param in self.diffusion.denoiser.named_parameters()
            }
        else:
            for name, param in self.diffusion.denoiser.named_parameters():
                self.ema_params[name] = (
                    self.ema_decay * self.ema_params[name] +
                    (1 - self.ema_decay) * param.data
                )

    def _apply_ema(self):
        """Apply EMA parameters to denoiser (for inference)."""
        if self.ema_params is not None:
            for name, param in self.diffusion.denoiser.named_parameters():
                param.data.copy_(self.ema_params[name])

    def train_step(self,
                   x_0: torch.Tensor,
                   module_sizes: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: torch.Tensor) -> float:
        """
        Single training step.

        Args:
            x_0: (N, 2) clean placement coordinates
            module_sizes: (N, 2) module dimensions
            edge_index: (2, E) netlist edges
            edge_attr: (E, 1) edge weights

        Returns:
            loss: scalar MSE loss
        """
        x_0, module_sizes, edge_index, edge_attr = _unwrap_batch(
            x_0, module_sizes, edge_index, edge_attr,
        )

        self.diffusion.train()

        # Random timestep
        t = torch.randint(
            0, self.diffusion.timesteps, (1,),
            device=self.device
        )

        # Forward: add noise — x_0 is (N, 2), batch dim (1, N, 2) for forward_noise
        x_t, noise = self.diffusion.forward_noise(x_0.unsqueeze(0), t)
        x_t = x_t.squeeze(0)
        noise = noise.squeeze(0)

        # Predict noise
        eps_pred = self.diffusion.denoiser(
            x_t, t, module_sizes, edge_index, edge_attr
        )

        # MSE loss
        loss = nn.functional.mse_loss(eps_pred, noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.diffusion.denoiser.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()

        # Update EMA
        self._update_ema()

        self.losses.append(loss.item())

        if self.clear_cuda_cache and self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return loss.item()

    def train(self,
              dataloader,  # yields (x_0, module_sizes, edge_index, edge_attr)
              num_epochs: int = 100,
              log_interval: int = 10,
              save_path: Optional[str] = None) -> List[Dict]:
        """
        Main training loop.

        Args:
            dataloader: iterable yielding (x_0, module_sizes, edge_index, edge_attr)
            num_epochs: number of training epochs
            log_interval: logging frequency
            save_path: path to save checkpoint

        Returns:
            training_log: list of metrics per epoch
        """
        log = []
        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch in dataloader:
                x_0, module_sizes, edge_index, edge_attr = [
                    b.to(self.device) if isinstance(b, torch.Tensor) else b
                    for b in batch
                ]
                loss = self.train_step(x_0, module_sizes, edge_index, edge_attr)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            metrics = {
                "epoch": epoch,
                "loss": avg_loss,
                "lr": self.scheduler.get_last_lr()[0],
            }
            log.append(metrics)

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}, lr={metrics['lr']:.2e}")

            # Save best
            if avg_loss < best_loss and save_path:
                best_loss = avg_loss
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    "diffusion_state_dict": self.diffusion.state_dict(),
                    "diffusion_model_config": self.model_config,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "ema_params": self.ema_params,
                    "epoch": epoch,
                    "best_loss": best_loss,
                }, save_path)

        return log

    @torch.no_grad()
    def denoise(self,
                x_k: torch.Tensor,
                module_sizes: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                nets: Optional[list] = None,
                use_ema: bool = True,
                **sample_kwargs) -> torch.Tensor:
        """
        Denoise a coarse placement to legalized result.

        Args:
            x_k: (N, 2) coarse placement (from MDP)
            module_sizes: (N, 2)
            edge_index: (2, E)
            edge_attr: (E, 1)
            nets: optional net list for energy guidance
            use_ema: use EMA parameters for inference

        Returns:
            x_0: (N, 2) denoised placement
        """
        if use_ema:
            self._apply_ema()

        x_0 = self.diffusion.sample(
            x_k, module_sizes, edge_index, edge_attr, nets,
            **sample_kwargs
        )
        return x_0
