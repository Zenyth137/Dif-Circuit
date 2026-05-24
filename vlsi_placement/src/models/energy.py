"""
Differentiable energy functions for guided diffusion sampling.

Energy terms:
  1. Overlap energy: penalizes module overlap (soft penalty, continuous)
  2. HPWL energy: encourages short wirelength

The gradient of these energies w.r.t. module coordinates provides
a force field that guides the reverse diffusion toward legal,
compact placements.

Reference: VLSI.tex Section 3.2.3 — Energy-Guided Sampling
"""

import torch
import torch.nn as nn
from typing import Optional


class OverlapEnergy(nn.Module):
    """
    Differentiable overlap penalty.

    E_ovlp = Σ_{i≠j} max(0, Δx_ij) · max(0, Δy_ij)

    where Δx_ij = (w_i + w_j)/2 - |cx_i - cx_j|
          Δy_ij = (h_i + h_j)/2 - |cy_i - cy_j|

    Uses a smooth approximation to max(0, x) for gradient flow.
    """

    def __init__(self, smooth: bool = True, eps: float = 1e-3):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def _soft_relu(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth approximation to max(0, x) = softplus(x)."""
        if self.smooth:
            return nn.functional.softplus(x, beta=10.0)
        return torch.clamp(x, min=0.0)

    def forward(self, positions: torch.Tensor,
                widths: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (N, 2) module center coordinates
            widths: (N,) module widths
            heights: (N,) module heights

        Returns:
            energy: scalar overlap energy
        """
        N = positions.size(0)
        energy = torch.tensor(0.0, device=positions.device)

        for i in range(N):
            xi, yi = positions[i, 0], positions[i, 1]
            wi, hi = widths[i], heights[i]

            for j in range(i + 1, N):
                xj, yj = positions[j, 0], positions[j, 1]
                wj, hj = widths[j], heights[j]

                # Half-size sum
                hw = (wi + wj) / 2.0
                hh = (hi + hj) / 2.0

                # Penetration depth
                dx = hw - torch.abs(xi - xj)
                dy = hh - torch.abs(yi - yj)

                overlap = self._soft_relu(dx) * self._soft_relu(dy)
                energy = energy + overlap

        return energy


class HPWLEnergy(nn.Module):
    """
    Differentiable HPWL energy using Log-Sum-Exp smooth bounding box.

    E_hpwl = Σ_e BB_e(positions)
    where BB_e = (smooth_max(x) - smooth_min(x)) + (smooth_max(y) - smooth_min(y))

    Uses Log-Sum-Exp (LSE) operator for smooth, fully-differentiable min/max.
    All modules in a net receive gradient proportional to their distance from
    the bounding-box extreme, avoiding the sparse-gradient problem of hard min/max.

    As temperature τ → 0, LSE converges to true max/min.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: LSE temperature τ.
                         Smaller τ → tighter approximation to hard min/max but
                         risk of numerical overflow. Larger τ → smoother gradients
                         but softer bounding box. Typical range: 0.1 ~ 5.0.
        """
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _smooth_max(x: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Log-Sum-Exp smooth maximum.

        smooth_max(x) = τ · log( Σ_i exp(x_i / τ) )

        Gradient distributes to ALL elements, weighted by exp(x_i/τ) / Σexp(x_j/τ).
        As τ → 0, gradient concentrates on argmax only.
        """
        return tau * torch.logsumexp(x / tau, dim=0)

    @staticmethod
    def _smooth_min(x: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Log-Sum-Exp smooth minimum.

        smooth_min(x) = -τ · log( Σ_i exp(-x_i / τ) )
        """
        return -tau * torch.logsumexp(-x / tau, dim=0)

    def forward(self, positions: torch.Tensor, nets: list) -> torch.Tensor:
        """
        Args:
            positions: (N, 2) module center coordinates
            nets: list of lists, each containing module indices

        Returns:
            energy: scalar HPWL energy (fully differentiable)
        """
        tau = self.temperature
        total = torch.tensor(0.0, device=positions.device)

        for net in nets:
            if len(net) == 0:
                continue
            net_pos = positions[torch.tensor(net, device=positions.device)]

            # Smooth bounding box via LSE — all modules get gradient
            x_min = self._smooth_min(net_pos[:, 0], tau)
            x_max = self._smooth_max(net_pos[:, 0], tau)
            y_min = self._smooth_min(net_pos[:, 1], tau)
            y_max = self._smooth_max(net_pos[:, 1], tau)

            total = total + (x_max - x_min) + (y_max - y_min)

        return total


class EnergyFunction(nn.Module):
    """
    Combined energy function for guided sampling.

    E(X) = λ_ovlp · E_ovlp + λ_hpwl · E_hpwl

    Provides gradient() method that returns ∇_X E(X).
    """

    def __init__(self,
                 lambda_overlap: float = 10.0,
                 lambda_hpwl: float = 0.1,
                 hpwl_temperature: float = 1.0):
        super().__init__()
        self.lambda_overlap = lambda_overlap
        self.lambda_hpwl = lambda_hpwl
        self.overlap_energy = OverlapEnergy(smooth=True)
        self.hpwl_energy = HPWLEnergy(temperature=hpwl_temperature)

    def forward(self,
                positions: torch.Tensor,
                module_sizes: torch.Tensor,
                nets: Optional[list] = None) -> torch.Tensor:
        """Compute total energy."""
        w = module_sizes[:, 0]
        h = module_sizes[:, 1]

        e_ovlp = self.overlap_energy(positions, w, h)

        if nets is not None:
            e_hpwl = self.hpwl_energy(positions, nets)
        else:
            e_hpwl = torch.tensor(0.0, device=positions.device)

        return self.lambda_overlap * e_ovlp + self.lambda_hpwl * e_hpwl

    def gradient(self,
                 positions: torch.Tensor,
                 module_sizes: torch.Tensor,
                 nets: Optional[list] = None) -> torch.Tensor:
        """
        Compute gradient of energy w.r.t. positions.

        Args:
            positions: (N, 2) requiring grad
            module_sizes: (N, 2)
            nets: list of nets

        Returns:
            grad: (N, 2) ∂E/∂X
        """
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)

        energy = self.forward(positions, module_sizes, nets)
        grad = torch.autograd.grad(energy, positions, create_graph=False)[0]
        return grad
