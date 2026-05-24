"""
Diffusion Model for VLSI placement legalization and gap compression.

Implements:
  - Forward process: q(X_t | X_0) = N(√ᾶ_t X_0, (1-ᾶ_t)I)
  - Reverse process: DenoiserNet + energy-guided sampling

Energy-guided sampling combines:
  - Data-driven denoising (Edge-GNN prediction)
  - Physics-driven overlap repulsion (energy gradient)
  - Stochastic noise for exploration

Reference: VLSI.tex Section 3.2
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Tuple
from .denoiser import DenoiserNet
from .energy import EnergyFunction


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (from improved DDPM)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


def linear_beta_schedule(timesteps: int,
                         beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule (from original DDPM)."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionPlacer(nn.Module):
    """
    Diffusion-based placement legalization.

    Forward: adds Gaussian noise to coordinates
    Reverse: Edge-GNN denoiser + energy gradient guidance → legal placement

    Variance alignment: the MDP Phase 1 outputs X_K with grid quantization error
    of variance σ²_q = (cell_w² + cell_h²) / 12. We solve for the diffusion timestep
    K such that the forward-process cumulative noise variance (1 - ᾶ_K) matches
    this quantization variance, ensuring the reverse process starts from a
    correctly-matched distribution.
    """

    def __init__(self,
                 num_modules: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 # Energy guidance weights
                 lambda_overlap: float = 10.0,
                 lambda_hpwl: float = 0.1,
                 hpwl_temperature: float = 1.0,
                 guidance_scale: float = 1.0,  # η in the paper
                 # Canvas geometry (for clipping + variance calibration)
                 canvas_width: float = 1000.0,
                 canvas_height: float = 1000.0,
                 # Gradient safety
                 max_energy_grad: float = 50.0,  # tanh saturation ceiling
                 use_dynamic_weights: bool = True,
                 ):
        super().__init__()
        self.num_modules = num_modules
        self.timesteps = timesteps
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_energy_grad = max_energy_grad
        self.use_dynamic_weights = use_dynamic_weights

        # Denoiser network
        self.denoiser = DenoiserNet(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Energy function for guidance (weights will be set per-step)
        self.energy_fn = EnergyFunction(
            lambda_overlap=lambda_overlap,
            lambda_hpwl=lambda_hpwl,
            hpwl_temperature=hpwl_temperature,
        )
        self.guidance_scale = guidance_scale
        self.base_lambda_overlap = lambda_overlap
        self.base_lambda_hpwl = lambda_hpwl

        # Beta schedule
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

    def forward_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: sample X_t given X_0 and timestep t.

        q(X_t | X_0) = N(√ᾶ_t X_0, (1-ᾶ_t)I)
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def _get_dynamic_weights(self, t_idx: int, start_t: int) -> Tuple[float, float]:
        """
        Dynamic weight annealing for energy guidance.

        Early steps (high noise): λ_hpwl dominates → guide toward low-wirelength topology
        Late steps (low noise):  λ_overlap dominates → eliminate remaining overlaps

        Uses cosine schedule: overlap weight ramps from 0→1, HPWL weight ramps from 1→0.
        """
        if not self.use_dynamic_weights:
            return self.base_lambda_overlap, self.base_lambda_hpwl

        # Progress: 0.0 (start) → 1.0 (end)
        progress = 1.0 - (t_idx / max(start_t, 1))
        progress = max(0.0, min(1.0, progress))

        # Cosine annealing
        w_overlap_frac = 0.5 * (1.0 - np.cos(np.pi * progress))  # 0→1
        w_hpwl_frac = 0.5 * (1.0 + np.cos(np.pi * progress))     # 1→0

        lambda_ovlp = self.base_lambda_overlap * w_overlap_frac
        lambda_hpwl = self.base_lambda_hpwl * w_hpwl_frac

        return lambda_ovlp, lambda_hpwl

    @torch.no_grad()
    def sample(self,
               x_k: torch.Tensor,
               module_sizes: torch.Tensor,
               edge_index: torch.Tensor,
               edge_attr: torch.Tensor,
               nets: Optional[list] = None,
               start_t: Optional[int] = None,
               num_steps: Optional[int] = None,
               return_trajectory: bool = False) -> torch.Tensor:
        """
        Reverse diffusion sampling with clipped energy guidance
        and dynamic weight annealing.

        Starts from X_K (coarse MDP output with noise/overlap) and denoises
        to produce legalized placement X_0.

        Dynamic weights: λ_hpwl starts high (guide topology), λ_overlap
        ramps up late (eliminate remaining overlaps). Energy gradient is
        clipped to prevent "Big Bang" explosion from degenerate inputs.

        Args:
            x_k: (N, 2) initial placement (from MDP Phase 1)
            module_sizes: (N, 2) width, height
            edge_index: (2, E) netlist edges
            edge_attr: (E, 1) edge weights
            nets: optional list of nets for HPWL energy
            start_t: timestep to start reverse process
            num_steps: number of denoising steps
            return_trajectory: if True, return list of all intermediate X_t

        Returns:
            x_0: (N, 2) final legalized placement
        """
        device = x_k.device

        if start_t is None:
            start_t = self.timesteps // 2

        if num_steps is None:
            step_indices = list(range(start_t, 0, -1))
        else:
            step_indices = np.linspace(start_t, 1, num_steps).astype(int).tolist()
            step_indices = list(dict.fromkeys(step_indices))

        x_t = x_k.clone()
        trajectory = [x_t.clone()] if return_trajectory else None
        total_steps = len(step_indices)

        for i, t_idx in enumerate(step_indices):
            t = torch.tensor([t_idx], device=device, dtype=torch.long)
            alpha_t = self.alphas[t_idx]
            alpha_cumprod_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            # --- Dynamic weight annealing ---
            lambda_ovlp, lambda_hpwl = self._get_dynamic_weights(t_idx, start_t)
            self.energy_fn.lambda_overlap = lambda_ovlp
            self.energy_fn.lambda_hpwl = lambda_hpwl

            # Denoising step
            eps_pred = self.denoiser(x_t, t, module_sizes, edge_index, edge_attr)
            coef = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps_pred)

            # Energy guidance with gradient clipping
            if self.guidance_scale > 0 and nets is not None:
                x_for_energy = x_t.detach().requires_grad_(True)
                energy_grad = self.energy_fn.gradient(
                    x_for_energy, module_sizes, nets,
                )
                # --- GRADIENT CLIPPING: prevent "Big Bang" ---
                # Smooth tanh saturation — no dead zones, everywhere differentiable.
                # |grad| << max_norm → linear (no distortion)
                # |grad| >> max_norm → asymptotes to ±max_norm (graceful saturation)
                energy_grad = self.max_energy_grad * torch.tanh(
                    energy_grad / self.max_energy_grad
                )
                x_t = x_t - self.guidance_scale * energy_grad

            # Noise injection
            if t_idx > 1:
                sigma_t = torch.sqrt(beta_t)
                x_t = x_t + torch.randn_like(x_t) * sigma_t

            # Canvas bounds
            x_t[:, 0] = torch.clamp(x_t[:, 0], 0.0, self.canvas_width)
            x_t[:, 1] = torch.clamp(x_t[:, 1], 0.0, self.canvas_height)

            if return_trajectory:
                trajectory.append(x_t.clone())

        if return_trajectory:
            return trajectory
        return x_t

    def calibrate_start_t(self,
                           grid_cell_w: float,
                           grid_cell_h: float) -> int:
        """
        Find the diffusion timestep K where the forward-process noise variance
        matches the MDP grid quantization error variance.

        Grid quantization: each module's coordinate is quantized to a grid cell
        center, introducing error ε ~ Uniform(-cell/2, +cell/2) in each dimension.
        Variance of uniform quantization error:
            σ²_q = cell_w² / 12 + cell_h² / 12  (sum of independent x and y errors)

        We normalize by canvas area to get dimensionless variance, then find K
        such that (1 - ᾶ_K) ≈ σ²_normalized.

        Args:
            grid_cell_w: width of one grid cell = canvas_width / grid_size
            grid_cell_h: height of one grid cell = canvas_height / grid_size

        Returns:
            K: timestep index (0-indexed) for reverse process start
        """
        # Quantization error variance (uniform distribution in each dimension)
        var_q_x = (grid_cell_w ** 2) / 12.0
        var_q_y = (grid_cell_h ** 2) / 12.0

        # Normalize by canvas dimensions so variance is dimensionless
        # (the denoiser works in raw coordinate space, so we match raw variance)
        # Actually: the denoiser sees coordinates in [0, canvas] range.
        # The forward noise variance (1-ᾶ_t) is the variance of noise added
        # to coordinates in raw pixel units.  We need σ²_noise ≈ var_q_x + var_q_y.
        target_variance = var_q_x + var_q_y

        # Search for closest timestep
        one_minus_alpha_bar = (1.0 - self.alphas_cumprod).cpu().numpy()

        # Find K that minimizes |(1-ᾶ_K) - target_variance|
        diffs = np.abs(one_minus_alpha_bar - target_variance)
        K = int(np.argmin(diffs))

        # Clamp to valid range
        K = max(1, min(K, self.timesteps - 1))

        actual_variance = float(one_minus_alpha_bar[K])
        print(f"[Variance Alignment] Grid cell: {grid_cell_w:.2f}×{grid_cell_h:.2f}, "
              f"quantization var: {target_variance:.4f}, "
              f"matched K={K} (1-ᾶ_K={actual_variance:.4f})")

        return K

    def load_ema(self, ema_params: Optional[dict]) -> None:
        """Load EMA parameters for the denoiser if available."""
        if ema_params is None:
            return
        for name, param in self.denoiser.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name].to(param.data.device))

    def compute_loss(self,
                     x_0: torch.Tensor,
                     module_sizes: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_attr: torch.Tensor,
                     t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute diffusion training loss (MSE between predicted and true noise).

        Args:
            x_0: (N, 2) clean coordinates
            module_sizes: (N, 2) width, height
            edge_index, edge_attr: netlist graph
            t: optional timestep; if None, randomly sampled

        Returns:
            loss: scalar MSE loss
        """
        if t is None:
            t = torch.randint(0, self.timesteps, (1,), device=x_0.device)

        x_t, noise = self.forward_noise(x_0.unsqueeze(0), t)
        x_t = x_t.squeeze(0)
        noise = noise.squeeze(0)

        eps_pred = self.denoiser(x_t, t, module_sizes, edge_index, edge_attr)
        loss = nn.functional.mse_loss(eps_pred, noise)
        return loss
