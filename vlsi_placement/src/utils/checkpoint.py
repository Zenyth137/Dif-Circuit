"""Load training checkpoints (PyTorch 2.6+ compatible)."""

from typing import Any, Dict, Optional
import torch

# Passed to DiffusionPlacer.__init__ (not inference-only keys)
DIFFUSION_MODEL_KEYS = frozenset({
    "hidden_dim",
    "num_layers",
    "timesteps",
    "beta_schedule",
    "lambda_overlap",
    "lambda_hpwl",
    "hpwl_temperature",
    "guidance_scale",
    "canvas_width",
    "canvas_height",
})


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load a trusted local checkpoint (weights + training metadata)."""
    return torch.load(path, map_location=device, weights_only=False)


def infer_diffusion_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Recover DiffusionPlacer kwargs from a saved state_dict."""
    timesteps = int(state_dict["betas"].numel())

    layer_ids = set()
    for key in state_dict:
        if key.startswith("denoiser.convs."):
            layer_ids.add(int(key.split(".")[2]))
    num_layers = len(layer_ids)

    hidden_dim = int(state_dict["denoiser.node_proj.0.weight"].shape[0])

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "timesteps": timesteps,
        "beta_schedule": "cosine",
        "lambda_overlap": 10.0,
        "lambda_hpwl": 0.1,
        "hpwl_temperature": 1.0,
        "guidance_scale": 1.0,
        "canvas_width": 1000.0,
        "canvas_height": 1000.0,
    }


def resolve_diffusion_model_config(
    checkpoint: Dict[str, Any],
    yaml_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
  Build DiffusionPlacer kwargs: prefer config stored at train time,
  else infer architecture from weights.
    """
    yaml_defaults = yaml_defaults or {}

    if "diffusion_model_config" in checkpoint:
        cfg = dict(checkpoint["diffusion_model_config"])
        source = "checkpoint metadata"
    else:
        cfg = infer_diffusion_model_config(checkpoint["diffusion_state_dict"])
        source = "inferred from state_dict"

    for key, value in yaml_defaults.items():
        if key in DIFFUSION_MODEL_KEYS and key not in cfg:
            cfg[key] = value

    cfg = {k: cfg[k] for k in DIFFUSION_MODEL_KEYS if k in cfg}
    print(f"Diffusion architecture ({source}): "
          f"hidden_dim={cfg['hidden_dim']}, num_layers={cfg['num_layers']}, "
          f"timesteps={cfg['timesteps']}")
    return cfg
