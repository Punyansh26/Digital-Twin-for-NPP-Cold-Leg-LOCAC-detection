"""
Model Factory — resolves a version string to a constructed nn.Module.

Usage:
    from src.core.model_factory import load_operator, operator_param_count

    model = load_operator("deeponet_fourier", config)
    print(operator_param_count(model))
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Build (no checkpoint)
# ---------------------------------------------------------------------------

def build_operator(version: str, config: dict) -> nn.Module:
    """
    Construct and return a fresh (untrained) operator.

    Args:
        version: One of the ModelVersion string values.
        config:  Merged config dict (config.yaml + model_config.yaml).

    Returns:
        nn.Module ready for training.

    Raises:
        ValueError: if version is not recognised.
    """
    if version == "deeponet_fourier":
        from src.deeponet.deeponet_fourier import DeepONetFourier
        return DeepONetFourier.from_legacy_config(config)

    elif version == "deeponet":
        from src.deeponet.model import DeepONet
        return DeepONet(config)

    elif version == "transolver":
        from src.operators.transolver_operator import TransolverOperator
        return TransolverOperator.from_config(config)

    elif version == "clifford":
        from src.operators.clifford_operator import CliffordNeuralOperator
        return CliffordNeuralOperator.from_config(config)

    else:
        raise ValueError(
            f"Unknown model version: {version!r}\n"
            f"Valid choices: deeponet | deeponet_fourier | transolver | clifford"
        )


# ---------------------------------------------------------------------------
# Load from checkpoint
# ---------------------------------------------------------------------------

def load_operator(
    version:         str,
    config:          dict,
    checkpoint_path: Path,
    device:          torch.device,
) -> nn.Module:
    """
    Construct an operator and load weights from a checkpoint file.

    The checkpoint may have been saved with version metadata (``model_version``
    key).  If present, the stored version is verified against *version*.

    Args:
        version:         Target architecture (e.g. ``"deeponet_fourier"``).
        config:          Merged config dict.
        checkpoint_path: Path to ``.pth`` checkpoint.
        device:          Target device.

    Returns:
        Loaded, eval-mode nn.Module on *device*.
    """
    model = build_operator(version, config)

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Version compatibility check
    saved_version = ckpt.get("model_version") or ckpt.get("operator")
    if saved_version and saved_version != version:
        print(
            f"  WARNING: checkpoint was saved with version={saved_version!r}, "
            f"but loading as {version!r}. Proceeding; weights may be incompatible."
        )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def operator_param_count(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    # Prefer the model's own method if available
    if hasattr(model, "count_parameters"):
        return model.count_parameters()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_version_banner(
    version:    str,
    device:     torch.device | str,
    dataset:    str = "AP1000 Cold-Leg CFD",
    n_params:   int | None = None,
) -> None:
    """
    Print the version/tier banner at pipeline startup.

    Example output::

        ===================================
         AP1000 LOCAC DIGITAL TWIN SYSTEM
        ===================================
        Selected Operator : deeponet_fourier
        Tier              : Tier 1 — Optimized DeepONet (Fourier + Sobolev + Divergence)
        CUDA device       : cuda (RTX 4060)
        Dataset           : AP1000 Cold-Leg CFD
        Parameters        : 1,451,012
    """
    from src.core.model_versions import get_tier_label

    device_str = str(device)
    if device_str == "cuda" and torch.cuda.is_available():
        gpu_name   = torch.cuda.get_device_name(0)
        device_str = f"cuda ({gpu_name})"

    print("\n" + "=" * 43)
    print("   AP1000 LOCAC DIGITAL TWIN SYSTEM   ")
    print("=" * 43)
    print(f"  Selected Operator : {version}")
    print(f"  Tier              : {get_tier_label(version)}")
    print(f"  Device            : {device_str}")
    print(f"  Dataset           : {dataset}")
    if n_params is not None:
        print(f"  Parameters        : {n_params:,}")
    print("=" * 43 + "\n")
