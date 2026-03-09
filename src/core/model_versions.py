"""
Model Version Enum — central registry of all supported architectures.

Usage:
    from src.core.model_versions import ModelVersion, get_tier_label

    version = ModelVersion("deeponet_fourier")
    print(get_tier_label(version))   # "Tier 1 — Optimized DeepONet (Fourier + Sobolev + Divergence)"
"""

from __future__ import annotations

from enum import Enum


class ModelVersion(str, Enum):
    """Supported neural operator architectures."""

    # Tier 1 — upgraded Fourier DeepONet (default)
    TIER1_DEEPONET = "deeponet_fourier"

    # Tier 2 — alternative neural operators
    TIER2_TRANSOLVER = "transolver"
    TIER2_CLIFFORD   = "clifford"

    # Tier 3 — temporal / probabilistic extensions
    TIER3_MAMBA     = "mamba"
    TIER3_DIFFUSION = "diffusion"

    # Legacy baseline
    LEGACY_DEEPONET = "deeponet"

    @classmethod
    def choices(cls) -> list[str]:
        """Return all string values — suitable for argparse choices."""
        return [v.value for v in cls]

    @classmethod
    def training_choices(cls) -> list[str]:
        """Versions supported by train_deeponet.py (Tier 1 + legacy)."""
        return [cls.TIER1_DEEPONET.value, cls.LEGACY_DEEPONET.value]

    @classmethod
    def operator_choices(cls) -> list[str]:
        """Versions supported by train_operator.py (Tier 2)."""
        return [cls.TIER2_TRANSOLVER.value, cls.TIER2_CLIFFORD.value]

    @classmethod
    def inference_choices(cls) -> list[str]:
        """Versions supported by run_inference.py."""
        return [
            cls.LEGACY_DEEPONET.value,
            cls.TIER1_DEEPONET.value,
            cls.TIER2_TRANSOLVER.value,
            cls.TIER2_CLIFFORD.value,
        ]

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Tier metadata
# ---------------------------------------------------------------------------

TIER_LABELS: dict[str, str] = {
    "deeponet_fourier": "Tier 1 — Optimized DeepONet (Fourier + Sobolev + Divergence)",
    "transolver":       "Tier 2 — Transformer Neural Operator (Transolver++)",
    "clifford":         "Tier 2 — Clifford Neural Operator (Geometric Algebra)",
    "mamba":            "Tier 3 — Temporal Digital Twin (Mamba SSM)",
    "diffusion":        "Tier 3 — Probabilistic Turbulence (Diffusion DDPM)",
    "deeponet":         "Legacy  — Baseline DeepONet",
}


def get_tier_label(version: str | ModelVersion) -> str:
    """Return the human-readable tier description for a model version."""
    key = str(version)
    return TIER_LABELS.get(key, f"Unknown version: {key!r}")
