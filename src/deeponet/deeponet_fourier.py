"""
Fourier-Enhanced DeepONet Architecture (deeponet_fourier).

Upgraded architecture combining:
  - Fourier Feature Encoding on trunk coordinates (mitigates spectral bias)
  - Adaptive GELU activations in the first two trunk layers
  - Reduced parameter count vs. the original architecture
  - GELU activations throughout

Branch network:  3 → 128 → 256 → 256  (output_dim = 256)
Trunk network:   FourierEncoding(3→512) → AdaptGELU → 256
                 → AdaptGELU → 256 → GELU → 256
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_encoding import FourierFeatureEncoding
from .adaptive_activation import AdaptiveActivationLayer


# ---------------------------------------------------------------------------
# Branch network
# ---------------------------------------------------------------------------

class BranchNetFourier(nn.Module):
    """
    Branch network for Fourier-enhanced DeepONet.

    Processes simulation parameters (velocity, break_size, temperature).

    Architecture:  input_dim → hidden_dims → output_dim
    Default:       3 → 128 → 256 → 256 (output)
    Activation:    GELU
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256]

        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:                 # not the last layer
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] → [B, output_dim]"""
        return self.network(x)


# ---------------------------------------------------------------------------
# Trunk network
# ---------------------------------------------------------------------------

class TrunkNetFourier(nn.Module):
    """
    Fourier-enhanced trunk network.

    Architecture (defaults):
        FourierEncoding(3 → 512)
        → AdaptiveGELU layer (512 → 256)
        → AdaptiveGELU layer (256 → 256)
        → Linear (256 → output_dim=256)
    """

    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 256,
        mapping_size: int = 256,
        scale: float = 10.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.fourier_enc = FourierFeatureEncoding(
            input_dim=coord_dim,
            mapping_size=mapping_size,
            scale=scale,
        )
        fourier_out = self.fourier_enc.output_dim  # 2 * mapping_size = 512

        # Adaptive activation layers (first two hidden layers)
        adapt_blocks: List[nn.Module] = []
        prev_dim = fourier_out
        for h_dim in hidden_dims:
            adapt_blocks.append(AdaptiveActivationLayer(prev_dim, h_dim))
            if dropout > 0:
                adapt_blocks.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.adapt_layers = nn.Sequential(*adapt_blocks)

        # Final linear projection (no activation — output of trunk network)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity="linear")
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, coord_dim] → [N, output_dim]"""
        h = self.fourier_enc(x)       # [N, 512]
        h = self.adapt_layers(h)      # [N, 256]
        return self.output_layer(h)   # [N, 256]


# ---------------------------------------------------------------------------
# DeepONetFourier — main model
# ---------------------------------------------------------------------------

class DeepONetFourier(nn.Module):
    """
    Fourier-Enhanced Deep Operator Network.

    Supports both the new ``model_config.yaml`` (``fourier_deeponet`` key)
    and the legacy ``config.yaml`` (``deeponet`` key) for backward
    compatibility.

    Args:
        config: Configuration dict loaded from YAML.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        # Resolve config section (new key preferred, legacy fallback)
        cfg = config.get("fourier_deeponet", config.get("deeponet", {}))

        b_cfg = cfg.get("branch_net", {})
        t_cfg = cfg.get("trunk_net", {})

        b_input  = b_cfg.get("input_dim",   3)
        b_hidden = b_cfg.get("hidden_dims", [128, 256])
        b_out    = b_cfg.get("output_dim",  256)

        t_coord  = t_cfg.get("coord_dim",    3)
        t_hidden = t_cfg.get("hidden_dims",  [256, 256])
        t_out    = t_cfg.get("output_dim",   256)
        t_map    = t_cfg.get("mapping_size", 256)
        t_scale  = t_cfg.get("fourier_scale", 10.0)

        self.n_outputs    = cfg.get("n_outputs", 4)
        self.output_fields = cfg.get(
            "output_fields",
            ["pressure", "velocity_magnitude", "turbulence_k", "temperature"],
        )

        assert b_out == t_out, (
            f"Branch output_dim ({b_out}) must equal trunk output_dim ({t_out}) "
            "for the inner-product to be valid."
        )

        self.branch_nets = nn.ModuleList([
            BranchNetFourier(b_input, b_hidden, b_out)
            for _ in range(self.n_outputs)
        ])
        self.trunk_nets = nn.ModuleList([
            TrunkNetFourier(t_coord, t_hidden, t_out, t_map, t_scale)
            for _ in range(self.n_outputs)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.n_outputs)
        ])

    # ------------------------------------------------------------------
    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            branch_input: [B, branch_input_dim]
            trunk_input:  [N, coord_dim]

        Returns:
            Tensor of shape [B, n_outputs, N].
        """
        outputs = []
        for i in range(self.n_outputs):
            b_out = self.branch_nets[i](branch_input)          # [B, 256]
            t_out = self.trunk_nets[i](trunk_input)            # [N, 256]
            out   = torch.matmul(b_out, t_out.T) + self.biases[i]  # [B, N]
            outputs.append(out)
        return torch.stack(outputs, dim=1)                     # [B, n_out, N]

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        branch = sum(
            p.numel() for net in self.branch_nets
            for p in net.parameters() if p.requires_grad
        )
        trunk = sum(
            p.numel() for net in self.trunk_nets
            for p in net.parameters() if p.requires_grad
        )
        return {
            "total":   self.count_parameters(),
            "branch":  branch,
            "trunk":   trunk,
            "biases":  self.n_outputs,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_config(cls, config: dict) -> "DeepONetFourier":
        """
        Build a DeepONetFourier from a legacy config.yaml dict.

        The original hidden dims are replaced with the reduced architecture
        specified in the upgrade requirements.
        """
        deeponet_cfg = config["deeponet"]
        wrapped = {
            "fourier_deeponet": {
                "branch_net": {
                    "input_dim":   deeponet_cfg["branch_net"]["input_dim"],
                    "hidden_dims": [128, 256],
                    "output_dim":  256,
                },
                "trunk_net": {
                    "coord_dim":     3,
                    "hidden_dims":   [256, 256],
                    "output_dim":    256,
                    "mapping_size":  256,
                    "fourier_scale": 10.0,
                },
                "n_outputs":     deeponet_cfg["n_outputs"],
                "output_fields": deeponet_cfg["output_fields"],
            }
        }
        return cls(wrapped)
