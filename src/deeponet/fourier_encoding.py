"""
Fourier Feature Encoding for DeepONet Trunk Network.

Mitigates spectral bias by mapping coordinates to a rich high-frequency
feature space before feeding into the trunk network.

    x ∈ R^d
    B ∈ R^{d × m}  (random frequency matrix)
    γ(x) = [sin(2π B x), cos(2π B x)]  ∈ R^{2m}

References:
    Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import math


class FourierFeatureEncoding(nn.Module):
    """
    Random Fourier Feature Encoding.

    Maps input coordinates x ∈ R^d to a 2m-dimensional feature vector via
    random sinusoidal projections, simultaneously capturing low and high
    spatial frequencies to overcome spectral bias.

    Args:
        input_dim:    Coordinate dimension d (default 3 for x,y,z).
        mapping_size: Number of random frequencies m (default 256).
                      Output dimension will be 2*m = 512.
        scale:        Standard deviation of the random frequency matrix B.
                      Higher scale → higher-frequency components (default 10).
        trainable:    If True, B is learned (default False — fixed random features).
    """

    def __init__(
        self,
        input_dim: int = 3,
        mapping_size: int = 256,
        scale: float = 10.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim    = input_dim
        self.mapping_size = mapping_size
        self.scale        = scale

        B_init = torch.randn(input_dim, mapping_size) * scale
        if trainable:
            self.B = nn.Parameter(B_init)
        else:
            self.register_buffer("B", B_init)

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Output dimension = 2 * mapping_size (sin + cos channels)."""
        return self.mapping_size * 2

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Coordinates tensor of shape [..., input_dim].

        Returns:
            Fourier features of shape [..., 2 * mapping_size].
        """
        # [..., d] × [d, m] → [..., m]
        x_proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"mapping_size={self.mapping_size}, "
            f"scale={self.scale}, "
            f"output_dim={self.output_dim})"
        )
