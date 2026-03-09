"""
Clifford Neural Operator for AP1000 Digital Twin.

Uses Geometric Algebra Cl(3,0) to represent and process flow fields in a
rotationally-equivariant manner.

Multivector basis (8 components):
    grade-0 (scalar):    1         index 0
    grade-1 (vectors):   e1, e2, e3        indices 1-3
    grade-2 (bivectors): e12, e13, e23     indices 4-6
    grade-3 (trivector): e123              index 7

Velocity vector:   v = vx·e1 + vy·e2 + vz·e3
Pressure scalar:   p = p·1

Clifford convolution layers maintain rotational equivariance of vector
quantities through the geometric product.

References:
    Brandstetter et al., "Clifford Neural Layers for PDE Modeling",
    ICLR 2023.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cl(3,0) geometric product
# ---------------------------------------------------------------------------

def clifford_product_3d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Geometric product of two multivectors in Cl(3,0).

    Metric signature (+,+,+): e_i² = +1 for i ∈ {1,2,3}.

    Args:
        a, b: [..., 8]  multivector tensors.

    Returns:
        [..., 8]  product multivector.
    """
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    a4, a5, a6, a7 = a[..., 4], a[..., 5], a[..., 6], a[..., 7]

    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    b4, b5, b6, b7 = b[..., 4], b[..., 5], b[..., 6], b[..., 7]

    # Grade-0 (scalar)
    c0 = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
    # Grade-1 (e1, e2, e3)
    c1 =  a0*b1 + a1*b0 - a2*b4 + a3*b5 + a4*b2 - a5*b3 - a6*b7 - a7*b6
    c2 =  a0*b2 + a1*b4 + a2*b0 - a3*b6 - a4*b1 + a5*b7 + a6*b3 - a7*b5
    c3 =  a0*b3 - a1*b5 + a2*b6 + a3*b0 - a4*b7 - a5*b1 - a6*b2 + a7*b4
    # Grade-2 (e12, e13, e23)
    c4 =  a0*b4 + a1*b2 - a2*b1 + a3*b7 + a4*b0 - a5*b6 + a6*b5 - a7*b3
    c5 =  a0*b5 - a1*b3 + a2*b7 + a3*b1 + a4*b6 + a5*b0 - a6*b4 - a7*b2
    c6 =  a0*b6 - a1*b7 - a2*b3 + a3*b2 - a4*b5 + a5*b4 + a6*b0 + a7*b1
    # Grade-3 (e123)
    c7 =  a0*b7 + a1*b6 - a2*b5 + a3*b4 + a4*b3 - a5*b2 + a6*b1 + a7*b0

    return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1)


# ---------------------------------------------------------------------------
# Clifford linear layer
# ---------------------------------------------------------------------------

class CliffordLinear(nn.Module):
    """
    Clifford algebra linear (fully-connected) layer.

    Computes W ⊗ x where ⊗ is the geometric product and W is a learned
    set of multivector weights.

    Args:
        in_channels:  Number of input multivector channels.
        out_channels: Number of output multivector channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Each weight is a multivector: [out, in, 8]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 8) * (in_channels ** -0.5)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_channels, 8]

        Returns:
            [..., out_channels, 8]
        """
        *batch, C_in, _ = x.shape
        C_out = self.weight.shape[0]

        # Vectorised geometric product via broadcasting:
        # x:      [..., 1, C_in, 8]
        # weight: [C_out, C_in, 8]  → broadcasted to [..., C_out, C_in, 8]
        x_exp = x.unsqueeze(-3)                                  # [..., 1, Cin, 8]
        w_exp = self.weight.unsqueeze(0)                         # [1, Cout, Cin, 8]

        prod  = clifford_product_3d(w_exp, x_exp)               # [..., Cout, Cin, 8]
        out   = prod.sum(dim=-2)                                 # [..., Cout, 8]
        return out + self.bias


class CliffordLayerNorm(nn.Module):
    """Layer normalisation applied over (channels × 8) multivector axes."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm([n_channels, 8])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ---------------------------------------------------------------------------
# Input / output projections
# ---------------------------------------------------------------------------

class PhysicsToMultivector(nn.Module):
    """
    Embed scalar + vector physics inputs as multivectors.

    Pressure  (scalar) → grade-0 component.
    Velocity  (vector) → grade-1 components (e1, e2, e3).
    """

    def __init__(
        self, scalar_dim: int, vector_dim: int, out_channels: int
    ) -> None:
        super().__init__()
        self.scalar_proj = nn.Linear(scalar_dim, out_channels)
        self.vector_proj = nn.Linear(vector_dim, out_channels * 3)
        self.out_channels = out_channels

    def forward(
        self,
        scalar: torch.Tensor,  # [..., scalar_dim]
        vector: torch.Tensor,  # [..., vector_dim]
    ) -> torch.Tensor:
        """Returns [..., out_channels, 8]"""
        *batch, _ = scalar.shape
        C = self.out_channels
        mv = torch.zeros(*batch, C, 8, device=scalar.device, dtype=scalar.dtype)
        mv[..., 0]   = self.scalar_proj(scalar)
        vec = self.vector_proj(vector).reshape(*batch, C, 3)
        mv[..., 1:4] = vec
        return mv


class MultivectorToFields(nn.Module):
    """Project multivector representation to scalar output fields."""

    def __init__(self, n_channels: int, n_outputs: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_channels * 8, n_outputs)

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """mv: [..., n_channels, 8]  →  [..., n_outputs]"""
        *batch, C, _ = mv.shape
        return self.proj(mv.reshape(*batch, C * 8))


# ---------------------------------------------------------------------------
# CliffordNeuralOperator — main model
# ---------------------------------------------------------------------------

class CliffordNeuralOperator(nn.Module):
    """
    Clifford Neural Operator with rotational equivariance.

    Args:
        coord_dim:   Spatial coordinate dimension (default 3).
        branch_dim:  Branch parameter dimension (default 3).
        n_channels:  Multivector channels per Clifford layer (default 32).
        n_layers:    Number of Clifford layers (default 4).
        n_outputs:   Number of output flow fields (default 4).
    """

    def __init__(
        self,
        coord_dim: int = 3,
        branch_dim: int = 3,
        n_channels: int = 32,
        n_layers: int = 4,
        n_outputs: int = 4,
    ) -> None:
        super().__init__()
        self.n_outputs  = n_outputs
        self.n_channels = n_channels

        self.embed = PhysicsToMultivector(branch_dim, coord_dim, n_channels)

        self.clifford_layers = nn.ModuleList([
            CliffordLinear(n_channels, n_channels) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            CliffordLayerNorm(n_channels) for _ in range(n_layers)
        ])

        self.readout = MultivectorToFields(n_channels, n_outputs)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "CliffordNeuralOperator":
        cfg = config.get("clifford", {})
        return cls(
            coord_dim  = cfg.get("coord_dim",  3),
            branch_dim = cfg.get("branch_dim", 3),
            n_channels = cfg.get("n_channels", 32),
            n_layers   = cfg.get("n_layers",   4),
            n_outputs  = cfg.get("n_outputs",  4),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        branch_input: torch.Tensor,  # [B, branch_dim]
        mesh_points:  torch.Tensor,  # [N, coord_dim]
    ) -> torch.Tensor:
        """Returns [B, n_outputs, N]"""
        B = branch_input.shape[0]
        N = mesh_points.shape[0]

        # Expand to [B, N] inputs
        branch_exp = branch_input.unsqueeze(1).expand(B, N, -1)   # [B, N, branch_dim]
        mesh_exp   = mesh_points.unsqueeze(0).expand(B, N, -1)    # [B, N, coord_dim]

        h = self.embed(branch_exp, mesh_exp)   # [B, N, C, 8]

        for layer, norm in zip(self.clifford_layers, self.norms):
            h_res = h
            h     = layer(h)                   # [B, N, C, 8]
            h     = norm(h)
            # Scalar-gate residual: activate only grade-0
            h     = h + h_res
            h[..., 0] = F.gelu(h[..., 0])

        out = self.readout(h)                  # [B, N, n_outputs]
        return out.permute(0, 2, 1)            # [B, n_outputs, N]

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
