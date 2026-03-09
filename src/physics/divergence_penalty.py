"""
Physics-based Divergence Penalty for velocity field regularization.

For incompressible flow, the continuity equation requires:
    ∇ · u = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z = 0

Penalty:
    L_div = λ · || ∇ · u ||²

When only velocity magnitude (scalar) is predicted, a proxy divergence
based on spatial variation of the magnitude is computed via finite
differences.  When full vector components are available, use the
``compute_full_divergence_penalty`` method.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DivergencePenalty(nn.Module):
    """
    Divergence penalty for incompressible flow regularization.

    Args:
        weight:       Penalty weight λ (default 0.01).
        use_autograd: If True, attempt autograd-based divergence when
                      coordinates with requires_grad=True are supplied.
                      Otherwise (and as fallback) uses finite differences.
    """

    def __init__(
        self,
        weight: float = 0.01,
        use_autograd: bool = False,
    ) -> None:
        super().__init__()
        self.weight       = weight
        self.use_autograd = use_autograd

    # ------------------------------------------------------------------
    # Finite-difference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _central_diff(field: torch.Tensor) -> torch.Tensor:
        """
        Central finite differences along the spatial (last) axis.

        field: [B, N]  →  dfield: [B, N]
        """
        f_pad = F.pad(field, (1, 1), mode="replicate")
        return (f_pad[..., 2:] - f_pad[..., :-2]) / 2.0

    def compute_divergence_fd(
        self,
        vx: torch.Tensor,
        vy: torch.Tensor,
        vz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate divergence from 1-D sorted velocity components.

        Args:
            vx, vy, vz: [B, N] velocity components.

        Returns:
            divergence: [B, N]
        """
        div = self._central_diff(vx) + self._central_diff(vy)
        if vz is not None:
            div = div + self._central_diff(vz)
        return div

    # ------------------------------------------------------------------
    # Autograd path
    # ------------------------------------------------------------------

    @staticmethod
    def _autograd_divergence(
        velocity: torch.Tensor,
        coords: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Autograd divergence: ∂vx/∂x + ∂vy/∂y + ∂vz/∂z.

        Args:
            velocity: [B, 3, N] — vector velocity field.
            coords:   [N, 3]    — must have requires_grad=True.

        Returns:
            divergence norm scalar or None on failure.
        """
        try:
            def _grad_component(comp_idx: int) -> torch.Tensor:
                return torch.autograd.grad(
                    velocity[:, comp_idx, :].sum(),
                    coords,
                    create_graph=True,
                    retain_graph=True,
                )[0]  # [N, 3]

            dvx = _grad_component(0)[:, 0]   # ∂vx/∂x  [N]
            dvy = _grad_component(1)[:, 1]   # ∂vy/∂y  [N]
            dvz = _grad_component(2)[:, 2]   # ∂vz/∂z  [N]
            return dvx + dvy + dvz           # [N]
        except RuntimeError:
            return None

    # ------------------------------------------------------------------
    # Primary forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute divergence penalty from field predictions.

        When predictions follow the standard AP1000 field ordering
        [pressure, velocity_magnitude, turbulence_k, temperature], a proxy
        divergence based on the spatial variation of the velocity magnitude
        is used.

        Args:
            predictions: [B, n_fields, N] — predicted fields.
            coords:      [N, 3] — spatial coordinates (optional; for future
                         vector-field extension).

        Returns:
            Weighted penalty scalar.
        """
        # Velocity magnitude field (index 1)
        vel_mag = predictions[:, 1, :]           # [B, N]
        div_proxy = self._central_diff(vel_mag)  # [B, N]
        return self.weight * torch.mean(div_proxy ** 2)

    # ------------------------------------------------------------------
    def compute_full_divergence_penalty(
        self,
        vx: torch.Tensor,
        vy: torch.Tensor,
        vz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Divergence penalty from explicit velocity vector components.

        Args:
            vx, vy, vz: [B, N] velocity components.

        Returns:
            Weighted penalty scalar.
        """
        div = self.compute_divergence_fd(vx, vy, vz)
        return self.weight * torch.mean(div ** 2)
