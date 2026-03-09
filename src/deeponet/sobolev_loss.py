"""
Sobolev Gradient-Enhanced Loss for DeepONet Training.

Penalises both function values and their spatial derivatives, encouraging
physically smooth and consistent field predictions.

    L = α · ||u_pred − u_true||²
      + β · ||∇u_pred − ∇u_true||²

Gradients are computed via:
  1. PyTorch autograd    (when trunk coordinates are provided with requires_grad=True)
  2. Finite differences  (fallback or when autograd is not applicable)

References:
    Son et al., "Sobolev Training for Physics-Informed Neural Networks",
    arXiv 2021.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobolevLoss(nn.Module):
    """
    Sobolev gradient-enhanced loss.

    Args:
        alpha:        Weight for MSE term (default 1.0).
        beta:         Weight for gradient term (default 0.1).
        use_autograd: Prefer autograd when coordinates are available
                      (default True).  Falls back to finite differences
                      silently when autograd is not feasible.
        fd_eps:       Not used directly; retained for API compatibility.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        use_autograd: bool = True,
        fd_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.alpha        = alpha
        self.beta         = beta
        self.use_autograd = use_autograd
        self.fd_eps       = fd_eps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fd_gradient(u: torch.Tensor) -> torch.Tensor:
        """
        Central finite-difference gradient along the spatial (last) axis.

        u: [B, n_fields, N] → grad: [B, n_fields, N]
        """
        u_pad = F.pad(u, (1, 1), mode="replicate")
        return (u_pad[..., 2:] - u_pad[..., :-2]) / 2.0

    @staticmethod
    def _autograd_gradient(
        u: torch.Tensor,
        coords: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Autograd gradient: d(sum(u)) / d(coords).

        Returns None when the computation graph is not available.
        """
        try:
            (grad,) = torch.autograd.grad(
                u.sum(),
                coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )
            return grad
        except (RuntimeError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        target_grad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Sobolev loss.

        Args:
            pred:        Predicted fields   [B, n_fields, N].
            target:      Ground-truth fields [B, n_fields, N].
            coords:      Trunk coordinates  [N, 3] (must have requires_grad=True
                         for the autograd path).
            target_grad: Pre-computed target gradients [N, 3] (optional).

        Returns:
            (total_loss, components_dict) where components_dict has
            keys ``mse_loss``, ``grad_loss``, ``total_loss``.
        """
        mse_loss  = F.mse_loss(pred, target)
        grad_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        if self.beta > 0.0:
            grad_computed = False

            # --- attempt autograd ---
            if (
                self.use_autograd
                and coords is not None
                and coords.requires_grad
            ):
                ag = self._autograd_gradient(pred, coords)
                if ag is not None:
                    if target_grad is not None:
                        grad_loss = F.mse_loss(ag, target_grad)
                    else:
                        tg = self._autograd_gradient(target.detach(), coords)
                        if tg is not None:
                            grad_loss = F.mse_loss(ag, tg.detach())
                        else:
                            grad_loss = F.mse_loss(
                                self._fd_gradient(pred),
                                self._fd_gradient(target),
                            )
                    grad_computed = True

            # --- fallback: finite differences ---
            if not grad_computed:
                grad_loss = F.mse_loss(
                    self._fd_gradient(pred),
                    self._fd_gradient(target),
                )

        total_loss = self.alpha * mse_loss + self.beta * grad_loss

        components: Dict[str, float] = {
            "mse_loss":   mse_loss.item(),
            "grad_loss":  grad_loss.item() if torch.is_tensor(grad_loss) else float(grad_loss),
            "total_loss": total_loss.item(),
        }
        return total_loss, components
