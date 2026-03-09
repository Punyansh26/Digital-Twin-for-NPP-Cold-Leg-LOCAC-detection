"""
Multi-Fidelity Residual Learning for DeepONet.

Architecture:
    u_base     ← operator trained on low-fidelity (RANS) data
    u_residual ← operator trained on (HF − u_base) residuals
    u_final    = u_base + u_residual

Training procedure:
    Stage 1 — Train base operator on RANS CFD data.
    Stage 2 — Freeze base, train residual operator on (LES − u_base).
    Stage 3 (optional) — Joint fine-tuning with unfrozen base.

This module is optional and is activated via ``use_residual_multifidelity``
in model_config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class MultiFidelityDeepONet(nn.Module):
    """
    Multi-fidelity residual neural operator.

    Combines a base operator (low-fidelity) with a correction operator
    (residual) to achieve high-fidelity predictions.

    Args:
        base_model:     Base operator (e.g. DeepONetFourier trained on RANS).
        residual_model: Correction operator trained on (HF − LF) difference.
        freeze_base:    If True, freeze base parameters at construction time.
    """

    def __init__(
        self,
        base_model: nn.Module,
        residual_model: nn.Module,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.base_model     = base_model
        self.residual_model = residual_model

        if freeze_base:
            self.freeze_base_network()

    # ------------------------------------------------------------------
    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            branch_input:      [B, branch_dim]
            trunk_input:       [N, trunk_dim or coord_dim]
            return_components: If True return (u_final, u_base, u_residual).

        Returns:
            u_final [B, n_outputs, N]  — or tuple if return_components=True.
        """
        u_base     = self.base_model(branch_input, trunk_input)
        u_residual = self.residual_model(branch_input, trunk_input)
        u_final    = u_base + u_residual

        if return_components:
            return u_final, u_base, u_residual
        return u_final

    # ------------------------------------------------------------------
    def load_base_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> None:
        """Load pre-trained base model weights from a checkpoint file."""
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.base_model.load_state_dict(state)
        print(f"✓ Loaded base model from {checkpoint_path}")

    def freeze_base_network(self) -> None:
        """Freeze base network parameters."""
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        print("✓ Base network frozen (residual training mode)")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for joint fine-tuning."""
        for param in self.parameters():
            param.requires_grad_(True)
        print("✓ All parameters unfrozen (joint fine-tuning mode)")

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "total":    _count(self),
            "base":     _count(self.base_model),
            "residual": _count(self.residual_model),
        }


# ---------------------------------------------------------------------------
# Two-stage training helper
# ---------------------------------------------------------------------------

class MultiFidelityTrainer:
    """
    Two-stage training procedure for multi-fidelity DeepONet.

    Stage 1: ``train_base_epoch``     — updates only base_model.
    Stage 2: ``train_residual_epoch`` — freezes base, updates residual_model
                                        on (target − base_prediction).
    """

    def __init__(
        self,
        mf_model: MultiFidelityDeepONet,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model     = mf_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

    # ------------------------------------------------------------------
    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Supports both plain loss modules and (loss, dict) returning ones."""
        result = self.criterion(pred, target)
        return result[0] if isinstance(result, tuple) else result

    # ------------------------------------------------------------------
    def train_base_epoch(self, dataloader) -> float:
        """Train only the base model."""
        self.model.base_model.train()
        self.model.residual_model.eval()
        total = 0.0
        for branch, trunk, target in dataloader:
            branch  = branch.to(self.device)
            trunk   = trunk.to(self.device)
            target  = target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model.base_model(branch, trunk)
            loss = self._compute_loss(pred, target)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / max(len(dataloader), 1)

    # ------------------------------------------------------------------
    def train_residual_epoch(self, dataloader) -> float:
        """Train the residual model on (target − base_prediction)."""
        self.model.base_model.eval()
        self.model.residual_model.train()
        total = 0.0
        for branch, trunk, target in dataloader:
            branch = branch.to(self.device)
            trunk  = trunk.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                base_pred       = self.model.base_model(branch, trunk)
            residual_target = target - base_pred
            self.optimizer.zero_grad()
            residual_pred = self.model.residual_model(branch, trunk)
            loss = self._compute_loss(residual_pred, residual_target)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / max(len(dataloader), 1)
