"""
Train alternative neural operators: Transolver++ or Clifford Neural Operator.

Usage:
    python scripts/train_operator.py --operator transolver
    python scripts/train_operator.py --operator clifford --epochs 300
    python scripts/train_operator.py --operator transolver --lr 5e-4 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.deeponet.dataset import create_dataloaders
from src.deeponet.train import MetricsCalculator, EarlyStopping


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_operator(operator: str, config: dict, model_config: dict) -> nn.Module:
    full_cfg = {**config, **model_config}
    if operator == "transolver":
        from src.operators.transolver_operator import TransolverOperator
        return TransolverOperator.from_config(full_cfg)
    elif operator == "clifford":
        from src.operators.clifford_operator import CliffordNeuralOperator
        return CliffordNeuralOperator.from_config(full_cfg)
    else:
        raise ValueError(f"Unknown operator: {operator!r}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OperatorTrainer:
    """
    Training loop for Transolver / Clifford operators.

    Supports:
      - Mixed-precision AMP
      - Cosine Annealing LR schedule
      - Early stopping
      - Extended per-field metrics
    """

    def __init__(
        self,
        operator:    str,
        config_path: Path,
        model_config_path: Path,
        lr:          float = 1e-3,
        epochs:      int   = 500,
        batch_size:  int   = 16,
    ) -> None:
        with open(config_path)       as f: self.config       = yaml.safe_load(f)
        with open(model_config_path) as f: self.model_config = yaml.safe_load(f)

        self.operator_name = operator
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.device        = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Training {operator} on {self.device}")

        # Data
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config=self.config, batch_size=batch_size, num_workers=0
        )
        print(f"  Train: {len(self.train_loader)} batches | "
              f"Val: {len(self.val_loader)} batches | "
              f"Test: {len(self.test_loader)} batches")

        # Model
        self.model = build_operator(
            operator, self.config, self.model_config
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        # Optimiser + scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=self.config.get("training", {}).get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.criterion = nn.MSELoss()

        # AMP
        self.use_amp = (self.device.type == "cuda")
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Helpers
        self.metrics       = MetricsCalculator()
        self.field_names   = self.config["deeponet"]["output_fields"]
        self.early_stop    = EarlyStopping(patience=50, min_delta=1e-6)
        self.best_val_loss = float("inf")
        self.output_dir    = project_root / "results" / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _run_batch(self, branch, trunk, target):
        branch = branch.to(self.device)
        trunk  = trunk.to(self.device)
        target = target.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred = self.model(branch, trunk)
            loss = self.criterion(pred, target)
        return loss, pred.detach()

    # ------------------------------------------------------------------
    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for branch, trunk, target in self.train_loader:
            self.optimizer.zero_grad()
            loss, _ = self._run_batch(branch, trunk, target)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
        return total / len(self.train_loader)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        self.model.eval()
        total = 0.0
        all_preds, all_targets = [], []
        for branch, trunk, target in self.val_loader:
            branch = branch.to(self.device)
            trunk  = trunk.to(self.device)
            target = target.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(branch, trunk)
            total += self.criterion(pred, target).item()
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = self.metrics.compute_all(preds, targets, self.field_names)
        return total / len(self.val_loader), metrics

    # ------------------------------------------------------------------
    def fit(self) -> None:
        print(f"\n{'='*60}\nTraining {self.operator_name}\n{'='*60}")
        for epoch in range(1, self.epochs + 1):
            t0         = time.perf_counter()
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            if epoch % 20 == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                print(
                    f"Epoch {epoch:4d}/{self.epochs}  "
                    f"train={train_loss:.6f}  val={val_loss:.6f}  "
                    f"RelL2={metrics.get('overall_rel_l2', 0.0):.4f}  "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                    f"({elapsed:.1f}s)"
                )

            if self.early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nBest validation loss: {self.best_val_loss:.6f}")
        self._test()

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / f"{self.operator_name}_best.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "operator": self.operator_name,
            },
            path,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _test(self) -> None:
        print(f"\n{'='*40}\nTest evaluation\n{'='*40}")
        ckpt_path = self.output_dir / f"{self.operator_name}_best.pth"
        if ckpt_path.exists():
            ckpt  = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
        self.model.eval()

        all_preds, all_targets = [], []
        for branch, trunk, target in self.test_loader:
            branch = branch.to(self.device)
            trunk  = trunk.to(self.device)
            pred   = self.model(branch, trunk)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = self.metrics.compute_all(preds, targets, self.field_names)
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Transolver++ or Clifford Neural Operator"
    )
    p.add_argument("--operator",    required=True,
                   choices=["transolver", "clifford"])
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int,   default=16, dest="batch_size")
    p.add_argument("--config",      type=Path,
                   default=project_root / "configs" / "config.yaml")
    p.add_argument("--model-config", type=Path,
                   default=project_root / "configs" / "model_config.yaml",
                   dest="model_config")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"Train Neural Operator: {args.operator.upper()}")
    print("=" * 60)

    trainer = OperatorTrainer(
        operator          = args.operator,
        config_path       = args.config,
        model_config_path = args.model_config,
        lr                = args.lr,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
    )
    trainer.fit()
