"""
Train Diffusion Turbulence Super-Resolution Model.

Pipeline:
  1. Load a pre-trained DeepONet (or DeepONetFourier) operator.
  2. Run operator forward passes on training data to produce mean-flow predictions.
  3. Train the DDPM diffusion model to generate turbulence fluctuations
     conditioned on the mean flow features.

Usage:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --epochs 200 --n-samples 5
    python scripts/train_diffusion.py --operator-ckpt results/models/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from src.deeponet.dataset import create_dataloaders
from src.deeponet.deeponet_fourier import DeepONetFourier
from src.generative.diffusion_turbulence_model import DiffusionTurbulenceModel


# ---------------------------------------------------------------------------
# Diffusion trainer
# ---------------------------------------------------------------------------

class DiffusionTrainer:
    """
    Two-stage training pipeline for the diffusion turbulence model.

    Stage 1 – Characterise noise: runs the pre-trained operator on the
              training split, collects (predictions, targets) pairs, and
              computes residuals  Δ = target − prediction.
    Stage 2 – DDPM training: train the denoising UNet to reconstruct Δ
              conditioned on the mean-flow features.

    Args:
        operator_ckpt:   Path to pre-trained operator checkpoint (.pth).
        config_path:     configs/config.yaml
        model_config_path: configs/model_config.yaml
        epochs:          Training epochs for the diffusion model.
        lr:              Learning rate.
        batch_size:      Mini-batch size.
    """

    def __init__(
        self,
        operator_ckpt:     Path,
        config_path:       Path,
        model_config_path: Path,
        epochs:            int   = 100,
        lr:                float = 1e-4,
        batch_size:        int   = 8,
    ) -> None:
        with open(config_path)       as f: self.config       = yaml.safe_load(f)
        with open(model_config_path) as f: self.model_config = yaml.safe_load(f)
        self.config.update(self.model_config)

        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Diffusion training on {self.device}")

        # Data loaders
        self.train_loader, self.val_loader, _ = create_dataloaders(
            config=self.config, batch_size=batch_size, num_workers=0
        )

        # Pre-trained operator (frozen)
        self.operator = self._load_operator(operator_ckpt)
        n_nodes       = self._count_nodes()

        # Diffusion model
        cfg = self.config.get("diffusion_turbulence", {})
        self.diff_model = DiffusionTurbulenceModel(
            n_nodes      = n_nodes,
            n_fields     = cfg.get("n_fields",     4),
            cond_dim     = cfg.get("cond_dim",     8),
            n_diff_steps = cfg.get("n_diff_steps", 1000),
        ).to(self.device)
        n_params = sum(p.numel() for p in self.diff_model.parameters()
                       if p.requires_grad)
        print(f"  Diffusion parameters: {n_params:,}")

        self.optimizer  = torch.optim.Adam(
            self.diff_model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        self.output_dir = project_root / "results" / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _load_operator(self, ckpt_path: Path) -> torch.nn.Module:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Operator checkpoint not found: {ckpt_path}\n"
                "Run train_deeponet.py first."
            )
        operator = DeepONetFourier.from_legacy_config(self.config).to(self.device)
        ckpt     = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        operator.load_state_dict(ckpt.get("model_state_dict", ckpt))
        operator.eval()
        for p in operator.parameters():
            p.requires_grad_(False)
        print(f"✓ Operator loaded from {ckpt_path}")
        return operator

    # ------------------------------------------------------------------
    def _count_nodes(self) -> int:
        for _, trunk, _ in self.train_loader:
            return trunk.shape[0] if trunk.dim() == 2 else trunk.shape[1]
        return 1000

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_residuals(self, branch, trunk, target):
        """Compute  Δ = target − operator_prediction."""
        branch = branch.to(self.device)
        trunk  = trunk.to(self.device)
        target = target.to(self.device)
        pred   = self.operator(branch, trunk)   # [B, n_fields, N]
        return target - pred, pred              # residual, mean-flow

    # ------------------------------------------------------------------
    def train_epoch(self) -> float:
        self.diff_model.train()
        total = 0.0
        for branch, trunk, target in self.train_loader:
            residuals, mean_flow = self._get_residuals(branch, trunk, target)
            # x0 = fluctuations relative to operator mean
            x0   = residuals                                      # [B, n_fields, N]
            tke  = mean_flow[:, 2, :]                             # turbulence_k channel
            cond = self.diff_model.encode_condition(mean_flow, tke)  # [B, cond_dim]

            self.optimizer.zero_grad()
            loss = self.diff_model.training_loss(x0, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diff_model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(self.train_loader)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def val_epoch(self) -> float:
        self.diff_model.eval()
        total = 0.0
        for branch, trunk, target in self.val_loader:
            residuals, mean_flow = self._get_residuals(branch, trunk, target)
            x0   = residuals
            tke  = mean_flow[:, 2, :]
            cond = self.diff_model.encode_condition(mean_flow, tke)
            total += self.diff_model.training_loss(x0, cond).item()
        return total / len(self.val_loader)

    # ------------------------------------------------------------------
    def fit(self) -> None:
        print(f"\n{'='*60}\nTraining Diffusion Turbulence Model\n{'='*60}")
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0         = time.perf_counter()
            train_loss = self.train_epoch()
            val_loss   = self.val_epoch()
            self.scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                self._save(epoch, val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{self.epochs}  "
                    f"train={train_loss:.6f}  val={val_loss:.6f}  "
                    f"({time.perf_counter()-t0:.1f}s)"
                )

        print(f"\nBest validation loss: {best_val:.6f}")
        print(f"Model saved to {self.output_dir / 'diffusion_model.pth'}")

    # ------------------------------------------------------------------
    def _save(self, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.diff_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            self.output_dir / "diffusion_model.pth",
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def demo_sample(self, n_samples: int = 5) -> None:
        """Generate sample turbulence realisations from the trained model."""
        print(f"\nGenerating {n_samples} turbulence realisations...")
        ckpt_path = self.output_dir / "diffusion_model.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.diff_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.diff_model.eval()

        for branch, trunk, target in self.train_loader:
            _, mean_flow = self._get_residuals(branch[:1], trunk, target[:1])
            tke          = mean_flow[:, 2, :]
            samples      = self.diff_model.sample(
                mean_flow, tke, n_samples=n_samples
            )
            print(f"  Samples shape: {samples.shape}  "
                  f"std={samples.std().item():.4f}")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train diffusion turbulence super-resolution model"
    )
    p.add_argument("--operator-ckpt", type=Path,
                   default=project_root / "results" / "models" / "best_model.pth",
                   dest="operator_ckpt")
    p.add_argument("--config",       type=Path,
                   default=project_root / "configs" / "config.yaml")
    p.add_argument("--model-config", type=Path,
                   default=project_root / "configs" / "model_config.yaml",
                   dest="model_config")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch-size",  type=int,   default=8, dest="batch_size")
    p.add_argument("--demo",        action="store_true",
                   help="Generate sample realisations after training")
    p.add_argument("--n-samples",   type=int,   default=5, dest="n_samples")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60 + "\nTrain Diffusion Turbulence Model\n" + "=" * 60)

    trainer = DiffusionTrainer(
        operator_ckpt     = args.operator_ckpt,
        config_path       = args.config,
        model_config_path = args.model_config,
        epochs            = args.epochs,
        lr                = args.lr,
        batch_size        = args.batch_size,
    )
    trainer.fit()

    if args.demo:
        trainer.demo_sample(n_samples=args.n_samples)
