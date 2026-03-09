"""
Train DeepONet / DeepONetFourier model.

Upgraded training pipeline with:
  - Sobolev gradient-enhanced loss
  - Divergence penalty regularization
  - Mixed-precision (AMP) training
  - Extended metrics (RelL2, R², MAE, Derivative L2)
  - Configurable operator selection

Usage:
    python scripts/train_deeponet.py
    python scripts/train_deeponet.py --operator deeponet          # legacy arch
    python scripts/train_deeponet.py --operator deeponet_fourier  # upgraded
    python scripts/train_deeponet.py --epochs 1000 --lr 5e-4
    python scripts/train_deeponet.py --no-sobolev --no-divergence # MSE only
    python scripts/train_deeponet.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.deeponet.dataset import create_dataloaders
from src.deeponet.model import DeepONet, DeepONetLoss
from src.deeponet.deeponet_fourier import DeepONetFourier
from src.deeponet.sobolev_loss import SobolevLoss
from src.deeponet.train import MetricsCalculator, EarlyStopping
from src.physics.divergence_penalty import DivergencePenalty
from src.core.model_factory import print_version_banner, operator_param_count
from src.core.model_versions import ModelVersion


# ---------------------------------------------------------------------------
# Extended metrics
# ---------------------------------------------------------------------------

def compute_extended_metrics(pred: torch.Tensor, target: torch.Tensor,
                              field_names: list) -> dict:
    """Compute RelL2, R², MAE, and spatial Derivative-L2 per field."""
    metrics = {}
    for i, field in enumerate(field_names):
        pf = pred[:, i, :].float()
        tf = target[:, i, :].float()
        metrics[f"{field}_rel_l2"]   = MetricsCalculator.relative_l2_error(pf, tf)
        metrics[f"{field}_r2"]       = MetricsCalculator.r2_score(pf, tf)
        metrics[f"{field}_mae"]      = MetricsCalculator.mae(pf, tf)
        dp = pf[:, 1:] - pf[:, :-1]
        dt = tf[:, 1:] - tf[:, :-1]
        metrics[f"{field}_deriv_l2"] = ((dp - dt).norm() / (dt.norm() + 1e-10)).item()
    return metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class UpgradedDeepONetTrainer:
    def __init__(
        self,
        config_path:       Path,
        model_config_path: Path,
        operator:          str   = "deeponet_fourier",
        sobolev_weight:    float = 0.1,
        divergence_weight: float = 0.01,
        use_sobolev:       bool  = True,
        use_divergence:    bool  = True,
    ) -> None:
        with open(config_path)       as f: self.config       = yaml.safe_load(f)
        with open(model_config_path) as f: self.model_config = yaml.safe_load(f)
        self.config.update(self.model_config)

        self.operator_name  = operator
        self.use_sobolev    = use_sobolev
        self.use_divergence = use_divergence

        self.output_dir = project_root / self.config["output_paths"]["models"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
               and self.config.get("device", {}).get("use_cuda", True)
            else "cpu"
        )

        self.model = self._build_model()
        print_version_banner(
            version  = self.operator_name,
            device   = self.device,
            n_params = operator_param_count(self.model),
        )

        n_out = self.config["deeponet"]["n_outputs"]
        self.mse_criterion = DeepONetLoss(weights=[1.0] * n_out)
        self.sobolev_loss  = SobolevLoss(alpha=1.0, beta=sobolev_weight,
                                         use_autograd=False)
        self.div_penalty   = DivergencePenalty(weight=divergence_weight)

        train_cfg = self.config["training"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        sched = train_cfg["scheduler"]
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min",
            patience=sched["patience"], factor=sched["factor"],
            min_lr=sched.get("min_lr", 1e-6),
        )
        es = train_cfg["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=es["patience"], min_delta=es["min_delta"]
        )

        use_amp = (train_cfg.get("mixed_precision", True)
                   and self.device.type == "cuda")
        self.scaler    = GradScaler("cuda") if use_amp else None
        self.grad_clip = float(train_cfg.get("gradient_clip", 1.0))

        self.field_names = self.config["deeponet"]["output_fields"]
        self.history: dict = {
            "train_loss": [], "val_loss": [], "learning_rate": []
        }

    def _build_model(self) -> nn.Module:
        if self.operator_name == "deeponet_fourier":
            return DeepONetFourier.from_legacy_config(self.config).to(self.device)
        elif self.operator_name == "deeponet":
            return DeepONet(self.config).to(self.device)
        raise ValueError(f"Unknown operator: {self.operator_name!r}")

    def _loss(self, output, target):
        mse   = self.mse_criterion(output, target)
        total = mse
        parts = {"mse": mse.item()}
        if self.use_sobolev:
            sob, sob_info = self.sobolev_loss(output, target)
            total = total + sob
            parts["sobolev"] = sob_info["total_loss"]
        if self.use_divergence:
            div   = self.div_penalty(output)
            total = total + div
            parts["divergence"] = div.item()
        parts["total"] = total.item()
        return total, parts

    def train_epoch(self, loader) -> float:
        self.model.train()
        running = 0.0
        for branch, trunk, target in tqdm(loader, desc="Train", leave=False):
            branch = branch.to(self.device)
            trunk  = trunk.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler:
                with autocast("cuda"):
                    out = self.model(branch, trunk)
                    loss, _ = self._loss(out, target)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(branch, trunk)
                loss, _ = self._loss(out, target)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            running += loss.item()
        return running / len(loader)

    def validate(self, loader):
        self.model.eval()
        total, preds, tgts = 0.0, [], []
        with torch.no_grad():
            for branch, trunk, target in tqdm(loader, desc="Val  ", leave=False):
                branch = branch.to(self.device)
                trunk  = trunk.to(self.device)
                target = target.to(self.device)
                out    = self.model(branch, trunk)
                loss, _ = self._loss(out, target)
                total += loss.item()
                preds.append(out.cpu())
                tgts.append(target.cpu())
        avg  = total / len(loader)
        mets = compute_extended_metrics(
            torch.cat(preds), torch.cat(tgts), self.field_names
        )
        return avg, mets

    def train(self, train_loader, val_loader) -> None:
        epochs   = self.config["training"]["epochs"]
        best_val = float("inf")
        print(f"\n{'='*60}")
        print(f"Operator   : {self.operator_name}")
        print(f"Sobolev    : {'ON' if self.use_sobolev   else 'OFF'}")
        print(f"DivPenalty : {'ON' if self.use_divergence else 'OFF'}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            t0 = time.time()
            tl = self.train_epoch(train_loader)
            vl, metrics = self.validate(val_loader)
            self.scheduler.step(vl)
            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["learning_rate"].append(lr)
            print(
                f"Epoch {epoch+1:4d}/{epochs}  "
                f"train={tl:.5f}  val={vl:.5f}  "
                f"lr={lr:.2e}  ({time.time()-t0:.1f}s)"
            )
            for fn in self.field_names:
                print(
                    f"  {fn}: R²={metrics[f'{fn}_r2']:.4f}  "
                    f"RelL2={metrics[f'{fn}_rel_l2']:.4f}  "
                    f"MAE={metrics[f'{fn}_mae']:.6f}  "
                    f"dL2={metrics[f'{fn}_deriv_l2']:.4f}"
                )
            if vl < best_val:
                best_val = vl
                self._save("best_model.pth", epoch, metrics)
                print("  ✓ best model saved")
            self.early_stopping(vl)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        self._save("final_model.pth", epochs - 1, metrics)
        self._save_history()
        print(f"\n{'='*60}\nDone  (best val={best_val:.5f})\n{'='*60}")

    def _save(self, fname, epoch, metrics):
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics":              metrics,
                "operator":             self.operator_name,
                "model_version":        self.operator_name,
                "config":               self.config,
            },
            self.output_dir / fname,
        )

    def _save_history(self):
        hist_path = self.output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ History → {hist_path}")
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(self.history["train_loss"], label="Train")
            axes[0].plot(self.history["val_loss"],   label="Val")
            axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss curves")
            axes[0].legend(); axes[0].grid(True)
            axes[1].plot(self.history["learning_rate"])
            axes[1].set(xlabel="Epoch", ylabel="LR", title="Learning Rate")
            axes[1].set_yscale("log"); axes[1].grid(True)
            plt.tight_layout()
            plot_dir = project_root / self.config["output_paths"]["plots"]
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / "training_curves.png", dpi=150)
            plt.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(trainer, test_loader, n_repeats=5):
    trainer.model.eval()
    branch, trunk, _ = next(iter(test_loader))
    branch = branch.to(trainer.device)
    trunk  = trunk.to(trainer.device)
    times  = []
    with torch.no_grad():
        for _ in range(n_repeats + 2):
            t0 = time.perf_counter()
            trainer.model(branch, trunk)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    mean_ms = 1000 * sum(times[2:]) / len(times[2:])
    speedup = 3600.0 / (mean_ms / 1000)
    print(f"\nBenchmark  ({trainer.operator_name})")
    print(f"  Inference : {mean_ms:.3f} ms / batch")
    print(f"  Speedup   : {speedup:,.0f}×  (vs 1 h CFD)")
    if speedup >= 1000:
        print("  ✓ >1000× target achieved!")
    return {"mean_ms": mean_ms, "speedup": speedup}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train AP1000 DeepONet (upgraded)")
    p.add_argument("--operator",          default=None,
                   choices=["deeponet", "deeponet_fourier"],
                   help="Operator architecture (legacy flag, prefer --model-version)")
    p.add_argument("--model-version",     default=None,
                   dest="model_version",
                   choices=["deeponet", "deeponet_fourier"],
                   help="Select neural operator architecture")
    p.add_argument("--epochs",            type=int,   default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--sobolev-weight",    type=float, default=0.1,
                   dest="sobolev_weight")
    p.add_argument("--divergence-weight", type=float, default=0.01,
                   dest="divergence_weight")
    p.add_argument("--no-sobolev",        action="store_true")
    p.add_argument("--no-divergence",     action="store_true")
    p.add_argument("--benchmark",         action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --model-version takes precedence over --operator; default to deeponet_fourier
    operator = args.model_version or args.operator or "deeponet_fourier"

    config_path       = project_root / "configs" / "config.yaml"
    model_config_path = project_root / "configs" / "model_config.yaml"
    h5_path           = (
        project_root / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
    )

    if not h5_path.exists():
        print(f"ERROR: Dataset not found at {h5_path}")
        print("Run:  python scripts/generate_dataset.py")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=project_root / "configs"
    ) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = Path(tmp.name)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            h5_path, cfg["training"]["batch_size"], num_workers=0
        )
        trainer = UpgradedDeepONetTrainer(
            config_path       = tmp_path,
            model_config_path = model_config_path,
            operator          = operator,
            sobolev_weight    = args.sobolev_weight,
            divergence_weight = args.divergence_weight,
            use_sobolev       = not args.no_sobolev,
            use_divergence    = not args.no_divergence,
        )
        trainer.train(train_loader, val_loader)
        if args.benchmark:
            run_benchmark(trainer, test_loader)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    print("=" * 60 + "\nSTEP 3: Train Upgraded DeepONet\n" + "=" * 60)
    main()
