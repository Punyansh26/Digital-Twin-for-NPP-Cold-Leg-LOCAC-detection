"""
edit_005_train_baseline_deeponet.py
====================================
Train an UNMODIFIED (vanilla) DeepONet baseline under the IDENTICAL protocol
used for DeepONetFourier, Transolver++, and Clifford Operator in this study.

Architecture (from Table II "Baseline" column in paper):
  Branch:  3 → 256 → 512 → 512 → 256,  ReLU + Dropout(10%)
  Trunk:   3 → 256 → 512 → 256,          ReLU + Dropout(10%)
  Output:  dot-product + bias (4 independent pairs, one per field)
  Loss:    MSE only (no Sobolev, no divergence penalty)

Protocol (identical to existing model runs):
  Dataset:  data/deeponet_dataset/deeponet_dataset.h5  (same 70/15/15 split)
  Optimizer: Adam, lr=1e-3
  Scheduler: ReduceLROnPlateau (patience=20, factor=0.5)
  Early stopping: patience=50, min_delta=1e-6
  Batch size: 4
  Max epochs: 2000
  Mixed precision: yes (if CUDA available)
  Hardware: NVIDIA RTX 4060 (same as other runs)

Outputs:
  results/models/baseline_deeponet_best.pth
  results/models/baseline_deeponet_final.pth
  results/models/baseline_deeponet_results.json   ← metrics for Table I
  results/plots/baseline_deeponet_training_curves.png
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.deeponet.dataset import create_dataloaders

# ── hyper-parameters (must match the other architecture runs) ─────────────────
BRANCH_DIMS   = [3, 256, 512, 512, 256]   # input + hiddens + output (basis dim)
TRUNK_DIMS    = [3, 256, 512, 256]
BASIS_DIM     = 256
N_OUTPUTS     = 4
FIELD_NAMES   = ["pressure", "velocity_magnitude", "turbulence_k", "temperature"]

LR            = 1e-3
BATCH_SIZE    = 4
MAX_EPOCHS    = 2000
SCHED_PAT     = 20
SCHED_FACTOR  = 0.5
ES_PATIENCE   = 50
ES_MIN_DELTA  = 1e-6
NUM_WORKERS   = 0
MIXED_PREC    = True

H5_PATH       = PROJECT_ROOT / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
OUT_DIR       = PROJECT_ROOT / "results" / "models"
PLOT_DIR      = PROJECT_ROOT / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── vanilla branch / trunk (ReLU + Dropout(10%)) ─────────────────────────────
class VanillaBranchNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VanillaTrunkNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VanillaDeepONet(nn.Module):
    """Unmodified DeepONet — 4 independent branch-trunk pairs, MSE loss."""

    def __init__(self, branch_dims, trunk_dims, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs
        self.branches = nn.ModuleList([VanillaBranchNet(branch_dims) for _ in range(n_outputs)])
        self.trunks   = nn.ModuleList([VanillaTrunkNet(trunk_dims)   for _ in range(n_outputs)])
        self.biases   = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_outputs)])

    def forward(self, branch_input, trunk_input):
        # branch_input: [B, 3]; trunk_input: [N, 3]
        outs = []
        for i in range(self.n_outputs):
            b = self.branches[i](branch_input)          # [B, p]
            t = self.trunks[i](trunk_input)             # [N, p]
            o = torch.matmul(b, t.T) + self.biases[i]  # [B, N]
            outs.append(o)
        return torch.stack(outs, dim=1)                 # [B, 4, N]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── metrics ───────────────────────────────────────────────────────────────────
def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return (1 - ss_res / ss_tot).item()

def rel_l2(pred, target):
    return (torch.norm(pred - target) / torch.norm(target)).item()

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def compute_all_metrics(predictions, targets, field_names):
    metrics = {}
    for i, field in enumerate(field_names):
        p, t = predictions[:, i, :], targets[:, i, :]
        metrics[f"{field}_r2"]     = r2_score(p, t)
        metrics[f"{field}_rel_l2"] = rel_l2(p, t)
        metrics[f"{field}_mae"]    = mae(p, t)
    return metrics


# ── early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best, self.early_stop = 0, None, False

    def __call__(self, val_loss):
        if self.best is None:
            self.best = val_loss
        elif val_loss > self.best - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best = val_loss
            self.counter = 0


# ── training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    for branch, trunk, target in tqdm(loader, desc="  train", leave=False):
        branch, trunk, target = branch.to(device), trunk.to(device), target.to(device)
        optimizer.zero_grad()
        if scaler:
            with autocast("cuda"):
                out  = model(branch, trunk)
                loss = F.mse_loss(out, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(branch, trunk)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device, field_names):
    model.eval()
    total, all_preds, all_tgts = 0.0, [], []
    for branch, trunk, target in tqdm(loader, desc="  val  ", leave=False):
        branch, trunk, target = branch.to(device), trunk.to(device), target.to(device)
        out  = model(branch, trunk)
        loss = F.mse_loss(out, target)
        total += loss.item()
        all_preds.append(out.cpu())
        all_tgts.append(target.cpu())
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_tgts)
    metrics = compute_all_metrics(preds, targets, field_names)
    return total / len(loader), metrics


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("Vanilla DeepONet Baseline — matched-condition training")
    print("=" * 64)
    print(f"Dataset : {H5_PATH}")
    print(f"Started : {datetime.now().isoformat()}")

    if not H5_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {H5_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        H5_PATH, BATCH_SIZE, NUM_WORKERS
    )
    print(f"Train / Val / Test batches: {len(train_loader)} / {len(val_loader)} / {len(test_loader)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = VanillaDeepONet(BRANCH_DIMS, TRUNK_DIMS, N_OUTPUTS).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=SCHED_PAT, factor=SCHED_FACTOR)
    es        = EarlyStopping(ES_PATIENCE, ES_MIN_DELTA)
    scaler    = GradScaler("cuda") if MIXED_PREC and torch.cuda.is_available() else None

    # ── training loop ─────────────────────────────────────────────────────────
    best_val, history = float("inf"), {"train_loss": [], "val_loss": [], "lr": []}
    best_metrics = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, metrics   = validate(model, val_loader, device, FIELD_NAMES)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        print(f"Epoch {epoch:4d}/{MAX_EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.2e}")
        for f in FIELD_NAMES:
            print(f"  {f:25s}  R²={metrics[f+'_r2']:.4f}  relL2={metrics[f+'_rel_l2']:.4f}  MAE={metrics[f+'_mae']:.6f}")

        if val_loss < best_val:
            best_val     = val_loss
            best_metrics = metrics.copy()
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "metrics": metrics, "val_loss": val_loss},
                       OUT_DIR / "baseline_deeponet_best.pth")
            print("  ✓ saved best checkpoint")

        es(val_loss)
        if es.early_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                "metrics": metrics, "val_loss": val_loss},
               OUT_DIR / "baseline_deeponet_final.pth")

    # ── test set evaluation ───────────────────────────────────────────────────
    ckpt = torch.load(OUT_DIR / "baseline_deeponet_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_metrics = validate(model, test_loader, device, FIELD_NAMES)

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "model": "Vanilla DeepONet (unmodified, matched-condition baseline)",
        "architecture": {
            "branch": "3→256→512→512→256 ReLU+Dropout(10%)",
            "trunk":  "3→256→512→256 ReLU+Dropout(10%)",
            "basis_dim": BASIS_DIM,
            "n_outputs": N_OUTPUTS,
            "parameters": model.count_parameters(),
            "loss": "MSE only"
        },
        "protocol": {
            "dataset": str(H5_PATH),
            "split": "70/15/15",
            "optimizer": "Adam",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": ES_PATIENCE,
            "scheduler": f"ReduceLROnPlateau(patience={SCHED_PAT}, factor={SCHED_FACTOR})",
            "epochs_trained": epoch,
            "device": str(device),
            "mixed_precision": MIXED_PREC,
            "timestamp": datetime.now().isoformat()
        },
        "val_metrics": {
            "best_val_loss": best_val,
            **{k: round(v, 6) for k, v in best_metrics.items()}
        },
        "test_metrics": {
            "test_loss": round(test_loss, 6),
            **{k: round(v, 6) for k, v in test_metrics.items()}
        }
    }

    results_path = OUT_DIR / "baseline_deeponet_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {results_path}")

    # ── training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set(xlabel="Epoch", ylabel="MSE Loss", title="Vanilla DeepONet — Loss")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["lr"])
    axes[1].set(xlabel="Epoch", ylabel="Learning Rate", title="Learning Rate Schedule")
    axes[1].set_yscale("log"); axes[1].grid(True)
    plt.tight_layout()
    plot_path = PLOT_DIR / "baseline_deeponet_training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Curves saved → {plot_path}")

    # ── final summary for paper ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PAPER TABLE I — MEASURED BASELINE RESULTS")
    print("=" * 64)
    print(f"  Parameters  : {model.count_parameters():,}")
    print(f"  Best val MSE: {best_val:.4e}")
    print(f"  Test MSE    : {test_loss:.4e}")
    print()
    for f in FIELD_NAMES:
        r2    = best_metrics[f"{f}_r2"]
        rel   = best_metrics[f"{f}_rel_l2"]
        mav   = best_metrics[f"{f}_mae"]
        print(f"  {f:25s}  R²={r2:.4f}  relL2={rel:.4f}  MAE={mav:.6f}")
    print("=" * 64)
    print("\nUse these values to fill the 'Baseline DeepONet (retrained, matched)' column in Table I.")


if __name__ == "__main__":
    main()
