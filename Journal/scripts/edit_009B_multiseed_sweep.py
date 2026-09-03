#!/usr/bin/env python
"""
edit_009B_multiseed_sweep.py
============================
Resumable multi-seed training sweep for Sec. VIII result stability (Edit 009B).

Trains the three neural operators (DeepONetFourier, Transolver++, Clifford)
over five fixed random seeds, then aggregates mean ± sample-std metrics. The
LOCA classifier remains available through ``--architectures gbc_classifier``
but is excluded from the default operator-only run.

Default quick protocol (from the project root):
    conda run -n minor_proj python Journal/scripts/edit_009B_multiseed_sweep.py

Resume a run without overwriting completed records:
    conda run -n minor_proj python Journal/scripts/edit_009B_multiseed_sweep.py \
        --resume-dir results/multiseed_sweep/quick_<timestamp>

Output directory:
    results/multiseed_sweep/quick_YYYYMMDD_HHMMSS/
        seed_<S>_<arch>_metrics.json   -- one JSON per (seed, arch) pair
        aggregate_table_VII.json       -- Table VII  (DeepONetFourier, per-field)
        aggregate_table_VIII.json      -- Table VIII (all three arches, mean field R²)
        aggregate_table_IX.json        -- Table IX   (classifier metrics)
        sweep_summary.txt              -- human-readable summary ready for LaTeX

Notes
-----
* The fixed dataset split (70/15/15, random_state=42 inside create_dataloaders)
  is PRESERVED.  The seed here controls only model init and dataloader shuffle
  ordering.
* Classifier seed controls GradientBoostingClassifier random_state AND the
  stratified 80/20 split.
* The default protocol uses at most 300 epochs and 4,096 reproducibly sampled
  mesh points per optimization/validation step. Held-out metrics still use all
  25,000 points and all 300 test cases.
* This shortened protocol is a new experiment. Do not aggregate its records
  with the earlier 2,000-epoch full-grid run.
* Completed JSON records are never overwritten. Failed attempts are retried to
  a new ``_retry_NN`` file when a run directory is resumed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH       = PROJECT_ROOT / "configs" / "config.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
H5_PATH           = PROJECT_ROOT / "data" / "deeponet_dataset" / "deeponet_dataset.h5"

SEEDS = [42, 7, 13, 99, 2024]
OPERATOR_ARCHES = ["deeponet_fourier", "transolver", "clifford"]


# ---------------------------------------------------------------------------
# Seeding helper
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Seeded DataLoaders
# ---------------------------------------------------------------------------

def _build_seeded_loaders(seed: int, batch_size: int, test_batch_size: int):
    """Return (train, val, test) DataLoaders with a seeded shuffle generator.

    The 70/15/15 split is fixed inside create_dataloaders (random_state=42);
    the generator here seeds only the per-epoch shuffle order.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.deeponet.dataset import DeepONetDataset, deeponet_collate_fn

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = DeepONetDataset(H5_PATH, "train")
    val_ds = DeepONetDataset(H5_PATH, "val")
    test_ds = DeepONetDataset(H5_PATH, "test")
    common = {"num_workers": 0, "pin_memory": torch.cuda.is_available(),
              "collate_fn": deeponet_collate_fn}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   generator=g, **common),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, **common),
    )


def _point_subset(trunk, target, max_points: int, generator):
    """Select a reproducible sorted point subset before GPU transfer."""
    import torch

    n_points = trunk.shape[0]
    if max_points <= 0 or max_points >= n_points:
        return trunk, target
    index = torch.randperm(n_points, generator=generator)[:max_points]
    index = torch.sort(index).values
    return trunk.index_select(0, index), target.index_select(-1, index)


# ---------------------------------------------------------------------------
# DeepONetFourier single-seed run
# ---------------------------------------------------------------------------

def run_deeponet_fourier(seed: int, cfg: dict, mcfg: dict, args) -> dict:
    import torch
    import torch.nn as nn
    import yaml
    from src.deeponet.deeponet_fourier import DeepONetFourier
    from src.deeponet.sobolev_loss import SobolevLoss
    from src.deeponet.model import DeepONetLoss
    from src.deeponet.train import MetricsCalculator, EarlyStopping
    from src.physics.divergence_penalty import DivergencePenalty
    from torch.amp import GradScaler, autocast
    from tqdm import tqdm

    set_global_seed(seed)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_cfg    = {**cfg, **mcfg}
    field_names = full_cfg["deeponet"]["output_fields"]
    batch_size  = args.batch_size
    epochs      = args.epochs
    lr          = args.learning_rate

    train_l, val_l, test_l = _build_seeded_loaders(
        seed, batch_size, args.test_batch_size
    )
    train_point_rng = torch.Generator().manual_seed(seed + 10_000)

    model    = DeepONetFourier.from_legacy_config(full_cfg).to(device)
    mse_loss = DeepONetLoss(weights=[1.0] * len(field_names))
    sob_loss = SobolevLoss(alpha=1.0, beta=0.1, use_autograd=False)
    div_pen  = DivergencePenalty(weight=0.01)
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min",
        patience=cfg["training"]["scheduler"]["patience"],
        factor=cfg["training"]["scheduler"]["factor"],
    )
    es       = EarlyStopping(
        patience=args.patience,
        min_delta=cfg["training"]["early_stopping"]["min_delta"],
    )
    use_amp  = device.type == "cuda"
    scaler   = GradScaler("cuda") if use_amp else None
    mc       = MetricsCalculator()

    def _loss(out, tgt):
        total = mse_loss(out, tgt)
        sob, _ = sob_loss(out, tgt)
        div = div_pen(out)
        return total + sob + div

    best_val, best_state, best_epoch = float("inf"), None, 0

    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        for branch, trunk, target in tqdm(
                train_l, desc=f"[DON s={seed} e={epoch+1}]", leave=False):
            trunk, target = _point_subset(
                trunk, target, args.train_points, train_point_rng
            )
            branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
            opt.zero_grad()
            if scaler:
                with autocast("cuda"):
                    out  = model(branch, trunk)
                    loss = _loss(out, target)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                out  = model(branch, trunk)
                loss = _loss(out, target)
                loss.backward()
                opt.step()

        model.eval()
        vl = 0.0
        val_point_rng = torch.Generator().manual_seed(seed + 20_000)
        with torch.no_grad():
            for branch, trunk, target in val_l:
                trunk, target = _point_subset(
                    trunk, target, args.val_points, val_point_rng
                )
                branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
                vl += _loss(model(branch, trunk), target).item()
        vl /= len(val_l)
        sched.step(vl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        es(vl)
        if es.early_stop:
            print(f"  [DON s={seed}] early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for branch, trunk, target in test_l:
            branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
            preds.append(model(branch, trunk).cpu())
            tgts.append(target.cpu())
    metrics = mc.compute_all_metrics(torch.cat(preds), torch.cat(tgts), field_names)
    metrics.update({"seed": seed, "arch": "deeponet_fourier",
                    "best_val_loss": best_val, "best_epoch": best_epoch,
                    "epochs_ran": epochs_ran,
                    "test_cases": len(test_l.dataset),
                    "test_points_per_case": int(test_l.dataset.trunk_data.shape[0])})
    return metrics


# ---------------------------------------------------------------------------
# Transolver++ / Clifford single-seed run
# ---------------------------------------------------------------------------

def run_operator(arch: str, seed: int, cfg: dict, mcfg: dict, args) -> dict:
    import torch
    import torch.nn as nn
    from src.deeponet.train import MetricsCalculator, EarlyStopping
    from tqdm.auto import tqdm

    set_global_seed(seed)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_cfg    = {**cfg, **mcfg}
    field_names = full_cfg["deeponet"]["output_fields"]
    batch_size  = args.batch_size
    epochs      = args.epochs

    train_l, val_l, test_l = _build_seeded_loaders(
        seed, batch_size, args.test_batch_size
    )
    train_point_rng = torch.Generator().manual_seed(seed + 10_000)

    if arch == "transolver":
        from src.operators.transolver_operator import TransolverOperator
        model = TransolverOperator.from_config(full_cfg).to(device)
    else:
        from src.operators.clifford_operator import CliffordNeuralOperator
        model = CliffordNeuralOperator.from_config(full_cfg).to(device)

    criterion = nn.MSELoss()
    opt       = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    es        = EarlyStopping(patience=args.patience, min_delta=1e-6)
    mc        = MetricsCalculator()
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_state, best_epoch = float("inf"), None, 0

    epochs_ran = 0
    epoch_desc = f"[{arch} s={seed}]"
    with tqdm(
        range(epochs), desc=epoch_desc, unit="epoch", dynamic_ncols=True
    ) as epoch_bar:
        for epoch in epoch_bar:
            epochs_ran = epoch + 1
            model.train()
            train_loss = 0.0
            train_desc = f"[{arch} s={seed} e={epoch + 1}/{epochs} train]"
            for branch, trunk, target in tqdm(
                train_l, desc=train_desc, unit="batch", leave=False,
                dynamic_ncols=True,
            ):
                trunk, target = _point_subset(
                    trunk, target, args.train_points, train_point_rng
                )
                branch, trunk, target = (
                    x.to(device) for x in (branch, trunk, target)
                )
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(branch, trunk)
                    loss = criterion(pred, target)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                train_loss += loss.detach().item()
            train_loss /= len(train_l)
            sched.step()

            model.eval()
            vl = 0.0
            val_point_rng = torch.Generator().manual_seed(seed + 20_000)
            val_desc = f"[{arch} s={seed} e={epoch + 1}/{epochs} val]"
            with torch.no_grad():
                for branch, trunk, target in tqdm(
                    val_l, desc=val_desc, unit="batch", leave=False,
                    dynamic_ncols=True,
                ):
                    trunk, target = _point_subset(
                        trunk, target, args.val_points, val_point_rng
                    )
                    branch, trunk, target = (
                        x.to(device) for x in (branch, trunk, target)
                    )
                    vl += criterion(model(branch, trunk), target).item()
            vl /= len(val_l)

            if vl < best_val:
                best_val = vl
                best_state = {
                    key: value.cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch + 1

            epoch_bar.set_postfix(
                train=f"{train_loss:.3e}", val=f"{vl:.3e}",
                best=f"{best_val:.3e}", lr=f"{opt.param_groups[0]['lr']:.2e}",
            )
            es(vl)
            if es.early_stop:
                tqdm.write(f"  [{arch} s={seed}] early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for branch, trunk, target in tqdm(
            test_l, desc=f"[{arch} s={seed} full-grid test]", unit="batch",
            dynamic_ncols=True,
        ):
            branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
            preds.append(model(branch, trunk).cpu())
            tgts.append(target.cpu())
    metrics = mc.compute_all_metrics(torch.cat(preds), torch.cat(tgts), field_names)
    metrics.update({"seed": seed, "arch": arch,
                    "best_val_loss": best_val, "best_epoch": best_epoch,
                    "epochs_ran": epochs_ran,
                    "test_cases": len(test_l.dataset),
                    "test_points_per_case": int(test_l.dataset.trunk_data.shape[0])})
    return metrics


# ---------------------------------------------------------------------------
# GBC Classifier single-seed run
# ---------------------------------------------------------------------------

def run_classifier(seed: int) -> dict:
    import yaml
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from src.accident_model.train_locac_model import LOCACDetector

    set_global_seed(seed)
    detector = LOCACDetector(config_path=str(CONFIG_PATH))

    nppad_root = PROJECT_ROOT / "data" / "nppad" / "operation_csv_data"
    if not all(any((nppad_root / name).glob("*.csv")) for name in ("Normal", "LOCAC")):
        raise FileNotFoundError(
            "NPPAD Normal/LOCAC CSV files are required; synthetic fallback is disabled"
        )
    X, y = detector.load_nppad_data()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    params = cfg["locac_model"]["gb_params"]
    clf = GradientBoostingClassifier(**params, random_state=seed)
    clf.fit(X_tr, y_tr)

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    return {
        "seed":        seed,
        "arch":        "gbc_classifier",
        "data_source": "PCTRAN-simulated NPPAD + synthetic transition augmentation",
        "scaler_fit": "training rows only",
        "accuracy":    float(accuracy_score(y_te, y_pred)),
        "precision":   float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_te, y_proba)),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate(records: list[dict], keys: list[str]) -> dict:
    import numpy as np
    result: dict = {}
    for k in keys:
        vals = [r[k] for r in records if k in r and isinstance(r[k], (int, float))]
        if vals:
            arr = np.array(vals, dtype=float)
            result[f"{k}_mean"] = float(arr.mean())
            result[f"{k}_std"]  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return result


def fmt(mean: float, std: float, d: int = 4) -> str:
    return f"${mean:.{d}f} \\pm {std:.{d}f}$"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Short, resumable, non-overwriting multi-seed operator sweep"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train-points", type=int, default=4096,
                        help="points per training step; 0 means the full mesh")
    parser.add_argument("--val-points", type=int, default=4096,
                        help="points per validation step; 0 means the full mesh")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--architectures", nargs="+", default=OPERATOR_ARCHES,
        choices=OPERATOR_ARCHES + ["gbc_classifier"],
    )
    parser.add_argument(
        "--resume-dir", type=Path,
        help="existing quick-protocol directory to resume without overwriting",
    )
    args = parser.parse_args()
    for name in ("epochs", "patience", "batch_size", "test_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.train_points < 0 or args.val_points < 0:
        parser.error("point counts must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol(args) -> dict:
    return {
        "protocol_version": "short-collocation-v2",
        "dataset": str(H5_PATH.relative_to(PROJECT_ROOT)),
        "dataset_bytes": H5_PATH.stat().st_size,
        "config_sha256": _sha256(CONFIG_PATH),
        "model_config_sha256": _sha256(MODEL_CONFIG_PATH),
        "split": "archived HDF5 train/val/test groups (1400/300/300)",
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "train_points_per_step": args.train_points or "full-grid",
        "validation_points_per_step": args.val_points or "full-grid",
        "test_scope": "all cases and all 25,000 mesh points",
        "learning_rate": args.learning_rate,
        "optimizer": "Adam(weight_decay=1e-5)",
        "deeponet_fourier_loss": "fieldwise MSE + Sobolev index proxy + velocity-variation index proxy",
        "transolver_clifford_loss": "MSE",
        "seed_scope": "initialization, minibatch order, and point collocation",
        "warning": "Do not combine with the earlier 2000-epoch full-grid records.",
    }


def _record_files(output_dir: Path, seed: int, arch: str) -> list[Path]:
    return sorted(output_dir.glob(f"seed_{seed}_{arch}_metrics*.json"))


def _successful_record(output_dir: Path, seed: int, arch: str):
    for path in _record_files(output_dir, seed, arch):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "error" not in record:
            return record, path
    return None, None


def _new_record_path(output_dir: Path, seed: int, arch: str) -> Path:
    base = output_dir / f"seed_{seed}_{arch}_metrics.json"
    if not base.exists():
        return base
    attempt = 1
    while True:
        candidate = output_dir / f"seed_{seed}_{arch}_metrics_retry_{attempt:02d}.json"
        if not candidate.exists():
            return candidate
        attempt += 1


def _new_derived_path(output_dir: Path, stem: str, suffix: str) -> Path:
    base = output_dir / f"{stem}{suffix}"
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{stem}_{stamp}{suffix}"
    attempt = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{stamp}_{attempt:02d}{suffix}"
        attempt += 1
    return candidate


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    import yaml

    args = parse_args()
    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {H5_PATH}; do not substitute a different dataset"
        )

    protocol = _protocol(args)
    if args.resume_dir:
        output_dir = args.resume_dir.expanduser().resolve()
        protocol_path = output_dir / "protocol.json"
        if not protocol_path.exists():
            raise RuntimeError(
                "Refusing to mix protocols: --resume-dir must contain protocol.json "
                "from this revised script"
            )
        saved_protocol = json.loads(protocol_path.read_text())
        if saved_protocol != protocol:
            raise RuntimeError(
                "Resume protocol does not match current arguments/configuration. "
                "Start a new run directory instead."
            )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / "multiseed_sweep" / f"quick_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        protocol_path = output_dir / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    with CONFIG_PATH.open() as stream:
        cfg = yaml.safe_load(stream)
    with MODEL_CONFIG_PATH.open() as stream:
        mcfg = yaml.safe_load(stream)
    field_names = {**cfg, **mcfg}["deeponet"]["output_fields"]

    print(f"Output directory: {output_dir}")
    print(
        f"Protocol: {args.epochs} epochs maximum, {args.train_points or 25000} "
        f"training points/step, {args.val_points or 25000} validation points/step, "
        "full-grid test evaluation"
    )

    for seed in args.seeds:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        for arch in args.architectures:
            existing, existing_path = _successful_record(output_dir, seed, arch)
            if existing is not None:
                print(f"  SKIP {arch}: completed record {existing_path.name}")
                continue

            print(f"\n--- {arch} (seed={seed}) ---")
            started = time.time()
            try:
                if arch == "deeponet_fourier":
                    rec = run_deeponet_fourier(seed, cfg, mcfg, args)
                elif arch in ("transolver", "clifford"):
                    rec = run_operator(arch, seed, cfg, mcfg, args)
                else:
                    rec = run_classifier(seed)
            except Exception as exc:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                rec = {
                    "seed": seed, "arch": arch,
                    "error_type": type(exc).__name__, "error": str(exc),
                }
            rec["elapsed_s"] = round(time.time() - started, 1)
            rec["protocol_version"] = protocol["protocol_version"]
            out = _new_record_path(output_dir, seed, arch)
            out.write_text(json.dumps(rec, indent=2) + "\n")
            print(f"  Saved {out.name} ({rec['elapsed_s']}s)")
            _release_accelerator_memory()

    arch_records: dict[str, list[dict]] = {arch: [] for arch in OPERATOR_ARCHES}
    clf_records: list[dict] = []
    for seed in args.seeds:
        for arch in OPERATOR_ARCHES + ["gbc_classifier"]:
            rec, _ = _successful_record(output_dir, seed, arch)
            if rec is None:
                continue
            if arch == "gbc_classifier":
                clf_records.append(rec)
            else:
                arch_records[arch].append(rec)

    field_metric_keys = (
        [f"{field}_r2" for field in field_names]
        + [f"{field}_rel_l2" for field in field_names]
        + [f"{field}_mae" for field in field_names]
    )
    t7 = aggregate(arch_records["deeponet_fourier"], field_metric_keys)
    t7.update({"arch": "deeponet_fourier",
               "n_successful_seeds": len(arch_records["deeponet_fourier"])})

    t8: dict = {}
    for arch, records in arch_records.items():
        for rec in records:
            values = [rec[f"{field}_r2"] for field in field_names]
            rec["mean_field_r2"] = sum(values) / len(values)
        values_to_aggregate = ["mean_field_r2"] + [f"{field}_r2" for field in field_names]
        t8[arch] = aggregate(records, values_to_aggregate)
        t8[arch].update({"arch": arch, "n_successful_seeds": len(records)})

    clf_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    t9 = aggregate(clf_records, clf_keys)
    t9.update({"arch": "gbc_classifier", "n_successful_seeds": len(clf_records)})

    for stem, payload in (
        ("aggregate_table_VII", t7),
        ("aggregate_table_VIII", t8),
        ("aggregate_table_IX", t9),
    ):
        path = _new_derived_path(output_dir, stem, ".json")
        path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "=" * 72,
        "SHORT-COLLOCATION MULTI-SEED SWEEP SUMMARY",
        f"Requested seeds: {args.seeds}",
        "Do not combine these results with the earlier 2000-epoch full-grid run.",
        "=" * 72,
        "",
        "DeepONetFourier per-field pooled R²",
        "-" * 72,
    ]
    for field in field_names:
        key = f"{field}_r2"
        if f"{key}_mean" in t7:
            lines.append(
                f"  {field:22s} {fmt(t7[f'{key}_mean'], t7[f'{key}_std'])} "
                f"(n={t7['n_successful_seeds']})"
            )
    lines += ["", "Mean-field pooled R² by architecture", "-" * 72]
    for arch in OPERATOR_ARCHES:
        record = t8[arch]
        if "mean_field_r2_mean" in record:
            lines.append(
                f"  {arch:22s} "
                f"{fmt(record['mean_field_r2_mean'], record['mean_field_r2_std'])} "
                f"(n={record['n_successful_seeds']})"
            )
    if clf_records:
        lines += ["", "Classifier metrics", "-" * 72]
        for key in clf_keys:
            if f"{key}_mean" in t9:
                lines.append(
                    f"  {key:22s} {fmt(t9[f'{key}_mean'], t9[f'{key}_std'])} "
                    f"(n={t9['n_successful_seeds']})"
                )
    summary = "\n".join(lines) + "\n"
    summary_path = _new_derived_path(output_dir, "sweep_summary", ".txt")
    summary_path.write_text(summary)
    print("\n" + summary)
    print(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
