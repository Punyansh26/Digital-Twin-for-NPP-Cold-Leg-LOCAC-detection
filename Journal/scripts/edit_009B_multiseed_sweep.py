#!/usr/bin/env python
"""
edit_009B_multiseed_sweep.py
============================
Multi-seed training sweep for Sec. VIII result stability (Edit 009B).

Trains all three neural operators (DeepONetFourier, Transolver++, Clifford)
and the LOCA classifier over 5 fixed random seeds, then aggregates
mean ± sample-std metrics ready to paste into Tables VII–IX.

Usage (from project root, using the minor_proj conda env):
    conda run -n minor_proj python Journal/scripts/edit_009B_multiseed_sweep.py

Output directory:
    results/multiseed_sweep/YYYYMMDD_HHMMSS/
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
* All hyperparameters are read from configs/ and are NOT altered by this script.
* Script deliberately does NOT patch the paper; after reviewing the output
  JSONs, the team fills [VERIFY] placeholders in edit_009_result_stability.md.
"""
from __future__ import annotations

import json
import os
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

TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "results" / "multiseed_sweep" / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 13, 99, 2024]


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

def _build_seeded_loaders(seed: int, batch_size: int):
    """Return (train, val, test) DataLoaders with a seeded shuffle generator.

    The 70/15/15 split is fixed inside create_dataloaders (random_state=42);
    the generator here seeds only the per-epoch shuffle order.
    """
    import torch
    from src.deeponet.dataset import create_dataloaders

    g = torch.Generator()
    g.manual_seed(seed)
    try:
        # Prefer passing generator when the API supports it
        return create_dataloaders(H5_PATH, batch_size=batch_size,
                                  num_workers=0, generator=g)
    except TypeError:
        # Fallback: some versions do not expose generator kwarg
        return create_dataloaders(H5_PATH, batch_size=batch_size, num_workers=0)


# ---------------------------------------------------------------------------
# DeepONetFourier single-seed run
# ---------------------------------------------------------------------------

def run_deeponet_fourier(seed: int, cfg: dict, mcfg: dict) -> dict:
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
    batch_size  = cfg["training"]["batch_size"]
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["learning_rate"]

    train_l, val_l, test_l = _build_seeded_loaders(seed, batch_size)

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
        patience=cfg["training"]["early_stopping"]["patience"],
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

    for epoch in range(epochs):
        model.train()
        for branch, trunk, target in tqdm(
                train_l, desc=f"[DON s={seed} e={epoch+1}]", leave=False):
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
        with torch.no_grad():
            for branch, trunk, target in val_l:
                branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
                vl += _loss(model(branch, trunk), target).item()
        vl /= len(val_l)
        sched.step(vl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if es(vl):
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
                    "best_val_loss": best_val, "best_epoch": best_epoch})
    return metrics


# ---------------------------------------------------------------------------
# Transolver++ / Clifford single-seed run
# ---------------------------------------------------------------------------

def run_operator(arch: str, seed: int, cfg: dict, mcfg: dict) -> dict:
    import torch
    import torch.nn as nn
    from src.deeponet.train import MetricsCalculator, EarlyStopping

    set_global_seed(seed)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_cfg    = {**cfg, **mcfg}
    field_names = full_cfg["deeponet"]["output_fields"]
    batch_size  = cfg["training"]["batch_size"]
    epochs      = cfg["training"]["epochs"]

    train_l, val_l, test_l = _build_seeded_loaders(seed, batch_size)

    if arch == "transolver":
        from src.operators.transolver_operator import TransolverOperator
        model = TransolverOperator.from_config(full_cfg).to(device)
    else:
        from src.operators.clifford_operator import CliffordNeuralOperator
        model = CliffordNeuralOperator.from_config(full_cfg).to(device)

    criterion = nn.MSELoss()
    opt       = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    es        = EarlyStopping(patience=50, min_delta=1e-6)
    mc        = MetricsCalculator()
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for branch, trunk, target in train_l:
            branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(branch, trunk)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for branch, trunk, target in val_l:
                branch, trunk, target = (x.to(device) for x in (branch, trunk, target))
                vl += criterion(model(branch, trunk), target).item()
        vl /= len(val_l)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if es(vl):
            print(f"  [{arch} s={seed}] early stop at epoch {epoch+1}")
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
    metrics.update({"seed": seed, "arch": arch,
                    "best_val_loss": best_val, "best_epoch": best_epoch})
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

    try:
        X, y = detector.load_nppad_data()
        data_source = "nppad"
    except Exception as e:
        print(f"  [GBC s={seed}] NPPAD unavailable ({e}), using synthetic data")
        X, y = detector.generate_synthetic_data()
        data_source = "synthetic"

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.20, random_state=seed, stratify=y
    )

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
        "data_source": data_source,
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

def main() -> None:
    import yaml

    if not H5_PATH.exists():
        print(f"ERROR: Dataset not found at {H5_PATH}")
        print("Run:  conda run -n minor_proj python scripts/generate_dataset.py")
        sys.exit(1)

    with open(CONFIG_PATH)       as f: cfg  = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH) as f: mcfg = yaml.safe_load(f)
    field_names = {**cfg, **mcfg}["deeponet"]["output_fields"]

    arch_records: dict[str, list[dict]] = {
        "deeponet_fourier": [], "transolver": [], "clifford": []
    }
    clf_records: list[dict] = []

    # ------------------------------------------------------------------ #
    # Neural operator sweeps
    # ------------------------------------------------------------------ #
    for seed in SEEDS:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        for arch in ["deeponet_fourier", "transolver", "clifford"]:
            print(f"\n--- {arch}  (seed={seed}) ---")
            t0 = time.time()
            try:
                rec = (run_deeponet_fourier(seed, cfg, mcfg)
                       if arch == "deeponet_fourier"
                       else run_operator(arch, seed, cfg, mcfg))
            except Exception as exc:
                print(f"  ERROR: {exc}")
                rec = {"seed": seed, "arch": arch, "error": str(exc)}
            rec["elapsed_s"] = round(time.time() - t0, 1)
            out = OUTPUT_DIR / f"seed_{seed}_{arch}_metrics.json"
            out.write_text(json.dumps(rec, indent=2))
            print(f"  Saved {out.name}  ({rec['elapsed_s']}s)")
            if "error" not in rec:
                arch_records[arch].append(rec)

    # ------------------------------------------------------------------ #
    # Classifier sweeps
    # ------------------------------------------------------------------ #
    for seed in SEEDS:
        print(f"\n--- GBC classifier  (seed={seed}) ---")
        try:
            rec = run_classifier(seed)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rec = {"seed": seed, "arch": "gbc_classifier", "error": str(exc)}
        out = OUTPUT_DIR / f"seed_{seed}_gbc_classifier_metrics.json"
        out.write_text(json.dumps(rec, indent=2))
        if "error" not in rec:
            clf_records.append(rec)

    # ------------------------------------------------------------------ #
    # Table VII: DeepONetFourier per-field
    # ------------------------------------------------------------------ #
    field_metric_keys = (
        [f"{f}_r2"      for f in field_names] +
        [f"{f}_rel_l2"  for f in field_names] +
        [f"{f}_mae"     for f in field_names]
    )
    t7 = aggregate(arch_records["deeponet_fourier"], field_metric_keys)
    t7["arch"] = "deeponet_fourier"
    (OUTPUT_DIR / "aggregate_table_VII.json").write_text(json.dumps(t7, indent=2))

    # ------------------------------------------------------------------ #
    # Table VIII: all three arches, per-field and mean-field R²
    # ------------------------------------------------------------------ #
    t8: dict = {}
    for arch, recs in arch_records.items():
        for rec in recs:
            fR2 = [rec[f"{f}_r2"] for f in field_names if f"{f}_r2" in rec]
            rec["mean_field_r2"] = sum(fR2) / len(fR2) if fR2 else float("nan")
        keys8 = ["mean_field_r2"] + [f"{f}_r2" for f in field_names]
        agg   = aggregate(recs, keys8)
        agg["arch"] = arch
        t8[arch] = agg
    (OUTPUT_DIR / "aggregate_table_VIII.json").write_text(json.dumps(t8, indent=2))

    # ------------------------------------------------------------------ #
    # Table IX: classifier
    # ------------------------------------------------------------------ #
    clf_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    t9 = aggregate(clf_records, clf_keys)
    t9["arch"] = "gbc_classifier"
    (OUTPUT_DIR / "aggregate_table_IX.json").write_text(json.dumps(t9, indent=2))

    # ------------------------------------------------------------------ #
    # Human-readable summary
    # ------------------------------------------------------------------ #
    lines = [
        "=" * 72,
        f"MULTI-SEED SWEEP SUMMARY  ({TIMESTAMP})",
        f"Seeds: {SEEDS}",
        "=" * 72,
        "",
        "TABLE VII — DeepONetFourier per-field R² (mean ± std, 5 seeds)",
        "-" * 72,
    ]
    for f in field_names:
        k = f"{f}_r2"
        if f"{k}_mean" in t7:
            lines.append(f"  {f:22s} R2  {fmt(t7[f'{k}_mean'], t7[f'{k}_std'])}")

    lines += [
        "",
        "TABLE VIII — Mean-field R² by architecture (mean ± std, 5 seeds)",
        "-" * 72,
    ]
    for arch in ["deeponet_fourier", "transolver", "clifford"]:
        if arch in t8 and "mean_field_r2_mean" in t8[arch]:
            a = t8[arch]
            lines.append(
                f"  {arch:22s} mean R²  {fmt(a['mean_field_r2_mean'], a['mean_field_r2_std'])}"
            )

    lines += [
        "",
        "TABLE IX — LOCA Classifier (mean ± std, 5 seeds)",
        "-" * 72,
    ]
    for k in clf_keys:
        if f"{k}_mean" in t9:
            lines.append(f"  {k:22s} {fmt(t9[f'{k}_mean'], t9[f'{k}_std'])}")

    lines += [
        "",
        "=" * 72,
        "NEXT STEP: Replace [VERIFY] tokens in",
        "  Journal/edits/edit_009_result_stability.md",
        "  with the mean ± std values shown above.",
        "=" * 72,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUTPUT_DIR / "sweep_summary.txt").write_text(summary)
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
