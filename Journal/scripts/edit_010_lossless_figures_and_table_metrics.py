"""
edit_010_lossless_figures_and_table_metrics.py
===============================================
Two tasks:
 1. Regenerate Figs. 2–5 comparative reconstruction panels
    (DeepONetFourier, Transolver++, Clifford Operator) as lossless PNG
    at 300 dpi with matched colormap ranges per field.
    Saved to: results/plots/paper_figures/

 2. Compute exact per-field metrics (R², Rel-L2, MAE-norm) for
    DeepONetFourier on the 300-sample test set → Table VII replacement.

Run from project root:
    conda run -n minor_proj python Journal/scripts/edit_010_lossless_figures_and_table_metrics.py

Dataset layout (HDF5):
    test/trunk:   (25000, 3)    — node coordinates
    test/branch:  (300, 3)      — inlet params (v, b, T)
    test/targets: (300, 4, 25000) — ground truth fields [p, |v|, k, T]

Model forward signature (all three architectures):
    model(branch: [B,3], trunk: [N,3]) -> [B, 4, N]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import torch
import yaml

# ── paths ────────────────────────────────────────────────────────────────────
FIGURE_DIR      = PROJECT_ROOT / "results" / "plots" / "paper_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DON  = PROJECT_ROOT / "results" / "models" / "deeponet_fourier_best.pth"
CHECKPOINT_TRA  = PROJECT_ROOT / "results" / "models" / "transolver_best.pth"
CHECKPOINT_CLI  = PROJECT_ROOT / "results" / "models" / "clifford_best.pth"
DATASET_PATH    = PROJECT_ROOT / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
CONFIG_PATH     = PROJECT_ROOT / "configs" / "model_config.yaml"

FIELDS = ["pressure", "velocity_magnitude", "turbulence_k", "temperature"]
FIELD_LABELS = {
    "pressure":           "Pressure $p$ (norm.)",
    "velocity_magnitude": "Velocity Magnitude $|\\mathbf{v}|$ (norm.)",
    "turbulence_k":       "Turbulence KE $k$ (norm.)",
    "temperature":        "Temperature $T$ (norm.)",
}
# Paper figure numbers: Figs. 2, 3, 4, 5
FIG_NUMBERS = {
    "pressure": 2,
    "velocity_magnitude": 3,
    "turbulence_k": 4,
    "temperature": 5,
}
CMAP = "viridis"
DPI  = 300

# ── dataset loader ────────────────────────────────────────────────────────────

def load_test_split():
    """Load the test split.
    Returns:
        coords: (N_nodes, 3)
        params: (N_test, 3)
        targets: (N_test, N_nodes, 4)  — transposed from (N_test, 4, N_nodes)
    """
    with h5py.File(DATASET_PATH, "r") as f:
        coords  = f["test/trunk"][:]              # (25000, 3)
        params  = f["test/branch"][:]             # (300, 3)
        targets = np.transpose(f["test/targets"][:], (0, 2, 1))  # (300, 25000, 4)
    return coords, params, targets

# ── metric helpers ────────────────────────────────────────────────────────────

def r2_score(pred, true):
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)

def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))

def mae_norm(pred, true):
    return float(np.mean(np.abs(pred - true)))

# ── model loaders ─────────────────────────────────────────────────────────────

def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_deeponet():
    from deeponet.deeponet_fourier import DeepONetFourier
    cfg = _load_cfg()
    model = DeepONetFourier(cfg)
    ckpt = torch.load(CHECKPOINT_DON, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def load_transolver():
    from operators.transolver_operator import TransolverOperator
    cfg = _load_cfg()
    model = TransolverOperator.from_config(cfg)
    ckpt = torch.load(CHECKPOINT_TRA, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def load_clifford():
    from operators.clifford_operator import CliffordNeuralOperator
    cfg = _load_cfg()
    model = CliffordNeuralOperator.from_config(cfg)
    ckpt = torch.load(CHECKPOINT_CLI, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ── inference helper ──────────────────────────────────────────────────────────

def predict(model, coords, param_vec):
    """
    Args:
        model:     any of the three architectures
        coords:    (N_nodes, 3) numpy array
        param_vec: (3,) numpy array

    Returns:
        (N_nodes, 4) numpy array — all four fields
    """
    with torch.no_grad():
        branch = torch.tensor(param_vec, dtype=torch.float32).unsqueeze(0)   # (1,3)
        trunk  = torch.tensor(coords,    dtype=torch.float32)                 # (N,3)
        out    = model(branch, trunk)   # (1, 4, N)  or (B,4,N)
    if out.dim() == 3:
        out = out.squeeze(0)           # (4, N)
    return out.T.numpy()               # (N, 4)


@torch.no_grad()
def predict_deeponet_batches(model, coords, params, batch_size=32):
    """Yield DeepONetFourier predictions without recomputing trunk bases.

    The standard ``forward`` method evaluates each of the four trunk networks
    for every batch.  The mesh is fixed across the test split, so the four
    trunk bases can be computed once and reused for all 300 branch inputs.
    """
    trunk = torch.as_tensor(coords, dtype=torch.float32)
    branch = torch.as_tensor(params, dtype=torch.float32)
    trunk_bases = [net(trunk) for net in model.trunk_nets]

    for start in range(0, len(branch), batch_size):
        stop = min(start + batch_size, len(branch))
        branch_batch = branch[start:stop]
        fields = []
        for field_idx, trunk_basis in enumerate(trunk_bases):
            branch_basis = model.branch_nets[field_idx](branch_batch)
            field = branch_basis @ trunk_basis.T + model.biases[field_idx]
            fields.append(field)
        yield start, stop, torch.stack(fields, dim=1).cpu().numpy()

# ── figure generation ─────────────────────────────────────────────────────────

def make_panel_figure(case_idx, field_name, field_idx,
                      x, y, true_vals,
                      pred_don, pred_tra, pred_cli):
    """
    3-row × 3-col figure (rows = architectures, cols = GT/Pred/Error).
    All architectures share the same vmin/vmax derived from ground truth.
    Saved as lossless PNG, 300 dpi.
    """
    vmin, vmax = float(true_vals.min()), float(true_vals.max())
    field_levels = np.linspace(vmin, vmax, 41)

    arch_names  = ["DeepONetFourier", "Transolver++", "Clifford Operator"]
    predictions = [pred_don, pred_tra, pred_cli]
    error_max = max(
        float(np.max(np.abs(pred[:, field_idx] - true_vals)))
        for pred in predictions if pred is not None
    )
    error_levels = np.linspace(0.0, max(error_max, 1e-12), 41)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig_num  = FIG_NUMBERS[field_name]
    fig.suptitle(
        f"Fig. {fig_num}: Field Reconstruction — {FIELD_LABELS[field_name]}\n"
        f"Representative test sample #{case_idx} (Synthetic Benchmark)",
        fontsize=13, fontweight="bold",
    )

    for row, (arch, pred) in enumerate(zip(arch_names, predictions)):
        ax_gt, ax_pr, ax_er = axes[row]

        # Ground truth (always available)
        tcf_gt = ax_gt.tricontourf(
            x, y, true_vals, levels=field_levels, cmap=CMAP, extend="both"
        )
        ax_gt.set_title(f"{arch}\nGround Truth", fontsize=9)
        ax_gt.axis("equal"); ax_gt.set_xlabel("x"); ax_gt.set_ylabel("y")
        plt.colorbar(tcf_gt, ax=ax_gt, format="%.3f")

        if pred is not None:
            pred_vals = pred[:, field_idx]

            # Prediction — same colorscale as GT
            tcf_pr = ax_pr.tricontourf(
                x, y, pred_vals, levels=field_levels, cmap=CMAP, extend="both"
            )
            ax_pr.set_title("Prediction", fontsize=9)
            ax_pr.axis("equal"); ax_pr.set_xlabel("x"); ax_pr.set_ylabel("y")
            plt.colorbar(tcf_pr, ax=ax_pr, format="%.3f")

            # Absolute error
            abs_err = np.abs(pred_vals - true_vals)
            tcf_er = ax_er.tricontourf(
                x, y, abs_err, levels=error_levels, cmap="Reds", extend="max"
            )
            ax_er.set_title("Absolute Error", fontsize=9)
            ax_er.axis("equal"); ax_er.set_xlabel("x"); ax_er.set_ylabel("y")
            plt.colorbar(tcf_er, ax=ax_er, format="%.4f")
        else:
            for ax in [ax_pr, ax_er]:
                ax.text(0.5, 0.5,
                        "Checkpoint unavailable\n(train model first)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="gray")
                ax.axis("off")

    # Reserve the top margin explicitly so the first-row architecture label
    # and the figure-level title are not clipped by ``bbox_inches='tight'``.
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = FIGURE_DIR / f"fig_{fig_num}_{field_name}_lossless.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"    Saved: {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path

# ── Table VII metric computation ──────────────────────────────────────────────

def compute_table_vii(model_don, coords, params, targets):
    """
    Compute per-field R², Rel-L2, MAE(norm.) for DeepONetFourier on all 300
    test samples. Returns a dict and prints the table to stdout.
    """
    n_test = targets.shape[0]
    per_field = {f: {"r2": [], "rel_l2": [], "mae": []} for f in FIELDS}
    target_means = targets.mean(axis=(0, 1), dtype=np.float64)
    pooled = {
        f: {"ss_res": 0.0, "ss_tot": 0.0, "target_sq": 0.0,
            "abs_err": 0.0, "count": 0}
        for f in FIELDS
    }

    for start, stop, pred_batch in predict_deeponet_batches(
        model_don, coords, params
    ):
        for local_idx, sample_idx in enumerate(range(start, stop)):
            pred = np.transpose(pred_batch[local_idx], (1, 0))  # (N, 4)
            true = targets[sample_idx]
            for fi, fname in enumerate(FIELDS):
                per_field[fname]["r2"].append(r2_score(pred[:, fi], true[:, fi]))
                per_field[fname]["rel_l2"].append(rel_l2(pred[:, fi], true[:, fi]))
                per_field[fname]["mae"].append(mae_norm(pred[:, fi], true[:, fi]))
                err = pred[:, fi].astype(np.float64) - true[:, fi].astype(np.float64)
                true64 = true[:, fi].astype(np.float64)
                pooled[fname]["ss_res"] += float(np.sum(err ** 2))
                pooled[fname]["ss_tot"] += float(
                    np.sum((true64 - target_means[fi]) ** 2)
                )
                pooled[fname]["target_sq"] += float(np.sum(true64 ** 2))
                pooled[fname]["abs_err"] += float(np.sum(np.abs(err)))
                pooled[fname]["count"] += int(true64.size)
        print(f"    evaluated {stop:3d}/{n_test} test samples", flush=True)

    header = (
        f"{'Field':<24} {'Pooled R²':>10} {'Pooled Rel-L2':>14} "
        f"{'Pooled MAE':>12} {'Per-case R² mean±SD':>22}"
    )
    print("\n" + "=" * 70)
    print("TABLE VII — DeepONetFourier (300-sample test set, Synthetic Benchmark)")
    print("=" * 70)
    print(header)
    print("-" * 70)

    results = {}
    for fi, fname in enumerate(FIELDS):
        r2m  = np.mean(per_field[fname]["r2"])
        r2s  = np.std(per_field[fname]["r2"])
        rl2m = np.mean(per_field[fname]["rel_l2"])
        rl2s = np.std(per_field[fname]["rel_l2"])
        maem = np.mean(per_field[fname]["mae"])
        maes = np.std(per_field[fname]["mae"])
        pool = pooled[fname]
        pooled_r2 = 1.0 - pool["ss_res"] / (pool["ss_tot"] + 1e-12)
        pooled_rel_l2 = np.sqrt(pool["ss_res"] / (pool["target_sq"] + 1e-12))
        pooled_mae = pool["abs_err"] / pool["count"]
        label = FIELD_LABELS[fname].split("$")[0].strip()
        print(
            f"{label:<24} {pooled_r2:>10.4f} {pooled_rel_l2:>14.4f} "
            f"{pooled_mae:>12.5f} {r2m:>10.4f}±{r2s:.4f}"
        )
        results[fname] = dict(
            pooled_r2=float(pooled_r2),
            pooled_rel_l2=float(pooled_rel_l2),
            pooled_mae=float(pooled_mae),
            r2_mean=float(r2m), r2_std=float(r2s),
            rel_l2_mean=float(rl2m), rel_l2_std=float(rl2s),
            mae_mean=float(maem), mae_std=float(maes),
            metric_scope=(
                "pooled metrics flatten all 300 test cases and nodes; "
                "mean/std summarize per-case metrics (population SD)"
            ),
        )
    print("=" * 70)
    return results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir  : {FIGURE_DIR}\n")

    # ── load dataset ─────────────────────────────────────────────────────────
    coords, params, targets = load_test_split()
    n_test, n_nodes, n_fields = targets.shape
    print(f"Dataset: {n_test} test samples | {n_nodes} nodes | {n_fields} fields")

    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)

    # ── load models ───────────────────────────────────────────────────────────
    print("\n--- Loading checkpoints ---")

    model_don, avail_don = None, False
    if CHECKPOINT_DON.exists():
        try:
            model_don = load_deeponet()
            avail_don = True
            print(f"  ✓ DeepONetFourier  ({CHECKPOINT_DON.name})")
        except Exception as e:
            print(f"  ✗ DeepONetFourier  failed: {e}")
    else:
        print(f"  ✗ DeepONetFourier  checkpoint not found")

    model_tra, avail_tra = None, False
    if CHECKPOINT_TRA.exists():
        try:
            model_tra = load_transolver()
            avail_tra = True
            print(f"  ✓ Transolver++     ({CHECKPOINT_TRA.name})")
        except Exception as e:
            print(f"  ✗ Transolver++     failed: {e}")
    else:
        print(f"  ✗ Transolver++     checkpoint not found")

    model_cli, avail_cli = None, False
    if CHECKPOINT_CLI.exists():
        try:
            model_cli = load_clifford()
            avail_cli = True
            print(f"  ✓ Clifford         ({CHECKPOINT_CLI.name})")
        except Exception as e:
            print(f"  ✗ Clifford         failed: {e}")
    else:
        print(f"  ✗ Clifford         checkpoint not found")

    # ── run inference for representative case ─────────────────────────────────
    CASE_IDX = 0   # first test-set sample
    print(f"\n--- Running inference on test sample #{CASE_IDX} ---")

    pred_don = predict(model_don, coords, params[CASE_IDX]) if avail_don else None
    pred_tra = predict(model_tra, coords, params[CASE_IDX]) if avail_tra else None
    pred_cli = predict(model_cli, coords, params[CASE_IDX]) if avail_cli else None

    # ── generate Figs. 2–5 ────────────────────────────────────────────────────
    print("\n--- Generating Figs. 2–5 (lossless PNG, 300 dpi) ---")
    for fi, fname in enumerate(FIELDS):
        true_vals = targets[CASE_IDX, :, fi].astype(float)
        print(f"  Fig. {FIG_NUMBERS[fname]} — {fname}")
        make_panel_figure(CASE_IDX, fname, fi, x, y, true_vals,
                          pred_don, pred_tra, pred_cli)

    # ── compute Table VII ─────────────────────────────────────────────────────
    if avail_don:
        print("\n--- Computing Table VII exact metrics (all 300 test samples) ---")
        metrics = compute_table_vii(model_don, coords, params, targets)
        out_json = PROJECT_ROOT / "Journal" / "scripts" / "edit_010_table_vii_metrics.json"
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics JSON → {out_json}")
    else:
        print("\n[SKIP] Table VII metrics skipped — DeepONetFourier checkpoint unavailable.")

    print("\n✓ Done. Lossless figures in:")
    for p in sorted(FIGURE_DIR.glob("*.png")):
        print(f"   {p}")


if __name__ == "__main__":
    main()
