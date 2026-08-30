"""
edit_008b_real_operator_ablation.py
=====================================
DEFINITIVE field-only ablation using REAL DeepONetFourier predictions.

This replaces the synthetic proxy used in edit_008 with actual operator
forward-pass outputs. The workflow is:

  1. Load the trained DeepONetFourier (results/models/deeponet_fourier_best.pth).
  2. Build a representative condition grid covering the NPPAD Normal/LOCAC
     parameter space (velocity, break_size, temperature).
  3. Run the operator on every grid point → get real fields (p, |v|, k, T).
  4. Compute the three CFD anomaly signals from those real fields:
       turb_anomaly  = max(0, (max_k - k_nominal) / k_nominal)
       pstd_anomaly  = max(0, (std_p - p_std_nominal) / p_std_nominal)
       flow_deficit  = max(0, 1 - v_inlet / v_nominal)
  5. For each NPPAD row, look up CFD signals for the nearest grid point.
  6. Run the Blended vs CFD-only ablation (5 seeds) and report results.

Run from project root:
    conda run -n minor_proj python Journal/scripts/edit_008b_real_operator_ablation.py

Output:
    Journal/scripts/edit_008b_ablation_results.json
    Journal/scripts/edit_008b_ablation_comparison.png
"""

import sys, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, accuracy_score)

# ── project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = PROJECT_ROOT / "results" / "models" / "deeponet_fourier_best.pth"
SCALERS_PATH = PROJECT_ROOT / "data"    / "deeponet_dataset" / "scalers.pkl"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"
MODEL_CFG    = PROJECT_ROOT / "configs" / "model_config.yaml"
H5_PATH      = PROJECT_ROOT / "data"    / "deeponet_dataset" / "deeponet_dataset.h5"

SEEDS     = [42, 7, 13, 99, 2024]
GB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=20)

# ── load config ────────────────────────────────────────────────────────────────
def load_config():
    with open(CONFIG_PATH)  as f: cfg = yaml.safe_load(f)
    with open(MODEL_CFG)    as f: cfg.update(yaml.safe_load(f))
    return cfg

# ── NPPAD loader ───────────────────────────────────────────────────────────────
def _extract_nppad_features(df):
    out = pd.DataFrame()
    out['P']        = df['P']
    out['TAVG']     = df['TAVG']
    out['WRCA']     = df['WRCA']
    out['PSGA']     = df['PSGA']
    out['SCMA']     = df['SCMA']
    out['DNBR']     = df['DNBR']
    out['DT_HL_CL'] = df['THA'] - df['TCA']
    return out

def load_nppad():
    nppad_dir = PROJECT_ROOT / "data" / "nppad" / "operation_csv_data"
    normal_dir, locac_dir = nppad_dir / "Normal", nppad_dir / "LOCAC"
    if not (normal_dir.exists() and locac_dir.exists()):
        raise RuntimeError(f"NPPAD not found at {nppad_dir}")
    rows, labels = [], []
    for fp in sorted(normal_dir.glob("*.csv")):
        df = _extract_nppad_features(pd.read_csv(fp))
        rows.append(df); labels.extend([0]*len(df))
    for fp in sorted(locac_dir.glob("*.csv")):
        df = _extract_nppad_features(pd.read_csv(fp))
        rows.append(df); labels.extend([1]*len(df))
    combined = pd.concat(rows, ignore_index=True)
    y = np.array(labels)
    print(f"  NPPAD loaded: {int((y==0).sum())} Normal, {int((y==1).sum())} LOCAC")
    return combined, y

# ── operator setup ─────────────────────────────────────────────────────────────
def load_operator(cfg, device):
    from src.deeponet.deeponet_fourier import DeepONetFourier
    model = DeepONetFourier(cfg).to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  DeepONetFourier loaded from {MODEL_PATH.name}")
    return model

def load_scalers():
    with open(SCALERS_PATH, 'rb') as f: sc = pickle.load(f)
    print(f"  Scalers loaded from {SCALERS_PATH.name}")
    return sc

def load_trunk(scalers, device):
    import h5py
    with h5py.File(H5_PATH, 'r') as f:
        trunk_norm = torch.FloatTensor(f['train']['trunk'][:]).to(device)
    trunk_phys = scalers['trunk'].inverse_transform(trunk_norm.cpu().numpy())
    print(f"  Trunk grid: {trunk_norm.shape[0]} nodes")
    return trunk_norm, trunk_phys

def denorm_field(data, field_name, scalers):
    sc = scalers['targets'][field_name]
    return sc.inverse_transform(data.reshape(-1,1)).reshape(data.shape)

# ── forward pass for one condition ─────────────────────────────────────────────
@torch.no_grad()
def operator_predict(model, scalers, trunk_norm, velocity, break_size, temperature, device):
    branch_raw = np.array([[velocity, break_size, temperature]])
    branch_norm = scalers['branch'].transform(branch_raw)
    branch_t = torch.FloatTensor(branch_norm).to(device)
    preds = model(branch_t, trunk_norm)          # [1, 4, N]
    preds_np = preds.cpu().numpy()[0]            # [4, N]
    field_names = ['pressure','velocity_magnitude','turbulence_k','temperature']
    fields = {}
    for i, fn in enumerate(field_names):
        fields[fn] = denorm_field(preds_np[i], fn, scalers)
    return fields

# ── CFD anomaly signals ────────────────────────────────────────────────────────
# Nominals derived from Normal-operation CFD: v=5.0 m/s, b=0%, T=305°C
K_NOMINAL   = 0.65   # m²/s² — typical TKE near AP1000 design point
PSTD_NOMINAL = 2.0   # Pa (normalised units) — nominal pressure std
V_NOMINAL    = 5.0   # m/s inlet velocity reference

def compute_cfd_anomaly(fields, trunk_phys):
    """Compute the three eta_CFD component signals from real operator fields."""
    # turb_anomaly: excess turbulence relative to nominal
    max_k = float(fields['turbulence_k'].max())
    turb_anomaly = max(0.0, (max_k - K_NOMINAL) / max(K_NOMINAL, 1e-6))

    # pstd_anomaly: pressure heterogeneity
    std_p = float(fields['pressure'].std())
    pstd_anomaly = max(0.0, (std_p - PSTD_NOMINAL) / max(PSTD_NOMINAL, 1e-6))

    # flow_deficit: inlet-region velocity reduction
    x_coords = trunk_phys[:, 0]
    inlet_mask = x_coords < np.percentile(x_coords, 10)
    v_inlet = float(fields['velocity_magnitude'][inlet_mask].mean())
    flow_deficit = max(0.0, 1.0 - v_inlet / V_NOMINAL)

    # clip each to [0,1] then return as array
    ta = min(1.0, turb_anomaly)
    pa = min(1.0, pstd_anomaly)
    fd = min(1.0, flow_deficit)
    return np.array([ta, pa, fd])   # shape (3,)

# ── build CFD-signal lookup table ──────────────────────────────────────────────
def build_cfd_lookup(model, scalers, trunk_norm, trunk_phys, device):
    """
    Run the operator over a grid of (velocity, break_size, temperature) and
    record the three CFD anomaly signals for each point.

    Grid: 5 velocities × 11 break sizes × 5 temperatures = 275 points.
    Normal-region: break_size = 0-1%; LOCAC-region: 1-10%.
    """
    velocities   = [4.0, 4.5, 5.0, 5.5, 6.0]
    break_sizes  = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    temperatures = [290.0, 297.5, 305.0, 312.5, 320.0]

    records = []
    total = len(velocities) * len(break_sizes) * len(temperatures)
    print(f"  Running operator on {total} grid points …", flush=True)

    for v in velocities:
        for b in break_sizes:
            for T in temperatures:
                fields = operator_predict(model, scalers, trunk_norm, v, b, T, device)
                anomaly = compute_cfd_anomaly(fields, trunk_phys)
                records.append({'v': v, 'b': b, 'T': T,
                                'ta': anomaly[0], 'pa': anomaly[1], 'fd': anomaly[2]})

    df = pd.DataFrame(records)
    print(f"  Grid done. Break-size range: {df['b'].min():.1f}–{df['b'].max():.1f}%")
    print(f"  turb_anomaly  : {df['ta'].min():.4f} – {df['ta'].max():.4f}")
    print(f"  pstd_anomaly  : {df['pa'].min():.4f} – {df['pa'].max():.4f}")
    print(f"  flow_deficit  : {df['fd'].min():.4f} – {df['fd'].max():.4f}")
    return df

# ── assign CFD signals to NPPAD rows ──────────────────────────────────────────
def assign_cfd_signals(labels, cfd_lookup, rng):
    """
    For each NPPAD row, sample plausible (v, b, T) inputs and look up the
    nearest grid point's CFD anomaly signals.

    Normal rows → b ~ Uniform(0, 0.5);  LOCAC rows → b ~ Uniform(1, 10)
    """
    n = len(labels)
    b_sample = np.where(labels == 0,
                        rng.uniform(0.0, 0.5, n),
                        rng.uniform(1.0, 10.0, n))
    v_sample = rng.uniform(4.0, 6.0, n)
    T_sample = rng.uniform(290.0, 320.0, n)

    anomalies = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        # Find nearest grid point by Manhattan distance in normalised space
        dv = (cfd_lookup['v'] - v_sample[i]) / 2.0
        db = (cfd_lookup['b'] - b_sample[i]) / 10.0
        dT = (cfd_lookup['T'] - T_sample[i]) / 30.0
        dist = np.abs(dv) + np.abs(db) + np.abs(dT)
        nearest = dist.idxmin()
        anomalies[i, 0] = cfd_lookup.loc[nearest, 'ta']
        anomalies[i, 1] = cfd_lookup.loc[nearest, 'pa']
        anomalies[i, 2] = cfd_lookup.loc[nearest, 'fd']

    return anomalies, b_sample, v_sample, T_sample

# ── severity & NPPAD mapping (identical to translator.py) ─────────────────────
def compute_eta(break_sizes, cfd_feats, w_input):
    eta_input = np.clip(break_sizes / 10.0, 0, 1)
    eta_cfd   = np.clip(cfd_feats.mean(axis=1), 0, 1)
    return np.clip(w_input * eta_input + (1-w_input) * eta_cfd, 0, 1)

def map_nppad(eta, velocity, temperature):
    n = len(eta)
    X = np.zeros((n, 7), dtype=np.float32)
    X[:,0] = 155.5 - eta * 95.0
    X[:,1] = temperature - eta * 50.0
    X[:,2] = 16515.0 * (velocity/5.0) * (1.0 - eta * 0.6)
    X[:,3] = 67.0 - eta * 30.0
    X[:,4] = 35.0 + eta * 20.0 - eta**2 * 60.0
    X[:,5] = 5.6 + eta * 130.0
    X[:,6] = 16.0 * (1.0 - eta)
    return X

# ── single-seed evaluation ─────────────────────────────────────────────────────
def run_one(labels, break_sizes, velocity, temperature, cfd_feats, w_input, seed):
    eta = compute_eta(break_sizes, cfd_feats, w_input)
    X   = map_nppad(eta, velocity, temperature)
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2,
                                           random_state=seed, stratify=labels)
    sc   = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    clf  = GradientBoostingClassifier(**GB_PARAMS, random_state=seed)
    clf.fit(Xtr_s, ytr)
    yp  = clf.predict(Xte_s)
    ypp = clf.predict_proba(Xte_s)[:,1]
    return dict(roc_auc=roc_auc_score(yte,ypp),
                recall=recall_score(yte,yp,zero_division=0),
                f1=f1_score(yte,yp,zero_division=0),
                accuracy=accuracy_score(yte,yp))

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("="*68)
    print("Edit 008b — Real Operator Field-Only Ablation")
    print("="*68)

    cfg    = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n[1] Loading DeepONetFourier …")
    model   = load_operator(cfg, device)
    scalers = load_scalers()
    trunk_norm, trunk_phys = load_trunk(scalers, device)

    print("\n[2] Building CFD-signal lookup table (real operator predictions) …")
    cfd_lookup = build_cfd_lookup(model, scalers, trunk_norm, trunk_phys, device)

    print("\n[3] Loading NPPAD dataset …")
    nppad_df, labels = load_nppad()
    n = len(labels)

    res = {'blended': [], 'cfd_only': []}
    print("\n[4] Multi-seed evaluation …")

    for i, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({i+1}/{len(SEEDS)}) …")
        rng  = np.random.RandomState(seed)
        cfd_feats, b_sample, v_sample, T_sample = assign_cfd_signals(
            labels, cfd_lookup, rng)

        r_b = run_one(labels, b_sample, v_sample, T_sample, cfd_feats, 0.8, seed)
        r_c = run_one(labels, b_sample, v_sample, T_sample, cfd_feats, 0.0, seed)
        res['blended'].append(r_b)
        res['cfd_only'].append(r_c)
        print(f"  Blended  AUC={r_b['roc_auc']:.4f} Rec={r_b['recall']:.4f} F1={r_b['f1']:.4f}")
        print(f"  CFD-only AUC={r_c['roc_auc']:.4f} Rec={r_c['recall']:.4f} F1={r_c['f1']:.4f}")

    metrics = ['roc_auc','recall','f1','accuracy']
    agg = {m: {'blended': (np.mean([d[m] for d in res['blended']]),
                            np.std( [d[m] for d in res['blended']])),
               'cfd_only':(np.mean([d[m] for d in res['cfd_only']]),
                            np.std( [d[m] for d in res['cfd_only']]))}
           for m in metrics}

    print("\n" + "="*68)
    print(f"FINAL RESULTS (real DeepONetFourier field predictions)")
    print(f"{'Metric':<12}  {'Blended (0.8/0.2)':>26}  {'CFD-only (0.0/1.0)':>26}")
    print("-"*70)
    for m in metrics:
        mb, sb = agg[m]['blended']
        mc, sc_ = agg[m]['cfd_only']
        print(f"{m.upper():<12}  {mb:.4f} +/- {sb:.4f}              {mc:.4f} +/- {sc_:.4f}")
    print("="*68)
    delta_auc = agg['roc_auc']['blended'][0] - agg['roc_auc']['cfd_only'][0]
    delta_rec = agg['recall']['blended'][0]   - agg['recall']['cfd_only'][0]
    print(f"\nDelta AUC  (Blended - CFD-only): {delta_auc:+.4f}")
    print(f"Delta Rec  (Blended - CFD-only): {delta_rec:+.4f}")

    json_path = Path(__file__).parent / "edit_008b_ablation_results.json"
    with open(json_path,'w') as f:
        json.dump({m:{k:list(v) for k,v in d.items()} for m,d in agg.items()}, f, indent=2)
    print(f"\nJSON  -> {json_path}")

    # Bar chart
    mlabels = ['ROC-AUC','Recall','F1','Accuracy']
    x = np.arange(4); w = 0.32
    fig, ax = plt.subplots(figsize=(8,5))
    bars_b = ax.bar(x-w/2, [agg[m]['blended'][0]  for m in metrics], w,
                    yerr=[agg[m]['blended'][1]  for m in metrics],
                    label='Blended (η_input=0.8, η_CFD=0.2)',
                    color='#1f77b4', capsize=5)
    bars_c = ax.bar(x+w/2, [agg[m]['cfd_only'][0] for m in metrics], w,
                    yerr=[agg[m]['cfd_only'][1] for m in metrics],
                    label='CFD-only — real DeepONetFourier (η_input=0.0, η_CFD=1.0)',
                    color='#ff7f0e', capsize=5)
    for bars in [bars_b, bars_c]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0,4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1.2, label='Paper target (0.95)')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'LOCA Classifier: Blended vs. CFD-Only Severity\n'
                 f'(Real DeepONetFourier predictions, {len(SEEDS)} seeds, mean ± std)', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(mlabels, fontsize=11)
    ax.set_ylim(0.85, 1.05); ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = Path(__file__).parent / "edit_008b_ablation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Plot  -> {plot_path}")
    print("\nDone. Use these numbers to update edit_008_field_only_ablation.md.")

if __name__ == "__main__":
    main()
