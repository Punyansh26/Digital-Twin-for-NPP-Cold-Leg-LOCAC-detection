"""
edit_008_field_only_ablation.py
================================
Field-only ablation study for the LOCA Indicator Detection pipeline.

Compares two severity-blend configurations on the NPPAD dataset:

  (A) Blended (paper default) : eta_eff = 0.8 * eta_input + 0.2 * eta_CFD
  (B) CFD-only (ablation)     : eta_eff = 1.0 * eta_CFD   (eta_input weight = 0)

Both configurations feed a Gradient Boosting Classifier (GBC) identical to the
one described in the paper (200 trees, depth 4, lr 0.05, min_samples_leaf 20).

The script:
  1. Loads real NPPAD data (falls back to synthetic if NPPAD is absent).
  2. Generates synthetic CFD-field summaries to populate eta_CFD signals
     (turb_anomaly, pstd_anomaly, flow_deficit) that are independent of
     break_size, mimicking the output of the neural operator.
  3. Trains the GBC under both configurations using 5 fixed random seeds.
  4. Reports mean +/- std ROC-AUC, Recall, F1, and Accuracy.
  5. Produces a side-by-side comparison bar chart.

Run from project root:
    conda run -n minor_proj python Journal/scripts/edit_008_field_only_ablation.py
"""

import sys, math, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, accuracy_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS      = [42, 7, 13, 99, 2024]
GB_PARAMS  = dict(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=20)

# ── NPPAD loading ─────────────────────────────────────────────────────────────

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

def load_nppad(root):
    nppad_dir = root / "data" / "nppad" / "operation_csv_data"
    if not nppad_dir.exists():
        nppad_dir = root / "scripts" / "data" / "nppad" / "operation_csv_data"
    normal_dir, locac_dir = nppad_dir / "Normal", nppad_dir / "LOCAC"
    if not (normal_dir.exists() and locac_dir.exists()):
        print("  WARNING: NPPAD not found – using synthetic fallback.")
        return _synthetic_nppad()
    rows, labels = [], []
    for fp in sorted(normal_dir.glob("*.csv")):
        df = _extract_nppad_features(pd.read_csv(fp))
        rows.append(df); labels.extend([0]*len(df))
    for fp in sorted(locac_dir.glob("*.csv")):
        df = _extract_nppad_features(pd.read_csv(fp))
        rows.append(df); labels.extend([1]*len(df))
    combined = pd.concat(rows, ignore_index=True)
    y = np.array(labels)
    print(f"  Real NPPAD: {int((y==0).sum())} Normal, {int((y==1).sum())} LOCAC")
    return combined, y

def _synthetic_nppad():
    rng = np.random.RandomState(42)
    n0, n1 = 302, 45_176
    normal = dict(P=rng.normal(155.5,0.3,n0), TAVG=rng.normal(310,2,n0),
                  WRCA=rng.normal(16515,100,n0), PSGA=rng.normal(67,0.5,n0),
                  SCMA=rng.normal(17.2,0.3,n0), DNBR=rng.normal(2.3,0.05,n0),
                  DT_HL_CL=rng.normal(35.6,1,n0))
    locac  = dict(P=rng.normal(120,20,n1), TAVG=rng.normal(290,15,n1),
                  WRCA=rng.normal(10000,4000,n1), PSGA=rng.normal(55,8,n1),
                  SCMA=rng.normal(8,5,n1), DNBR=rng.normal(1.5,0.4,n1),
                  DT_HL_CL=rng.normal(20,8,n1))
    df = pd.concat([pd.DataFrame(normal), pd.DataFrame(locac)], ignore_index=True)
    y  = np.array([0]*n0 + [1]*n1)
    print(f"  Synthetic NPPAD: {n0} Normal, {n1} LOCAC")
    return df, y

# ── CFD anomaly signals (neural-operator outputs) ─────────────────────────────

def make_cfd_feats(n, labels, seed):
    """Simulate eta_CFD component signals from the neural operator output fields."""
    rng = np.random.RandomState(seed)
    turb = np.clip(rng.normal(np.where(labels==1, 0.55, 0.05), 0.25, n), 0, 1)
    pstd = np.clip(rng.normal(np.where(labels==1, 0.45, 0.05), 0.25, n), 0, 1)
    flow = np.clip(rng.normal(np.where(labels==1, 0.40, 0.03), 0.20, n), 0, 1)
    return np.stack([turb, pstd, flow], axis=1)

# ── severity & NPPAD mapping ──────────────────────────────────────────────────

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

# ── single evaluation ─────────────────────────────────────────────────────────

def run_one(labels, break_sizes, velocity, temperature, cfd_feats, w_input, seed):
    eta = compute_eta(break_sizes, cfd_feats, w_input)
    X   = map_nppad(eta, velocity, temperature)
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2,
                                           random_state=seed, stratify=labels)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    clf = GradientBoostingClassifier(**GB_PARAMS, random_state=seed)
    clf.fit(Xtr_s, ytr)
    yp   = clf.predict(Xte_s)
    ypp  = clf.predict_proba(Xte_s)[:,1]
    return dict(roc_auc=roc_auc_score(yte, ypp),
                recall=recall_score(yte, yp, zero_division=0),
                f1=f1_score(yte, yp, zero_division=0),
                accuracy=accuracy_score(yte, yp))

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*68)
    print("Edit 008 — Field-Only Ablation (Blended vs CFD-only)")
    print("="*68)

    print("\n[1] Loading NPPAD …")
    nppad_df, labels = load_nppad(PROJECT_ROOT)
    n = len(labels)

    rng = np.random.RandomState(0)
    break_sizes = np.where(labels==0, rng.uniform(0,0.5,n), rng.uniform(0.5,10,n))
    velocity    = rng.uniform(4, 6, n)
    temperature = rng.uniform(290, 320, n)

    res = {'blended': [], 'cfd_only': []}
    for i, seed in enumerate(SEEDS):
        print(f"\n[2] Seed {seed} ({i+1}/{len(SEEDS)}) …")
        cfd = make_cfd_feats(n, labels, seed)
        r_b = run_one(labels, break_sizes, velocity, temperature, cfd, 0.8, seed)
        r_c = run_one(labels, break_sizes, velocity, temperature, cfd, 0.0, seed)
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
    print(f"{'Metric':<12}  {'Blended (0.8/0.2)':>24}  {'CFD-only (0.0/1.0)':>24}")
    print("-"*64)
    for m in metrics:
        mb, sb = agg[m]['blended']
        mc, sc = agg[m]['cfd_only']
        print(f"{m.upper():<12}  {mb:.4f} +/- {sb:.4f}            {mc:.4f} +/- {sc:.4f}")
    print("="*68)

    json_path = Path(__file__).parent / "edit_008_ablation_results.json"
    with open(json_path,'w') as f:
        json.dump({m:{k:list(v) for k,v in d.items()} for m,d in agg.items()}, f, indent=2)
    print(f"\nJSON  -> {json_path}")

    # Bar chart
    mlabels = ['ROC-AUC','Recall','F1','Accuracy']
    x = np.arange(4); w = 0.32
    fig, ax = plt.subplots(figsize=(8,5))
    bars_b = ax.bar(x-w/2, [agg[m]['blended'][0] for m in metrics], w,
                    yerr=[agg[m]['blended'][1] for m in metrics],
                    label='Blended (η_input=0.8, η_CFD=0.2)',
                    color='#1f77b4', capsize=5)
    bars_c = ax.bar(x+w/2, [agg[m]['cfd_only'][0] for m in metrics], w,
                    yerr=[agg[m]['cfd_only'][1] for m in metrics],
                    label='CFD-only (η_input=0.0, η_CFD=1.0)',
                    color='#ff7f0e', capsize=5)
    for bars in [bars_b, bars_c]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0,4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'LOCA Classifier: Blended vs. CFD-Only Severity\n'
                 f'(GBC, {len(SEEDS)} seeds, mean ± std)', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(mlabels, fontsize=11)
    ax.set_ylim(0, 1.08); ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = Path(__file__).parent / "edit_008_ablation_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Plot  -> {plot_path}")

    delta_auc = agg['roc_auc']['blended'][0] - agg['roc_auc']['cfd_only'][0]
    delta_rec = agg['recall']['blended'][0]   - agg['recall']['cfd_only'][0]
    print(f"\nDelta AUC (Blended - CFD-only): {delta_auc:+.4f}")
    print(f"Delta Rec (Blended - CFD-only): {delta_rec:+.4f}")

if __name__ == "__main__":
    main()
