"""
edit_008d_fixed_ablation.py  — FINAL corrected field-only ablation
===================================================================
Fixes from 008b:
  1. pstd_anomaly now uses relative pressure std (std_p / mean_p) with the
     correct baseline derived from normal-operation operator predictions
     (b=0% → rel_std = 0.0894%). Threshold set to that normal-op value.
  2. flow_deficit now uses the actual v_inlet from the normal-op prediction
     (~2.52 m/s) as the reference instead of 5.0 m/s.
  3. Threshold for turb_anomaly set from the actual b=0% max_k (0.709).

All three signals now have meaningful dynamic ranges across break sizes.

Run from project root:
    conda run -n minor_proj python Journal/scripts/edit_008d_fixed_ablation.py
"""

import sys, json, pickle
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SEEDS     = [42, 7, 13, 99, 2024]
GB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=20)

# ── Calibrated nominals (b=0%, v=5.0 m/s, T=305°C) ─────────────────────────
# Derived from diagnostic: edit_008c_diagnose.py
K_NOM         = 0.709    # max TKE at normal operation (m^2/s^2)
PREL_NOM      = 0.000894 # std_p / mean_p at normal operation (0.0894%)
V_INLET_NOM   = 2.519    # inlet-region velocity at normal operation (m/s)

# ── helpers ──────────────────────────────────────────────────────────────────
def load_cfg():
    with open(ROOT/"configs/config.yaml") as f: c = yaml.safe_load(f)
    with open(ROOT/"configs/model_config.yaml") as f: c.update(yaml.safe_load(f))
    return c

def load_operator(cfg, device):
    from src.deeponet.deeponet_fourier import DeepONetFourier
    m = DeepONetFourier(cfg).to(device)
    ck = torch.load(ROOT/"results/models/deeponet_fourier_best.pth",
                    map_location=device, weights_only=True)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m

def load_scalers():
    with open(ROOT/"data/deeponet_dataset/scalers.pkl","rb") as f: return pickle.load(f)

def load_trunk(sc, device):
    import h5py
    with h5py.File(ROOT/"data/deeponet_dataset/deeponet_dataset.h5",'r') as f:
        tn = torch.FloatTensor(f['train']['trunk'][:]).to(device)
    tp = sc['trunk'].inverse_transform(tn.cpu().numpy())
    return tn, tp

def denorm(data, fname, sc):
    return sc['targets'][fname].inverse_transform(data.reshape(-1,1)).flatten()

@torch.no_grad()
def predict(model, sc, tn, v, b, T, device):
    bn = sc['branch'].transform([[v,b,T]])
    bt = torch.FloatTensor(bn).to(device)
    out = model(bt, tn).cpu().numpy()[0]
    fnames = ['pressure','velocity_magnitude','turbulence_k','temperature']
    return {fn: denorm(out[i], fn, sc) for i,fn in enumerate(fnames)}

def cfd_anomaly(fields, trunk_phys):
    """Three CFD anomaly signals using calibrated normal-op thresholds."""
    x = trunk_phys[:,0]
    inlet = x < np.percentile(x, 10)

    max_k   = float(fields['turbulence_k'].max())
    std_p   = float(fields['pressure'].std())
    mean_p  = float(fields['pressure'].mean())
    v_inlet = float(fields['velocity_magnitude'][inlet].mean())

    turb_a = min(1.0, max(0.0, (max_k   - K_NOM)       / max(K_NOM,     1e-9)))
    pstd_a = min(1.0, max(0.0, (std_p/mean_p - PREL_NOM) / max(PREL_NOM, 1e-9)))
    flow_d = min(1.0, max(0.0, 1.0 - v_inlet / V_INLET_NOM))

    return np.array([turb_a, pstd_a, flow_d])

def build_lookup(model, sc, tn, tp, device):
    velocities   = [4.0, 4.5, 5.0, 5.5, 6.0]
    break_sizes  = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    temperatures = [290.0, 297.5, 305.0, 312.5, 320.0]
    total = len(velocities)*len(break_sizes)*len(temperatures)
    print(f"  Running operator on {total} grid points …", flush=True)
    recs = []
    for v in velocities:
        for b in break_sizes:
            for T in temperatures:
                a = cfd_anomaly(predict(model,sc,tn,v,b,T,device), tp)
                recs.append({'v':v,'b':b,'T':T,'ta':a[0],'pa':a[1],'fd':a[2]})
    df = pd.DataFrame(recs)
    print(f"  turb_anomaly  : {df['ta'].min():.4f} – {df['ta'].max():.4f}")
    print(f"  pstd_anomaly  : {df['pa'].min():.4f} – {df['pa'].max():.4f}")
    print(f"  flow_deficit  : {df['fd'].min():.4f} – {df['fd'].max():.4f}")
    return df

def _extract_nppad(df):
    o = pd.DataFrame()
    o['P']=df['P']; o['TAVG']=df['TAVG']; o['WRCA']=df['WRCA']
    o['PSGA']=df['PSGA']; o['SCMA']=df['SCMA']; o['DNBR']=df['DNBR']
    o['DT_HL_CL']=df['THA']-df['TCA']
    return o

def load_nppad():
    nd = ROOT/"data/nppad/operation_csv_data"
    rows,labels=[],[]
    for fp in sorted((nd/"Normal").glob("*.csv")):
        d=_extract_nppad(pd.read_csv(fp)); rows.append(d); labels.extend([0]*len(d))
    for fp in sorted((nd/"LOCAC").glob("*.csv")):
        d=_extract_nppad(pd.read_csv(fp)); rows.append(d); labels.extend([1]*len(d))
    df=pd.concat(rows,ignore_index=True); y=np.array(labels)
    print(f"  NPPAD: {int((y==0).sum())} Normal, {int((y==1).sum())} LOCAC")
    return df, y

def assign_cfd(labels, lookup, rng):
    n = len(labels)
    # Normal rows → b ∈ [0, 0.5%]; LOCAC rows → b ∈ [1, 10%]
    b = np.where(labels==0, rng.uniform(0,0.5,n), rng.uniform(1,10,n))
    v = rng.uniform(4,6,n); T = rng.uniform(290,320,n)
    A = np.zeros((n,3),dtype=np.float32)
    for i in range(n):
        dv=(lookup['v']-v[i])/2; db=(lookup['b']-b[i])/10; dT=(lookup['T']-T[i])/30
        A[i] = lookup.loc[(np.abs(dv)+np.abs(db)+np.abs(dT)).idxmin(),['ta','pa','fd']].values
    return A, b, v, T

def compute_eta(b, cfd, w):
    return np.clip(w*(b/10) + (1-w)*cfd.mean(axis=1), 0, 1)

def map_nppad_signals(eta, v, T):
    X = np.zeros((len(eta),7),dtype=np.float32)
    X[:,0]=155.5-eta*95; X[:,1]=T-eta*50
    X[:,2]=16515*(v/5)*(1-eta*0.6); X[:,3]=67-eta*30
    X[:,4]=35+eta*20-eta**2*60; X[:,5]=5.6+eta*130; X[:,6]=16*(1-eta)
    return X

def run_one(labels, b, v, T, cfd, w, seed):
    eta = compute_eta(b, cfd, w)
    X   = map_nppad_signals(eta, v, T)
    Xtr,Xte,ytr,yte = train_test_split(X,labels,test_size=0.2,random_state=seed,stratify=labels)
    sc=StandardScaler(); Xs=sc.fit_transform(Xtr); Xt=sc.transform(Xte)
    clf=GradientBoostingClassifier(**GB_PARAMS,random_state=seed)
    clf.fit(Xs,ytr); yp=clf.predict(Xt); ypp=clf.predict_proba(Xt)[:,1]
    return dict(roc_auc=roc_auc_score(yte,ypp), recall=recall_score(yte,yp,zero_division=0),
                f1=f1_score(yte,yp,zero_division=0), accuracy=accuracy_score(yte,yp))

def main():
    print("="*68)
    print("Edit 008d — FIXED Real-Operator Field-Only Ablation")
    print("="*68)
    cfg=load_cfg(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("\n[1] Loading operator …")
    model=load_operator(cfg,device); sc=load_scalers(); tn,tp=load_trunk(sc,device)
    print(f"    Calibrated nominals: K_NOM={K_NOM}, PREL_NOM={PREL_NOM:.6f}, V_INLET_NOM={V_INLET_NOM}")

    print("\n[2] Building CFD lookup (real operator, calibrated thresholds) …")
    lookup=build_lookup(model,sc,tn,tp,device)

    print("\n[3] Loading NPPAD …"); df,labels=load_nppad()

    res={'blended':[],'cfd_only':[]}
    print("\n[4] 5-seed evaluation …")
    for i,seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({i+1}/5) …")
        rng=np.random.RandomState(seed)
        cfd,b,v,T=assign_cfd(labels,lookup,rng)
        rb=run_one(labels,b,v,T,cfd,0.8,seed)
        rc=run_one(labels,b,v,T,cfd,0.0,seed)
        res['blended'].append(rb); res['cfd_only'].append(rc)
        print(f"  Blended  AUC={rb['roc_auc']:.4f} Rec={rb['recall']:.4f} F1={rb['f1']:.4f}")
        print(f"  CFD-only AUC={rc['roc_auc']:.4f} Rec={rc['recall']:.4f} F1={rc['f1']:.4f}")

    mkeys=['roc_auc','recall','f1','accuracy']
    agg={m:{'blended':(np.mean([d[m] for d in res['blended']]),
                        np.std( [d[m] for d in res['blended']])),
            'cfd_only':(np.mean([d[m] for d in res['cfd_only']]),
                         np.std( [d[m] for d in res['cfd_only']]))}
         for m in mkeys}

    print("\n"+"="*68)
    print(f"FINAL (real DeepONetFourier, calibrated thresholds)")
    print(f"{'Metric':<12} {'Blended (0.8/0.2)':>26} {'CFD-only (0.0/1.0)':>26}")
    print("-"*66)
    for m in mkeys:
        mb,sb=agg[m]['blended']; mc,sc_=agg[m]['cfd_only']
        print(f"{m.upper():<12} {mb:.4f}+/-{sb:.4f}              {mc:.4f}+/-{sc_:.4f}")
    print("="*68)
    d_auc=agg['roc_auc']['blended'][0]-agg['roc_auc']['cfd_only'][0]
    d_rec=agg['recall']['blended'][0] -agg['recall']['cfd_only'][0]
    print(f"Delta AUC={d_auc:+.4f}  Delta Rec={d_rec:+.4f}")

    # Save
    jpath=Path(__file__).parent/"edit_008d_ablation_results.json"
    with open(jpath,'w') as f:
        json.dump({m:{k:list(v) for k,v in d.items()} for m,d in agg.items()},f,indent=2)
    print(f"\nJSON  -> {jpath}")

    # Plot
    ml=['ROC-AUC','Recall','F1','Accuracy']
    x=np.arange(4); w=0.32
    fig,ax=plt.subplots(figsize=(8,5))
    b_vals=[agg[m]['blended'][0]  for m in mkeys]
    b_err =[agg[m]['blended'][1]  for m in mkeys]
    c_vals=[agg[m]['cfd_only'][0] for m in mkeys]
    c_err =[agg[m]['cfd_only'][1] for m in mkeys]
    bars_b=ax.bar(x-w/2,b_vals,w,yerr=b_err,label='Blended (η_in=0.8)',
                  color='#1f77b4',capsize=5)
    bars_c=ax.bar(x+w/2,c_vals,w,yerr=c_err,
                  label='CFD-only (η_in=0.0) — real DeepONetFourier',
                  color='#ff7f0e',capsize=5)
    for bars in [bars_b,bars_c]:
        for bar in bars:
            h=bar.get_height()
            ax.annotate(f'{h:.4f}',xy=(bar.get_x()+bar.get_width()/2,h),
                        xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    ax.axhline(0.95,color='red',linestyle='--',lw=1.2,label='Target (0.95)')
    ax.set_ylabel('Score',fontsize=12); ax.set_ylim(0.85,1.06)
    ax.set_title('LOCA Classifier: Blended vs. CFD-Only\n'
                 '(Real DeepONetFourier predictions, calibrated thresholds, 5 seeds)',fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(ml,fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y',linestyle='--',alpha=0.6)
    plt.tight_layout()
    ppath=Path(__file__).parent/"edit_008d_ablation_comparison.png"
    plt.savefig(ppath,dpi=300,bbox_inches='tight'); plt.close()
    print(f"Plot  -> {ppath}")

if __name__=="__main__":
    main()
