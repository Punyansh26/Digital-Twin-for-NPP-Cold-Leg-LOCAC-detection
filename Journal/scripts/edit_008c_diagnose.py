"""Quick diagnostic: what are the actual field ranges from the trained operator?"""
import sys, pickle, torch
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT/"configs/config.yaml") as f:    cfg = yaml.safe_load(f)
with open(ROOT/"configs/model_config.yaml") as f: cfg.update(yaml.safe_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.deeponet.deeponet_fourier import DeepONetFourier
model = DeepONetFourier(cfg).to(device)
ckpt  = torch.load(ROOT/"results/models/deeponet_fourier_best.pth", map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict']); model.eval()

with open(ROOT/"data/deeponet_dataset/scalers.pkl","rb") as f: scalers = pickle.load(f)

import h5py
with h5py.File(ROOT/"data/deeponet_dataset/deeponet_dataset.h5",'r') as f:
    trunk_norm = torch.FloatTensor(f['train']['trunk'][:]).to(device)
trunk_phys = scalers['trunk'].inverse_transform(trunk_norm.cpu().numpy())

def get_fields(v, b, T):
    branch = scalers['branch'].transform([[v,b,T]])
    bt = torch.FloatTensor(branch).to(device)
    with torch.no_grad(): out = model(bt, trunk_norm)
    out = out.cpu().numpy()[0]  # [4,N]
    fnames = ['pressure','velocity_magnitude','turbulence_k','temperature']
    return {fn: scalers['targets'][fn].inverse_transform(out[i].reshape(-1,1)).flatten()
            for i,fn in enumerate(fnames)}

print("\n=== Field ranges from real operator ===")
test_cases = [(5.0, 0.0, 305), (5.0, 2.0, 305), (5.0, 5.0, 305), (5.0, 10.0, 305)]
for v,b,T in test_cases:
    fields = get_fields(v,b,T)
    x_coords = trunk_phys[:,0]
    inlet = x_coords < np.percentile(x_coords, 10)
    v_inlet = float(fields['velocity_magnitude'][inlet].mean())
    std_p   = float(fields['pressure'].std())
    mean_p  = float(fields['pressure'].mean())
    max_k   = float(fields['turbulence_k'].max())
    print(f"\nb={b:4.1f}%: mean_p={mean_p:.1f} Pa  std_p={std_p:.1f} Pa  "
          f"rel_std_p={std_p/mean_p*100:.4f}%  max_k={max_k:.4f}  v_inlet={v_inlet:.4f} m/s")

    # Show what current formula gives
    turb_a = max(0.0, (max_k   - 0.65) / 0.65)
    pstd_a = max(0.0, (std_p   - 2.0)  / 2.0)
    flow_d = max(0.0, 1.0 - v_inlet / 5.0)
    print(f"         turb_anomaly={min(1,turb_a):.4f}  pstd_anomaly={min(1,pstd_a):.4f}  flow_deficit={flow_d:.4f}")

print("\n=== Normalised field stats (pre-denorm, scaler space 0-1) ===")
for v,b,T in test_cases:
    branch = scalers['branch'].transform([[v,b,T]])
    bt = torch.FloatTensor(branch).to(device)
    with torch.no_grad(): out = model(bt, trunk_norm)
    out = out.cpu().numpy()[0]
    x_coords = trunk_phys[:,0]
    inlet = x_coords < np.percentile(x_coords, 10)
    print(f"b={b:4.1f}%: norm_pstd={out[0].std():.6f}  norm_max_k={out[2].max():.6f}  norm_v_inlet={out[1][inlet].mean():.6f}")
