# AP1000 LOCAC Digital Twin — Change Log

**Project:** AP1000 Nuclear Reactor LOCAC Digital Twin Prototype
**Upgrade Session Date:** March 9, 2026
**Baseline → Upgraded Architecture**

---

## Overview

The original repository contained a baseline DeepONet implementation for predicting AP1000 primary-loop flow fields (pressure, velocity, turbulence, temperature) and detecting Loss of Coolant Accidents (LOCAs). This document records every change made during the upgrade session.

---

## 1. New Source Modules Created

### 1.1 `src/deeponet/fourier_encoding.py` *(NEW)*

**Purpose:** Eliminates spectral bias in the trunk network by mapping spatial coordinates through random Fourier features before MLP processing.

**Key class:** `FourierFeatureEncoding(input_dim=3, mapping_size=256, scale=10.0, trainable=False)`

**Formula:**
$$\gamma(\mathbf{x}) = \left[\sin(2\pi B\mathbf{x}),\ \cos(2\pi B\mathbf{x})\right], \quad B \in \mathbb{R}^{3 \times 256}$$

- Input: `[..., 3]` spatial coordinates
- Output: `[..., 512]` Fourier features
- `trainable=False` → fixed random projection (Rahimi & Recht, 2007)

---

### 1.2 `src/deeponet/adaptive_activation.py` *(NEW)*

**Purpose:** Trainable-slope activations that allow the network to adapt its nonlinearity per layer, improving convergence speed.

**Key classes:**
- `AdaptiveGELU(n_units, init_a=1.0)` — computes $f(x) = \text{GELU}(a \cdot x)$ where $a$ is a learned scalar per unit
- `AdaptiveActivationLayer(in_features, out_features)` — drop-in `Linear + AdaptiveGELU` combo

Used in the first two hidden layers of `TrunkNetFourier`.

---

### 1.3 `src/deeponet/sobolev_loss.py` *(NEW)*

**Purpose:** Gradient-enhanced training loss that penalises errors in both function values and their spatial derivatives, improving prediction of sharp gradients.

**Key class:** `SobolevLoss(alpha=1.0, beta=0.1, use_autograd=True, fd_eps=1e-3)`

**Formula:**
$$\mathcal{L} = \alpha \cdot \text{MSE}(u_\text{pred}, u_\text{true}) + \beta \cdot \|\nabla u_\text{pred} - \nabla u_\text{true}\|^2$$

- Returns `(total_loss, components_dict)` — components contain `mse_loss`, `grad_loss`, `total_loss`
- Automatically falls back to finite differences when autograd is unavailable

---

### 1.4 `src/deeponet/deeponet_fourier.py` *(NEW)*

**Purpose:** Upgraded DeepONet with Fourier-encoded trunk network and adaptive GELU activations, replacing the original ReLU MLP trunk.

**Architecture:**

| Sub-network | Layers | Activation |
|---|---|---|
| `BranchNetFourier` | 3 → 128 → 256 → 256 | GELU |
| `TrunkNetFourier` | Fourier(512) → 256 → 256 → 256 | Adaptive GELU × 2, then Linear |

**Key class:** `DeepONetFourier`

- `forward(branch_input [B,3], trunk_input [N,3]) → [B, n_outputs, N]`
- `count_parameters()` → 1,451,012 params (reduced from original)
- `parameter_breakdown()` → per-sub-network breakdown
- `from_legacy_config(config)` → classmethod for backward compatibility with `configs/config.yaml`

---

### 1.5 `src/deeponet/deeponet_base.py` *(NEW)*

**Purpose:** Backward-compatibility re-export alias so any code importing `DeepONetBase` continues to work without modification.

```python
from src.deeponet.model import DeepONet as DeepONetBase
```

Exports: `DeepONetBase`, `BranchNet`, `TrunkNet`, `DeepONetLoss`

---

### 1.6 `src/deeponet/residual_multifidelity.py` *(NEW)*

**Purpose:** Residual multi-fidelity learning — combines a coarse RANS base model with a fine LES-correction residual model.

**Formula:** $u_\text{final} = u_\text{base} + u_\text{residual}$

**Key classes:**
- `MultiFidelityDeepONet(base_model, residual_model, freeze_base=False)`
  - `freeze_base_network()` / `unfreeze_all()`
  - `load_base_checkpoint(path)`
  - `count_parameters() → dict`
- `MultiFidelityTrainer`
  - `train_base_epoch()` — trains base model on coarse data
  - `train_residual_epoch()` — trains residual on `target − base_pred`

Activated by setting `use_residual_multifidelity: true` in `configs/model_config.yaml` (default: `false`).

---

### 1.7 `src/physics/divergence_penalty.py` *(NEW)*

**Purpose:** Physics-informed regularisation that enforces incompressibility ($\nabla \cdot \mathbf{u} = 0$) during training.

**Formula:** $\mathcal{L}_\text{div} = \lambda \|\nabla \cdot \mathbf{u}\|^2,\quad \lambda = 0.01$

**Key class:** `DivergencePenalty(weight=0.01, use_autograd=False)`

- `forward(predictions [B, n_fields, N], coords=None) → scalar`
- `compute_full_divergence_penalty(vx, vy, vz)` for full vector fields

---

### 1.8 `src/physics/wall_shear_calculator.py` *(NEW)*

**Purpose:** Computes wall shear stress (WSS) and corrosion risk assessment for the primary coolant pipe.

**Formula:** $\tau_w = \mu \dfrac{\partial u}{\partial y}$

**Key class:** `WallShearCalculator(dynamic_viscosity=8.5e-5, pipe_radius=0.35, wall_threshold_fraction=0.10)`

- `compute_wss(velocity_field, coords) → dict`
- `assess_corrosion_risk(wss_results) → dict` — returns `risk_level` (Low / Medium / High / Critical)
- `save_wss_map(wss_results, output_path)`

---

### 1.9 `src/operators/transolver_operator.py` *(NEW)*

**Purpose:** Transformer-based neural operator that tokenises the computational mesh and applies multi-head self-attention across tokens.

**Architecture:** `MeshEmbedding (N → M tokens)` → `TransformerTokenBlock × n_layers` → scatter back to mesh

**Key class:** `TransolverOperator(coord_dim=3, branch_dim=3, embed_dim=256, n_tokens=64, n_layers=4, n_heads=8, n_outputs=4)`

- `forward(branch_input [B,3], mesh_points [N,3]) → [B, n_outputs, N]`
- `from_config(config)` classmethod
- ~106K parameters

---

### 1.10 `src/operators/clifford_operator.py` *(NEW)*

**Purpose:** Rotationally-equivariant neural operator using Clifford (geometric) algebra $\text{Cl}(3,0)$.

**Key function:** `clifford_product_3d(a, b: [...,8]) → [...,8]` — full $\text{Cl}(3,0)$ product table

Multivector layout: $[1, e_1, e_2, e_3, e_{12}, e_{13}, e_{23}, e_{123}]$

**Key classes:**
- `CliffordLinear(in_channels, out_channels)` — linear layer in geometric algebra
- `CliffordNeuralOperator(coord_dim=3, branch_dim=3, n_channels=32, n_layers=4, n_outputs=4)`
  - `forward(branch_input [B,3], mesh_points [N,3]) → [B, n_outputs, N]`
  - `from_config(config)` classmethod

---

### 1.11 `src/temporal/mamba_operator.py` *(NEW)*

**Purpose:** Linear-time ($O(T)$) sequence model for LOCAC transient trajectories, based on State Space Models (SSM).

**Key classes:**
- `SelectiveSSM(d_model, d_state=16, d_conv=4)` — input-selective state space recurrence
- `MambaBlock` — SSM + residual skip
- `MambaTemporalOperator(state_dim=16, d_model=128, n_layers=4, d_state=16)`
  - `forward(flow_states [B,T,state_dim]) → [B,T,state_dim]`
  - `predict_sequence(initial_state [B,state_dim], n_steps) → [B,n_steps,state_dim]`

Scales $O(T)$ vs $O(T^2)$ for transformers — critical for long transient sequences.

---

### 1.12 `src/temporal/liquid_nn_sensor_model.py` *(NEW)*

**Purpose:** Processes irregular-timestep sensor time series and predicts reactor state + LOCAC risk probability.

Uses Closed-form Continuous-depth (CfC) neurons that solve the ODE exactly:

$$h(t + \Delta t) = h(t) \cdot e^{-\Delta t \cdot \tau} + \left(1 - e^{-\Delta t \cdot \tau}\right) \cdot g(x, h)$$

**Key classes:**
- `CfCCell(input_size, hidden_size)` — handles irregular $\Delta t$ from sensor timestamps
- `LiquidNNSensorModel(n_sensors=8, hidden_size=64, latent_dim=32, n_layers=2)`
  - `forward(sensor_series [B,T,n_sensors], timestamps [B,T]) → (latent [B,latent_dim], risk [B,1])`
  - `predict_risk(...)` convenience method

---

### 1.13 `src/generative/diffusion_turbulence_model.py` *(NEW)*

**Purpose:** DDPM-based generative model that produces stochastic turbulence realisations conditioned on mean flow, enabling uncertainty quantification.

**Key classes:**
- `DDPMScheduler(n_steps=1000, beta_start=1e-4, beta_end=0.02)` — cosine noise schedule
- `TurbulenceDenoisingUNet(n_fields=4, cond_dim=8)` — 1-D U-Net with `ResBlock1D` and conditioning
- `DiffusionTurbulenceModel(n_nodes=1000, n_fields=4, cond_dim=8, n_diff_steps=1000)`
  - `encode_condition(mean_fields [B,n_fields,N], tke_field [B,N]) → [B, cond_dim]`
  - `training_loss(x0, cond) → scalar`
  - `sample(mean_fields, tke_field, n_samples=5) → [n_samples, n_fields, N]`

---

### 1.14 `src/physics/__init__.py` / `src/operators/__init__.py` / `src/temporal/__init__.py` / `src/generative/__init__.py` *(NEW)*

Package init files for new sub-packages, enabling clean imports.

---

## 2. New Configuration Files

### 2.1 `configs/model_config.yaml` *(NEW)*

Controls operator selection and all architecture hyperparameters.

Key fields:

| Field | Default | Effect |
|---|---|---|
| `operator` | `deeponet_fourier` | Active neural operator |
| `use_sobolev_loss` | `true` | Enable Sobolev loss |
| `use_divergence_penalty` | `true` | Enable divergence physics penalty |
| `use_residual_multifidelity` | `false` | Enable multi-fidelity learning |
| `temporal_model` | `none` | Activate Mamba / LiquidNN |
| `diffusion_model` | `false` | Enable diffusion super-resolution |

Contains sections for: `fourier_deeponet`, `deeponet` (legacy), `transolver`, `clifford`, `mamba`, `liquid_nn`, `diffusion_turbulence`.

---

### 2.2 `configs/training_config.yaml` *(NEW)*

Shared training hyperparameters across all scripts.

Key fields: `sobolev_weight`, `divergence_weight`, `learning_rate`, `epochs`, validation test matrix, benchmarking config.

---

## 3. Updated Scripts

### 3.1 `scripts/train_deeponet.py` *(UPDATED)*

Replaced the original 19-line stub with a comprehensive 300+ line training pipeline.

**New class:** `UpgradedDeepONetTrainer`

Features added:
- **Composite loss:** MSE + Sobolev gradient loss + Divergence physics penalty (each individually togglable)
- **AMP mixed precision** training (`torch.amp.autocast`)
- **Gradient clipping** (`max_norm=1.0`)
- **Extended metrics per field:** Relative L2, R², MAE, Derivative-L2
- **Early stopping** with `patience=50`
- **ReduceLROnPlateau** scheduler
- **Benchmarking mode:** measures inference latency vs 1-h CFD reference and reports speedup

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--operator` | `deeponet_fourier` | `deeponet` or `deeponet_fourier` |
| `--epochs` | 500 | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--sobolev-weight` | `0.1` | Sobolev loss weight β |
| `--divergence-weight` | `0.01` | Divergence penalty weight λ |
| `--no-sobolev` | — | Disable Sobolev loss |
| `--no-divergence` | — | Disable divergence penalty |
| `--benchmark` | — | Run benchmark after training |

---

### 3.2 `scripts/run_inference.py` *(UPDATED)*

Replaced the original 20-line stub (which merely called `src.inference.run_inference.main()`) with a full 290-line upgraded inference pipeline.

**New class:** `DigitalTwinInference` (standalone, re-implemented in scripts layer)

Features added:
- Multi-operator support: loads `deeponet`, `deeponet_fourier`, `transolver`, or `clifford` from checkpoint
- Optional **diffusion turbulence refinement** (`--diffusion N` for N stochastic realisations)
- Optional **wall shear stress computation** (`--wss`)
- **Time-series mode** (`--mode time_series`) — runs over a parameter sequence and plots LOCAC probability vs time
- **Benchmark mode** (`--mode benchmark`) — measures inference latency, reports speedup vs CFD

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--operator` | `deeponet_fourier` | Neural operator to use |
| `--mode` | `single` | `single` / `time_series` / `benchmark` |
| `--velocity` | `5.0` | Coolant inlet velocity (m/s) |
| `--break-size` | `2.0` | Break size (% pipe diameter) |
| `--temperature` | `305.0` | Coolant temperature (°C) |
| `--diffusion` | `0` | Number of turbulence realisations |
| `--wss` | — | Compute wall shear stress |

---

## 4. New Scripts Created

### 4.1 `scripts/train_operator.py` *(NEW)*

Train the Transolver++ or Clifford Neural Operator as an alternative to DeepONet.

**Class:** `OperatorTrainer`

Features:
- Selects `TransolverOperator` or `CliffordNeuralOperator` via `--operator` flag
- Cosine Annealing LR schedule
- AMP mixed precision
- Extended per-field metrics
- Early stopping (patience=50)
- Saves to `results/models/transolver_best.pth` or `clifford_best.pth`

**CLI:** `python scripts/train_operator.py --operator transolver --epochs 500`

---

### 4.2 `scripts/train_diffusion.py` *(NEW)*

Two-stage training pipeline for the diffusion turbulence super-resolution model.

**Class:** `DiffusionTrainer`

Pipeline:
1. Load a pre-trained DeepONet (frozen)
2. Run forward passes on training data → compute residuals $\Delta = \text{target} - \text{prediction}$
3. Train DDPM denoising UNet to reconstruct $\Delta$ conditioned on mean-flow features

Features:
- Loads pre-trained operator checkpoint (requires `best_model.pth` from `train_deeponet.py`)
- Cosine Annealing LR schedule
- Gradient clipping
- `--demo` flag to generate sample realisations after training
- Saves to `results/models/diffusion_model.pth`

**CLI:** `python scripts/train_diffusion.py --epochs 100 --demo --n-samples 5`

---

## 5. Files Removed

| File | Reason |
|---|---|
| `scripts/create_gitkeep.py` | One-time setup utility. Zero imports anywhere. Already executed (`.gitkeep` files in place). |

---

## 6. Dependency Scan Results

A full dependency scan confirmed that all other legacy files are still actively used:

| Legacy File | Still Referenced By |
|---|---|
| `src/deeponet/model.py` | `deeponet_base.py`, `scripts/train_deeponet.py`, `scripts/run_inference.py`, `src/inference/run_inference.py`, `src/deeponet/train.py`, `src/deeponet/visualize.py` |
| `src/deeponet/train.py` | `scripts/train_deeponet.py`, `scripts/train_operator.py` (MetricsCalculator, EarlyStopping) |
| `src/deeponet/dataset.py` | All three training scripts, `src/deeponet/train.py` |
| `src/deeponet/visualize.py` | `run_pipeline.py` |
| `src/inference/run_inference.py` | `notebooks/analysis.ipynb` (imports DigitalTwinInference) |
| `src/preprocessing/prepare_deeponet_data.py` | `run_pipeline.py`, `src/deeponet/dataset.py`, `src/deeponet/train.py` |
| `fluent/automation/generate_simulations.py` | `scripts/generate_dataset.py`, `src/preprocessing/prepare_deeponet_data.py` |
| `scripts/generate_mock_data.py` | `run_pipeline.py`, `scripts/generate_dataset.py` |

---

## 7. Verification Results

### 7.1 Module Import Test — 21/21 OK

All new and existing source modules import without errors in the `digital_twin` conda environment.

### 7.2 Script Syntax Check — 8/8 OK

```
scripts/train_deeponet.py     OK
scripts/train_operator.py     OK
scripts/train_diffusion.py    OK
scripts/train_locac_model.py  OK
scripts/run_inference.py      OK
scripts/generate_dataset.py   OK
scripts/generate_mock_data.py OK
run_pipeline.py               OK
```

### 7.3 Model Forward Pass Smoke Test — All Passed

| Model | Output Shape | Parameters |
|---|---|---|
| DeepONetFourier | `[2, 4, 20]` | 1,451,012 |
| TransolverOperator | `[2, 4, 20]` | 106,072 |
| CliffordNeuralOperator | `[2, 4, 20]` | 1,796 |
| MambaTemporalOperator | `[2, 10, 8]` | 17,256 |
| LiquidNNSensorModel | latent `[2,8]`, risk `[2,1]` | 2,257 |
| SobolevLoss | scalar | — |
| DivergencePenalty | scalar | — |

### 7.4 End-to-End Pipeline Run

Successfully executed:

```
python run_pipeline.py --use-mock-data --skip-training
```

| Step | Result |
|---|---|
| Step 1: Generate 2000 mock CFD cases | ✅ |
| Step 2: Preprocess → HDF5 (1400/300/300 split, 25k mesh pts) | ✅ |
| Steps 3–5: Skipped (using saved `best_model.pth` + `locac_detector.pkl`) | ✅ |
| Step 6: Inference on CUDA — DeepONetFourier, 188ms, NORMAL (p=0.0000) | ✅ |

---

## 8. Final Repository Structure

```
minorProjimproved/
│
├── configs/
│   ├── config.yaml                      # Legacy config (unchanged)
│   ├── model_config.yaml                # NEW — operator selection + architecture
│   └── training_config.yaml             # NEW — training hyperparameters
│
├── scripts/
│   ├── generate_dataset.py
│   ├── generate_mock_data.py
│   ├── train_deeponet.py                # UPDATED — full UpgradedDeepONetTrainer
│   ├── train_operator.py                # NEW — Transolver / Clifford training
│   ├── train_diffusion.py               # NEW — DDPM turbulence training
│   ├── train_locac_model.py
│   └── run_inference.py                 # UPDATED — multi-operator + WSS + diffusion
│
├── src/
│   ├── deeponet/
│   │   ├── model.py                     # Original DeepONet (kept)
│   │   ├── train.py                     # Original trainer (kept)
│   │   ├── dataset.py                   # Dataset + dataloaders (kept)
│   │   ├── visualize.py                 # Visualisation (kept)
│   │   ├── fourier_encoding.py          # NEW — Fourier Feature Encoding
│   │   ├── adaptive_activation.py       # NEW — AdaptiveGELU
│   │   ├── sobolev_loss.py              # NEW — Sobolev gradient loss
│   │   ├── deeponet_fourier.py          # NEW — DeepONetFourier
│   │   ├── deeponet_base.py             # NEW — backward-compat alias
│   │   └── residual_multifidelity.py    # NEW — multi-fidelity learning
│   │
│   ├── physics/                         # NEW package
│   │   ├── divergence_penalty.py        # NEW — ∇·u = 0 penalty
│   │   └── wall_shear_calculator.py     # NEW — τ_w = μ ∂u/∂y
│   │
│   ├── operators/                       # NEW package
│   │   ├── transolver_operator.py       # NEW — mesh-tokenised Transformer
│   │   └── clifford_operator.py         # NEW — Cl(3,0) equivariant operator
│   │
│   ├── temporal/                        # NEW package
│   │   ├── mamba_operator.py            # NEW — Selective SSM (O(T))
│   │   └── liquid_nn_sensor_model.py    # NEW — CfC neurons for sensors
│   │
│   ├── generative/                      # NEW package
│   │   └── diffusion_turbulence_model.py # NEW — DDPM turbulence super-resolution
│   │
│   ├── accident_model/train_locac_model.py
│   ├── feature_translation/translator.py
│   ├── inference/run_inference.py
│   └── preprocessing/prepare_deeponet_data.py
│
├── cleanup_report.md
├── run_pipeline.py
└── CHANGES.md                           # This file
```

---

## 9. Full Pipeline (Post-Upgrade)

```
ANSYS Fluent / Mock Data
        ↓
scripts/generate_dataset.py     (or generate_mock_data.py)
        ↓
src/preprocessing/prepare_deeponet_data.py
        ↓
scripts/train_deeponet.py       [DeepONetFourier + Sobolev + Divergence + AMP]
                                 OR
scripts/train_operator.py       [Transolver / Clifford]
        ↓ (optional)
scripts/train_diffusion.py      [DDPM turbulence super-resolution]
        ↓
scripts/train_locac_model.py    [GradientBoosting / MLP LOCAC classifier]
        ↓
scripts/run_inference.py        [multi-operator + WSS + diffusion realisations]
        ↓
notebooks/analysis.ipynb        [visualisation + evaluation]
```
