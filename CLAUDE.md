# CLAUDE.md — AP1000 Digital Twin for LOCAC Detection

> This file is the primary reference for AI coding assistants (Claude, Gemini, etc.) working in this repository. Read it fully before making any changes.

---

## Project Identity

**Full name**: AP1000 Digital Twin for Cold-Leg Loss of Coolant Accident (LOCAC) Detection  
**Domain**: Nuclear safety / Physics-ML / Scientific computing  
**Status**: Research prototype — functional end-to-end, not certified for real plant operations  
**Author**: punyansh26  
**Language**: Python 3.8+

---

## One-Paragraph Summary

This system builds a digital twin of the AP1000 pressurized-water reactor's cold-leg piping. It replaces expensive ANSYS Fluent CFD runs (hours each) with neural operators that infer the same 4-field solution (pressure, velocity magnitude, turbulence kinetic energy, temperature) in **< 20 ms** — a >1000× speedup. The predicted flow field is then translated into plant-level signals that feed a machine-learning LOCAC classifier (gradient boosting / MLP). The full pipeline can run on synthetic mock data without ANSYS Fluent installed.

---

## Repository Layout

```
minorProjimproved/
│
├── run_pipeline.py             ← Master orchestrator (start here)
├── requirements.txt
├── configs/
│   ├── config.yaml             ← Geometry, mesh, training hyper-params
│   └── model_config.yaml       ← Active model version + arch hyper-params
│
├── src/
│   ├── core/                   ← Model registry & version labels
│   │   ├── model_factory.py
│   │   └── model_versions.py
│   ├── deeponet/               ← Tier-1 DeepONet family
│   │   ├── model.py            ← Legacy baseline DeepONet
│   │   ├── deeponet_fourier.py ← DEFAULT: Fourier-enhanced DeepONet
│   │   ├── deeponet_base.py    ← Backward-compat alias
│   │   ├── fourier_encoding.py ← Random Fourier Feature Encoding
│   │   ├── adaptive_activation.py ← Learnable-slope GELU
│   │   ├── sobolev_loss.py     ← Gradient-enhanced physics loss
│   │   ├── residual_multifidelity.py ← Multi-fidelity residual layer
│   │   ├── dataset.py          ← PyTorch Dataset for HDF5 data
│   │   ├── train.py            ← Training loop with AMP + early stopping
│   │   └── visualize.py        ← Contour / error heatmap plots
│   ├── operators/              ← Tier-2 operators
│   │   ├── transolver_operator.py  ← Transolver++ (Transformer over tokens)
│   │   └── clifford_operator.py    ← Clifford Neural Operator (Geometric Algebra)
│   ├── temporal/               ← Tier-3 temporal extensions (optional)
│   │   ├── mamba_operator.py   ← Mamba SSM for transient sequences
│   │   └── liquid_nn_sensor_model.py ← Liquid NN / CfC for irregular sensor data
│   ├── generative/             ← Tier-3 generative extensions (optional)
│   │   └── diffusion_turbulence_model.py ← DDPM for turbulence UQ
│   ├── preprocessing/
│   │   └── prepare_deeponet_data.py ← CSV → HDF5 with scalers
│   ├── feature_translation/
│   │   └── translator.py       ← CFD fields → NPPAD-style plant signals
│   ├── accident_model/
│   │   └── train_locac_model.py ← XGBoost / MLP LOCAC classifier
│   └── inference/
│       └── run_inference.py    ← End-to-end inference pipeline
│
├── scripts/                    ← Standalone entry points (called by run_pipeline.py)
│   ├── generate_mock_data.py
│   ├── generate_dataset.py     ← Fluent journal generation (requires Fluent)
│   ├── train_deeponet.py       ← Train Tier-1 models
│   ├── train_operator.py       ← Train Tier-2 models
│   ├── train_diffusion.py      ← Train diffusion turbulence model
│   ├── train_locac_model.py    ← Thin wrapper around src/accident_model
│   └── run_inference.py        ← CLI inference entry point
│
├── fluent/                     ← ANSYS Fluent automation (optional)
│   ├── journals/               ← Auto-generated .jou scripts
│   └── automation/
│       └── generate_simulations.py
│
├── data/                       ← Runtime data (git-ignored)
│   ├── fluent_raw/             ← Raw CFD CSV exports
│   ├── fluent_processed/
│   ├── deeponet_dataset/       ← HDF5 files consumed by dataset.py
│   └── nppad/                  ← NPPAD nuclear plant dataset (optional)
│
├── results/                    ← All outputs (git-ignored)
│   ├── models/                 ← Saved checkpoints (.pth, .pkl)
│   ├── plots/
│   ├── metrics/
│   └── predictions/
│
├── notebooks/                  ← Jupyter analysis notebooks
├── paper.tex                   ← LaTeX paper draft
│
└── docs (Markdown in root):
    ├── README.md
    ├── AI_ARCHITECTURES.md     ← Deep-dive on every model
    ├── CHANGES.md              ← Session-by-session changelog
    ├── QUICKSTART.md
    └── PROJECT_SUMMARY.md
```

---

## Model Architecture Tiers

The project uses a **three-tier hierarchy** of neural operators. Only one Tier-1/2 operator is active at a time; Tier-3 extensions are additive.

### Tier 1 — DeepONet Family (default)

| Model key | Class | File | Notes |
|---|---|---|---|
| `deeponet` | `DeepONet` | `src/deeponet/model.py` | Legacy baseline, ReLU MLPs |
| `deeponet_fourier` | `DeepONetFourier` | `src/deeponet/deeponet_fourier.py` | **Default**. Fourier trunk + AdaptiveGELU |

**DeepONet formula**: `G(u)(y) ≈ Σₖ bₖ(u) · tₖ(y) + b₀`
- Branch network encodes **physics parameters** → basis coefficients
- Trunk network encodes **spatial coordinates** → basis functions
- Output = dot product → 4 simultaneous fields

**DeepONetFourier** adds:
- `FourierFeatureEncoding` on trunk coordinates (eliminates spectral bias)
- `AdaptiveGELU` activations (learnable slope per unit)
- Sobolev loss (penalises spatial gradient errors)
- Divergence-free penalty (`‖∇·u‖²`)
- ~1.45M parameters (reduced from original)

### Tier 2 — Alternative Operators

| Model key | Class | File | Notes |
|---|---|---|---|
| `transolver` | `TransolverOperator` | `src/operators/transolver_operator.py` | Transformer over learned mesh tokens (Wu et al., ICML 2024) |
| `clifford` | `CliffordOperator` | `src/operators/clifford_operator.py` | Geometric-algebra multivector features |

Both are trained via `scripts/train_operator.py`.

### Tier 3 — Optional Extensions

| Extension | Class | File | Enable via |
|---|---|---|---|
| Mamba temporal | `MambaTemporalOperator` | `src/temporal/mamba_operator.py` | `enable_temporal_model: mamba` in model_config.yaml |
| Liquid NN / CfC | `LiquidNNSensorModel` | `src/temporal/liquid_nn_sensor_model.py` | `enable_temporal_model: liquid_nn` |
| Diffusion DDPM | `DiffusionTurbulenceModel` | `src/generative/diffusion_turbulence_model.py` | `enable_diffusion: true` or `--diffusion N` flag |

---

## Configuration System

### Priority order (highest wins)
1. **CLI flag** (`--model-version`, `--epochs`, etc.)
2. **`configs/model_config.yaml`** — model architecture + operator selection
3. **`configs/config.yaml`** — geometry, simulation params, training defaults

### `configs/config.yaml` — key sections

```yaml
geometry:
  pipe_diameter: 0.7          # meters — AP1000 cold-leg diameter

parameter_sweep:              # training dataset spans this space
  velocity:    [4.0, 6.0]    # m/s inlet velocity
  break_size:  [0.0, 10.0]   # % of pipe diameter
  temperature: [290, 320]    # °C coolant temperature

training:
  batch_size: 4               # tuned for 8 GB VRAM
  epochs: 2000
  learning_rate: 0.001
  mixed_precision: true       # AMP — RTX 4060 optimised
  num_workers: 0              # MUST stay 0 on Windows (multiprocessing bug)

locac_model:
  type: "gradient_boosting"   # or "neural_network"
```

### `configs/model_config.yaml` — key fields

```yaml
model_version: deeponet_fourier   # active operator
enable_temporal_model: false      # mamba | liquid_nn | false
enable_diffusion:      false      # true | false

use_sobolev_loss:     true
sobolev_alpha:        1.0         # MSE weight
sobolev_beta:         0.1         # gradient-error weight

use_divergence_penalty: true
divergence_weight:    0.01
```

---

## Common Workflows

### Full pipeline (mock data, no Fluent)
```bash
python run_pipeline.py --use-mock-data
# or select a specific operator:
python run_pipeline.py --use-mock-data --model-version transolver
```

### Step-by-step
```bash
python scripts/generate_mock_data.py
python src/preprocessing/prepare_deeponet_data.py
python scripts/train_deeponet.py --model-version deeponet_fourier --epochs 300
python src/deeponet/visualize.py
python scripts/train_locac_model.py
python scripts/run_inference.py --mode single
```

### Train Tier-2 operators
```bash
python scripts/train_operator.py --model-version transolver --epochs 300 --lr 5e-4
python scripts/train_operator.py --model-version clifford --batch-size 8
```

### Train diffusion turbulence model
```bash
python scripts/train_diffusion.py --epochs 200 --demo --n-samples 10
```

### Inference modes
```bash
python scripts/run_inference.py --mode single --velocity 5.0 --break-size 2.0
python scripts/run_inference.py --mode time_series --diffusion 10 --wss
python scripts/run_inference.py --mode benchmark --model-version transolver
```

---

## CLI Flag Reference

### `run_pipeline.py`
| Flag | Default | Description |
|---|---|---|
| `--use-mock-data` | False | Skip Fluent; use synthetic data |
| `--skip-training` | False | Use pre-saved models |
| `--model-version` | from config | `deeponet` \| `deeponet_fourier` \| `transolver` \| `clifford` |

### `scripts/train_deeponet.py`
| Flag | Default | Description |
|---|---|---|
| `--model-version` | from config | `deeponet` \| `deeponet_fourier` |
| `--epochs` | config | Training epochs |
| `--lr` | config | Learning rate |
| `--sobolev-weight` | 0.1 | β in Sobolev loss |
| `--divergence-weight` | 0.01 | λ for divergence penalty |
| `--no-sobolev` | False | Disable Sobolev loss |
| `--no-divergence` | False | Disable divergence penalty |
| `--benchmark` | False | Speed/accuracy benchmark after training |

### `scripts/train_operator.py`
| Flag | Default | Description |
|---|---|---|
| `--model-version` | required | `transolver` \| `clifford` |
| `--epochs` | 500 | — |
| `--lr` | 1e-3 | — |
| `--batch-size` | 16 | — |

### `scripts/run_inference.py`
| Flag | Default | Description |
|---|---|---|
| `--model-version` | from config | Any model including `mamba`, `diffusion` |
| `--mode` | `single` | `single` \| `time_series` \| `benchmark` |
| `--velocity` | 5.0 | Inlet velocity (m/s) |
| `--break-size` | 2.0 | Break size (% of diameter) |
| `--temperature` | 305.0 | Coolant temperature (°C) |
| `--diffusion` | 0 | Number of DDPM turbulence realisations |
| `--wss` | False | Compute wall shear stress |

---

## Data Flow

```
ANSYS Fluent / mock generator
        │ CSV files (25 000 nodes × 4 fields × 2000 simulations)
        ▼
src/preprocessing/prepare_deeponet_data.py
        │ HDF5: branch_input [N_sim, 3], trunk_input [N_pts, 3],
        │       targets [N_sim, 4, N_pts]
        │ Scalers saved as .pkl alongside HDF5
        ▼
src/deeponet/dataset.py  (DeepONetDataset — PyTorch Dataset)
        │
        ▼
Training loop (src/deeponet/train.py or scripts/train_operator.py)
        │ Saves: results/models/best_model.pth
        ▼
src/feature_translation/translator.py
        │ 4-field grid → 7 scalar plant signals:
        │   average_pressure, pressure_gradient, mass_flow_rate,
        │   max_turbulence, temperature_difference, pressure_drop, flow_change
        ▼
src/accident_model/train_locac_model.py
        │ XGBoost (default) or MLP → LOCAC probability
        │ Saves: results/models/locac_detector.pkl
        ▼
src/inference/run_inference.py  ←  also via scripts/run_inference.py
        │ Returns: predicted fields + LOCAC decision + timing
        ▼
results/plots/ , results/predictions/
```

---

## Physics Domain Knowledge

- **AP1000**: Westinghouse Gen III+ pressurized water reactor
- **Cold leg**: Pipe returning cooled coolant (~290–320°C) from steam generator to reactor core; nominal pressure ~15.5 MPa, velocity ~4–6 m/s, density ~720 kg/m³
- **LOCAC (Cold-Leg Loss of Coolant Accident)**: Rupture anywhere in the cold-leg loop; break sizes modelled as 0–10% of pipe diameter (0.7 m)
- **CFD setup**: Pressure-based steady solver, RNG k-ε turbulence, ~25 000 mesh nodes per simulation
- **NPPAD**: Nuclear Power Plant Accident Database — time-series sensor data used to train / validate the LOCAC classifier

The neural operator maps:
- **Input space**: (velocity, break_size, temperature) — 3 scalars
- **Output space**: function over (x, y, z) → (pressure, velocity_mag, TKE, temperature) — 4 fields on a mesh

---

## Key Classes & Entry Points

| Symbol | File | Purpose |
|---|---|---|
| `DeepONetFourier` | `src/deeponet/deeponet_fourier.py` | Default operator forward pass |
| `FourierFeatureEncoding` | `src/deeponet/fourier_encoding.py` | Random Fourier features for trunk |
| `AdaptiveActivationLayer` | `src/deeponet/adaptive_activation.py` | GELU with learnable slope |
| `SobolevLoss` | `src/deeponet/sobolev_loss.py` | Physics-informed gradient loss |
| `TransolverOperator` | `src/operators/transolver_operator.py` | Tier-2 transformer operator |
| `CliffordOperator` | `src/operators/clifford_operator.py` | Tier-2 geometric-algebra operator |
| `MambaTemporalOperator` | `src/temporal/mamba_operator.py` | Tier-3 SSM for time sequences |
| `LiquidNNSensorModel` | `src/temporal/liquid_nn_sensor_model.py` | Tier-3 CfC sensor fusion |
| `DiffusionTurbulenceModel` | `src/generative/diffusion_turbulence_model.py` | Tier-3 DDPM turbulence UQ |
| `FeatureTranslator` | `src/feature_translation/translator.py` | CFD fields → plant signals |
| `DeepONetDataset` | `src/deeponet/dataset.py` | HDF5 dataset loader |
| `get_model` | `src/core/model_factory.py` | Central model instantiation |
| `get_tier_label` | `src/core/model_versions.py` | Human-readable tier string |
| `main()` | `run_pipeline.py` | Full pipeline orchestrator |

---

## Saved Artifact Paths

| Artifact | Path |
|---|---|
| Best operator checkpoint | `results/models/best_model.pth` |
| LOCAC classifier | `results/models/locac_detector.pkl` |
| Data scalers | Alongside HDF5 in `data/deeponet_dataset/` |
| Training plots | `results/plots/training_curves.png` |
| Field comparison plots | `results/plots/field_comparison_*.png` |
| Error heatmaps | `results/plots/error_heatmap_*.png` |

---

## Performance Targets

| Metric | Target | Notes |
|---|---|---|
| DeepONet R² | > 0.90 | All 4 output fields |
| LOCAC classifier accuracy | > 0.90 | On held-out test set |
| ROC-AUC | > 0.95 | |
| Inference time | < 20 ms | Per single prediction on GPU |
| CFD speedup | > 1000× | vs. estimated 3600 s Fluent run |
| GPU VRAM (training) | < 8 GB | Tuned for RTX 4060 |

---

## Hardware & Environment

- **Recommended GPU**: NVIDIA RTX 4060 (8 GB VRAM) or better
- **Minimum GPU**: 6 GB VRAM (reduce `batch_size` in `config.yaml`)
- **CPU mode**: Supported but significantly slower
- **CUDA version**: cu118 or cu121
- **Python**: 3.8+

### Install
```bash
pip install -r requirements.txt
# PyTorch with CUDA (adjust cu version as needed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Key dependencies
| Package | Role |
|---|---|
| `torch >= 2.0` | Model training + AMP |
| `numpy < 2.0` | Array ops (pinned — avoid numpy 2.x breaking changes) |
| `h5py` | HDF5 dataset storage |
| `scikit-learn` | Scalers, metrics, MLP fallback |
| `xgboost` | Primary LOCAC classifier |
| `scipy` | Signal processing utilities |
| `matplotlib` / `seaborn` / `plotly` | Visualisation |
| `PyYAML` | Config loading |
| `tqdm` | Progress bars |

---

## Known Gotchas & Constraints

1. **`num_workers: 0` in config.yaml** — must remain 0 on Windows due to PyTorch multiprocessing issues with `spawn`. Safe to increase on Linux.

2. **`numpy < 2.0`** — `requirements.txt` pins `numpy>=1.24.0,<2.0.0`. Do NOT upgrade to numpy 2.x; several dependencies break silently.

3. **`KMP_DUPLICATE_LIB_OK=TRUE`** — set at the top of `run_pipeline.py` to suppress OpenMP duplicate library warnings on Windows. Don't remove it.

4. **Model checkpoint naming** — all operators save to `results/models/best_model.pth` by default. If switching operators mid-project, rename old checkpoints or they will be overwritten.

5. **HDF5 on Windows** — `h5py==3.11.0` is pinned for Windows in `requirements.txt`. The platform conditional handles this automatically.

6. **`--operator` flag is a legacy alias** — `scripts/run_inference.py` and `scripts/train_deeponet.py` accept both `--operator` and `--model-version`; always prefer `--model-version` in new code.

7. **Tier-3 models are not trained by `run_pipeline.py`** — the orchestrator only runs Tier-1/2. Train Mamba / Liquid NN / Diffusion models separately, then reference them via `--model-version diffusion` during inference.

8. **Mock data is synthetic** — `generate_mock_data.py` produces physically plausible but simplified data. For publication results, use actual Fluent CFD outputs.

9. **`data/` and `results/` are git-ignored** — regenerate via the pipeline or restore from external storage.

---

## Code Conventions

- **Module-level `__init__.py`**: Every `src/` subdirectory has one; import from the package, not by file path.
- **Config loading**: Use `PyYAML`; merge `config.yaml` and `model_config.yaml`; CLI flags override both.
- **Device handling**: Models and tensors must be explicitly moved to `device` (from `torch.cuda.is_available()`). Use `model.to(device)`.
- **AMP**: Use `torch.cuda.amp.autocast()` and `GradScaler` for training; already wired in `src/deeponet/train.py`.
- **Saving models**: Save full state dict with `torch.save(model.state_dict(), path)`. Always also save the config used to instantiate the model so it can be re-created.
- **Logging**: Use `print()` — no logging framework is set up. Keep output scannable.
- **Type hints**: Used in new modules (deeponet_fourier, operators). Maintain in any new code.

---

## Adding a New Operator

1. Create `src/operators/my_operator.py` — implement `MyOperator(nn.Module)` with `forward(branch_input, trunk_input)` returning `[B, n_outputs, N]`.
2. Register it in `src/core/model_factory.py` → `get_model()`.
3. Add a tier label in `src/core/model_versions.py` → `get_tier_label()`.
4. Add hyper-param block in `configs/model_config.yaml`.
5. Add to the `--model-version choices` list in `scripts/train_operator.py` and `scripts/run_inference.py`.
6. Update `AI_ARCHITECTURES.md` and this file.

---

## References

- Lu et al. (2021). "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." *Nature Machine Intelligence*.
- Tancik et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *NeurIPS*.
- Wu et al. (2024). "Transolver: A Fast Transformer Solver for PDEs on General Geometries." *ICML*.
- Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines." *NeurIPS*.
- AP1000 Design Control Document, Westinghouse Electric.
- NRC LOCA Analysis Guidelines.

---

> **Disclaimer**: This is a research prototype for educational use only. It is NOT certified for real nuclear plant safety operations. All predictions must be validated against certified simulation tools and experimental data.
