# Repository Cleanup Report

**Date:** 2026-03-09  
**Scope:** AP1000 LOCAC Digital Twin — Remove Obsolete Legacy Files  

---

## Summary

A full dependency scan was performed across all Python source files, YAML configs, shell/batch scripts, and Jupyter notebooks. Every file was checked for direct imports and keyword references before any deletion decision was made.

**Result: 1 file deleted. 0 files were incorrectly identified as unused.**

---

## Deleted Files

| File | Reason |
|------|--------|
| `scripts/create_gitkeep.py` | One-time setup utility. Zero imports or references anywhere in the codebase. Not required by any training script, notebook, config, or pipeline entrypoint. Functionality was already executed (`.gitkeep` files exist). |

### Verification method

```
grep -R "create_gitkeep" .   →  (no matches)
grep -R "gitkeep"            →  (matches only in .gitkeep files themselves, not imports)
```

---

## Files Retained (with justification)

The following files were **considered for deletion** but confirmed as still required:

| File | Used By |
|------|---------|
| `src/deeponet/model.py` | `deeponet_base.py`, `scripts/train_deeponet.py`, `scripts/run_inference.py`, `src/inference/run_inference.py`, `src/deeponet/train.py`, `src/deeponet/visualize.py` |
| `src/deeponet/train.py` | `scripts/train_deeponet.py`, `scripts/train_operator.py` (provides `MetricsCalculator`, `EarlyStopping`) |
| `src/deeponet/dataset.py` | `scripts/train_deeponet.py`, `scripts/train_operator.py`, `scripts/train_diffusion.py`, `src/deeponet/train.py` |
| `src/deeponet/visualize.py` | `run_pipeline.py` |
| `src/inference/run_inference.py` | `notebooks/analysis.ipynb` (imports `DigitalTwinInference` directly) |
| `src/preprocessing/prepare_deeponet_data.py` | `run_pipeline.py`, `src/deeponet/dataset.py`, `src/deeponet/train.py`, `src/deeponet/visualize.py` |
| `fluent/automation/generate_simulations.py` | `scripts/generate_dataset.py`, `src/preprocessing/prepare_deeponet_data.py` |
| `scripts/generate_mock_data.py` | `run_pipeline.py`, `scripts/generate_dataset.py` |

All new upgraded modules (Fourier DeepONet, Transolver, Clifford, Mamba, Liquid NN, Diffusion, Sobolev loss, physics penalties) were explicitly protected per specification and remain untouched.

---

## Dependency Scan Methodology

1. Listed every `.py`, `.yaml`, and `.ipynb` file in the repository (excluding `__pycache__`, `.git`).
2. For each candidate file, searched all other files for:
   - Module import paths (e.g. `from src.deeponet.model import`)
   - Relative imports (e.g. `from .model import`)
   - Class/function names exported by the module
3. A file was only deleted if **all** of the following were true:
   - No import references found anywhere
   - Not listed in configs
   - Not used by any notebook
   - Not an active pipeline script

---

## Note on `scripts/benchmark.py`

Rule 3 in the cleanup specification lists `scripts/benchmark.py` as a file to keep. This file does not exist in the repository. Benchmarking functionality is embedded directly in `scripts/train_deeponet.py` (`--benchmark` flag) and `scripts/run_inference.py` (`--mode benchmark`). No action taken.

---

## Post-Cleanup Integrity Verification

All 21 source modules imported successfully with zero errors:

```
src.deeponet.model                      OK
src.deeponet.train                      OK
src.deeponet.dataset                    OK
src.deeponet.visualize                  OK
src.deeponet.fourier_encoding           OK
src.deeponet.adaptive_activation        OK
src.deeponet.sobolev_loss               OK
src.deeponet.deeponet_fourier           OK
src.deeponet.deeponet_base              OK
src.deeponet.residual_multifidelity     OK
src.physics.divergence_penalty          OK
src.physics.wall_shear_calculator       OK
src.operators.transolver_operator       OK
src.operators.clifford_operator         OK
src.temporal.mamba_operator             OK
src.temporal.liquid_nn_sensor_model     OK
src.generative.diffusion_turbulence_model OK
src.feature_translation.translator      OK
src.accident_model.train_locac_model    OK
src.inference.run_inference             OK
src.preprocessing.prepare_deeponet_data OK

21/21 modules OK
```

All pipeline scripts compiled without syntax errors:

```
scripts/train_deeponet.py    OK
scripts/train_operator.py    OK
scripts/train_diffusion.py   OK
scripts/train_locac_model.py OK
scripts/run_inference.py     OK
scripts/generate_dataset.py  OK
scripts/generate_mock_data.py OK
run_pipeline.py              OK
```

---

## Full Pipeline Status

The complete pipeline remains intact and reproducible:

```
CFD simulations (fluent/)
    → scripts/generate_dataset.py        [dataset creation]
    → scripts/train_deeponet.py          [Fourier DeepONet + Sobolev + Divergence]
    → scripts/train_operator.py          [Transolver / Clifford operators]
    → scripts/train_diffusion.py         [Diffusion turbulence super-resolution]
    → scripts/train_locac_model.py       [LOCAC classifier]
    → scripts/run_inference.py           [full inference + WSS + optional diffusion]
    → notebooks/analysis.ipynb           [visualization + evaluation]
```
