# Edit 009B — Multi-Seed Sweep Script & Reporting Infrastructure

**Date**: 2026-08-31
**Requested by**: Paper placeholder `[Input/confirmation required from the students]` — Sec. VIII
**Paper section**: Sec. VIII — EXPERIMENTAL RESULTS (Tables VII–IX)
**Edit type**: metric | addition
**Status**: superseded by Edit 014

> **Supersession note (2026-09-03):** The original implementation described
> below used 2,000 full-grid epochs, contained an inoperative early-stopping
> check, and triggered a Clifford CUDA out-of-memory error. It has been repaired
> with a distinct short-collocation protocol. See
> `edit_014_transolver_seed42_short_sweep.md` and do not use the historical
> runtime estimates or protocol description below as current instructions.

---

## Context

Edit 009 (`edit_009_result_stability.md`) supplied the required reporting
*protocol* and *drop-in LaTeX text* (with `[VERIFY]` tokens) for replacing the
single-run, bound-style metrics in Tables VII–IX with five-seed mean ± standard
deviation values.  That edit deliberately left all numerical cells as
`[VERIFY]` because no multi-seed run records existed at the time of writing.

Edit 009B provides the **executable infrastructure** that produces those records:
a self-contained Python sweep script that trains all three neural operators and
the LOCA classifier across the five canonical seeds and writes per-seed JSON
files plus an aggregated human-readable summary.  Once the team runs this
script, the `[VERIFY]` tokens in Edit 009 can be replaced with exact values.

The student confirmation placeholder being addressed is (Sec. VIII, page 9):

> "[Input/confirmation required from the students: before submission, (i)
> replace bound-style figures with exact measured values, and (ii) re-run all
> three architectures and the classifier across multiple random seeds (≥5
> recommended) and report mean ± standard deviation in place of single-run
> point estimates, so that reviewers can assess run-to-run variability.]"

---

## Evidence from Codebase

### Training scripts — no `--seed` flag exists

| Script | Gap identified |
|---|---|
| `scripts/train_deeponet.py:326–344` | CLI has no `--seed` argument; seed is not set before model construction |
| `scripts/train_operator.py:251–270` | Same omission |
| `src/accident_model/train_locac_model.py:38–47` | `random_state` hardcoded to `42` |

### Metric computation is present but not swept

- `scripts/train_deeponet.py:56–69` — `compute_extended_metrics()` computes
  per-field RelL₂, R², MAE, and derivative-L₂.
- `scripts/train_operator.py:157–174, 223–245` — `MetricsCalculator` used for
  validation and test evaluation but results are only printed, not persisted to
  JSON.
- `src/deeponet/train.py` — `MetricsCalculator.compute_all_metrics()` is the
  shared utility used by all three architectures.

### What the sweep script reuses

The sweep script (`Journal/scripts/edit_009B_multiseed_sweep.py`) wraps the
existing trainers directly, importing:
- `src.deeponet.deeponet_fourier.DeepONetFourier`
- `src.operators.transolver_operator.TransolverOperator`
- `src.operators.clifford_operator.CliffordNeuralOperator`
- `src.deeponet.sobolev_loss.SobolevLoss`
- `src.physics.divergence_penalty.DivergencePenalty`
- `src.deeponet.train.MetricsCalculator, EarlyStopping`
- `src.accident_model.train_locac_model.LOCACDetector`

All hyperparameters are loaded from `configs/config.yaml` and
`configs/model_config.yaml` without modification, ensuring the sweep
reproduces the paper's stated conditions exactly.

---

## Script: `Journal/scripts/edit_009B_multiseed_sweep.py`

Script is already written to disk.  Key design decisions:

| Decision | Rationale |
|---|---|
| Fixed seeds `{42, 7, 13, 99, 2024}` | Match the protocol in Edit 009 |
| Dataset split preserved (70/15/15, `random_state=42` inside `create_dataloaders`) | Only stochastic training components are varied, not the evaluation partition |
| Seed applied via `torch.manual_seed`, `np.random.seed`, `random.seed` before model construction | Reproducible weight initialization |
| DataLoader shuffle seeded via `torch.Generator` | Reproducible minibatch ordering |
| DeepONetFourier: Sobolev (β=0.1) + Divergence (λ=0.01) loss | Matches paper Sec. III |
| Transolver++/Clifford: MSE-only | Matches paper Sec. V-B/V-C |
| Classifier: `random_state=seed`, stratified 80/20 split | Seed controls both estimator and split |
| Per-seed JSON saved immediately after each run | Prevents data loss if later runs fail |
| Mean field R² computed per-run before aggregation | Avoids rounding artefacts from aggregated per-field values |
| `ddof=1` sample standard deviation | Correct for a 5-sample estimator |

### Running the script

```bash
# From project root:
conda run -n minor_proj python Journal/scripts/edit_009B_multiseed_sweep.py
```

Expected wall-clock time on an NVIDIA RTX 4060:
- DeepONetFourier × 5 seeds: ~2–4 h total (depends on early-stopping epoch)
- Transolver++ × 5 seeds: ~1–2 h total
- Clifford × 5 seeds: ~0.5–1 h total (tiny model, ~1,796 params)
- GBC classifier × 5 seeds: < 5 min total

### Output structure

```
results/multiseed_sweep/YYYYMMDD_HHMMSS/
├── seed_42_deeponet_fourier_metrics.json
├── seed_7_deeponet_fourier_metrics.json
├── seed_13_deeponet_fourier_metrics.json
├── seed_99_deeponet_fourier_metrics.json
├── seed_2024_deeponet_fourier_metrics.json
├── seed_42_transolver_metrics.json
│   ...  (same pattern for transolver and clifford)
├── seed_42_gbc_classifier_metrics.json
│   ...
├── aggregate_table_VII.json   ← DeepONetFourier per-field mean ± std
├── aggregate_table_VIII.json  ← All three arches mean-field R² mean ± std
├── aggregate_table_IX.json    ← Classifier metric mean ± std
└── sweep_summary.txt          ← Human-readable, LaTeX-formatted values
```

### Per-seed JSON schema (neural operators)

```json
{
  "seed": 42,
  "arch": "deeponet_fourier",
  "pressure_r2":        0.0000,
  "velocity_r2":        0.0000,
  "temperature_r2":     0.0000,
  "turbulence_k_r2":    0.0000,
  "pressure_rel_l2":    0.0000,
  ...
  "mean_field_r2":      0.0000,
  "best_val_loss":      0.0000,
  "best_epoch":         0,
  "elapsed_s":          0.0
}
```

### Per-seed JSON schema (classifier)

```json
{
  "seed":        42,
  "arch":        "gbc_classifier",
  "data_source": "nppad",
  "accuracy":    0.0000,
  "precision":   0.0000,
  "recall":      0.0000,
  "f1":          0.0000,
  "roc_auc":     0.0000
}
```

---

## Proposed Text (no new LaTeX — see Edit 009 for drop-ins)

Edit 009B does **not** introduce new LaTeX.  Its sole output is the measurement
infrastructure.  After running the script and verifying the aggregate JSONs, the
team should:

1. Open `Journal/edits/edit_009_result_stability.md`.
2. Replace every `\textit{[VERIFY: ...]}` token with the corresponding
   `mean ± std` value from `sweep_summary.txt`.
3. Change the **Status** field in Edit 009 from `draft` to `ready-for-review`.
4. Apply the four drop-in replacements from Edit 009 into the LaTeX source.

---

## Notes / Caveats

1. **Do not run until the dataset exists** — `data/deeponet_dataset/deeponet_dataset.h5`
   must be present.  If absent, run `conda run -n minor_proj python scripts/generate_dataset.py` first.
2. **Generator kwarg compatibility** — `create_dataloaders` may not expose a
   `generator` argument in all versions of the codebase.  The script falls back
   gracefully; the shuffle seed then has no effect, but the weight-init seed is
   still applied correctly.
3. **Data source consistency** — The classifier logs `"data_source": "nppad"` or
   `"synthetic"` in each seed JSON.  The five seeds must all use the **same** data
   source.  If any seed falls back to synthetic, note this in the paper footnote.
4. **Non-identical loss** — The architectural comparison in Table VIII remains
   preliminary because DeepONetFourier uses a harder Sobolev + divergence
   objective while Transolver++ and Clifford use MSE only.  This caveat must be
   preserved in the table footnote even after exact values are inserted.
5. **Archive the raw JSONs** — Store the dated `results/multiseed_sweep/` folder
   in a non-overwriting location before rerunning.  Git LFS or a shared drive is
   recommended.
6. **Independent verification** — Before submitting, at least one team member
   should independently recompute the mean ± std from the raw seed JSONs and
   confirm they match `sweep_summary.txt`.
