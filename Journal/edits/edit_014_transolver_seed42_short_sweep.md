# Edit 014 — Seed-42 Transolver Result and Shortened Multi-Seed Protocol

**Date**: 2026-09-03  
**Requested by**: Authors (insert completed Transolver result and reduce sweep runtime)  
**Paper section**: Sec. VIII — Experimental Results; Sec. IX — Discussion  
**Edit type**: metric + replacement + script correction  
**Status**: ready-for-review

---

## Context

The original five-seed script completed the seed-42 DeepONetFourier and
Transolver-inspired runs, then exhausted the 8-GB GPU while starting Clifford.
The next DeepONetFourier seed was manually interrupted because one run required
several hours. The authors requested immediate insertion of the completed
Transolver result and a shorter protocol for the remaining experiment.

---

## Evidence from Codebase / Experiment Record

- `results/multiseed_sweep/20260902_231150/seed_42_transolver_metrics.json`
  is valid JSON and records pooled metrics over the full held-out test loader:
  pressure R² 0.991775, velocity-magnitude R² 0.993163, turbulence-$k$ R²
  0.980636, and temperature R² 0.995113. Their arithmetic mean is 0.990172.
- A content-identical archival copy is retained with the completed manuscript
  at `Journal/completed_todo/evidence/seed_42_transolver_metrics.json`. The
  artifact audit verifies equality of the parsed key-value records; the archive
  adds only a terminating newline.
- The corresponding relative-$L_2$ errors are 0.022916, 0.044359, 0.089384,
  and 0.031441; normalized MAEs are 0.013808, 0.012909, 0.012279, and 0.013671.
- The record identifies seed 42, best validation MSE $3.03474\times10^{-4}$,
  best epoch 1,864, and elapsed time 24,252.3 s (6.74 h).
- `configs/config.yaml` supplied a 2,000-epoch general training limit to the
  original sweep. `Journal/scripts/edit_009B_multiseed_sweep.py` also called
  `EarlyStopping.__call__` as a Boolean, although the implementation in
  `src/deeponet/train.py` returns `None` and sets `early_stop`; consequently,
  the original sweep could not terminate through early stopping.
- `results/multiseed_sweep/20260902_231150/seed_42_clifford_metrics.json`
  records a CUDA allocation failure while requesting 3.05 GiB during the
  batch-four, full-25,000-point training path. It is an error record, not an
  accuracy result.
- The repaired sweep uses at most 300 epochs, 4,096 reproducibly sampled mesh
  points per training and validation step, batch-one full-grid testing, correct
  early-stopping state inspection, protocol manifests, and non-overwriting
  resume behavior. Full-test metrics still cover all 300 cases and all 25,000
  points.

---

## Proposed Text

<Section: Sec. VIII-C — Neural Operator Comparison>

> [DROP-IN ADDITION BEGINS]

Table~VIII reports the separately archived seed-42 Transolver-inspired metric
record over all 300 held-out analytic synthetic cases and all 25,000 mesh
points per case. The MSE-only run used a 2,000-epoch budget and retained the
minimum-validation-loss state from epoch 1,864. Pooled R² values were 0.991775,
0.993163, 0.980636, and 0.995113 for pressure, velocity magnitude, turbulent
kinetic energy, and temperature, respectively, giving a four-field mean of
0.990172. These values establish full-test accuracy for one initialization on
the fixed-mesh analytic benchmark; they do not establish run-to-run stability
or a controlled ranking against DeepONetFourier under its non-identical
objective.

> [DROP-IN ADDITION ENDS]

The text and exact table have been integrated in blue into
`Journal/completed_todo/paper.tex`.

---

## Notes / Caveats

1. Do not combine the 2,000-epoch seed-42 record with results from the new
   short-collocation protocol; training budgets and point-sampling protocols
   differ.
2. The complete shortened five-seed experiment is still expected to exceed
   20--30 minutes. The author-run start and resume commands are recorded in
   `Journal/completed_todo/COMPLETION_STATUS.md`.
3. The seed-42 Clifford error record supplies no accuracy evidence and is not
   reported as a paper result.
4. The shell error shown before the run was caused by a literal newline inside
   the quoted script path. Use the one-line commands in the completion-status
   file.
