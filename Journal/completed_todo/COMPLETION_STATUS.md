# Completion Status

**Started:** 2026-09-02  
**Target:** submission-oriented tracked manuscript with new text in blue and no
unresolved TODO markers.

## Evidence already available

- Exact pooled DeepONetFourier metrics for 300 held-out analytic synthetic cases:
  `Journal/scripts/edit_010_table_vii_metrics.json`.
- Seed-42 Transolver-inspired pooled metrics for the same 300-case, 25,000-point
  held-out analytic synthetic split:
  `Journal/completed_todo/evidence/seed_42_transolver_metrics.json` (a
  content-identical archival copy of
  `results/multiseed_sweep/20260902_231150/seed_42_transolver_metrics.json`).
  Its four field R² values are 0.991775, 0.993163, 0.980636, and 0.995113
  (mean 0.990172). The MSE-only run used a 2,000-epoch budget and selected
  epoch 1,864 by validation loss.
- Dataset-matched unmodified DeepONet metrics:
  `results/models/baseline_deeponet_results.json`.
- Single-split classifier artifact and plot:
  `results/models/locac_detector.pkl` and
  `results/plots/locac_detection_performance.png`.
- Five-seed translation-path ablation:
  `Journal/scripts/edit_008d_ablation_results.json`.
- Lossless matched-scale architecture panels:
  `results/plots/paper_figures/*.png`.

## Resolution policy for remaining manuscript TODOs

- Matched Fluent timing, geometry-aware gradients/divergence, grade-selective
  equivariance, label-independent detection, repeated-seed operator comparison,
  calibration/UQ, and temporal modeling are not claimed as completed results.
  They are expressed as bounded limitations or future experiments without
  action-note placeholders.
- The single-seed Transolver accuracy record is reported without converting it
  into a stable cross-architecture ranking. Clifford exact accuracy remains
  omitted because the attempted full-grid batch-four run exhausted 8-GB GPU
  memory before training.
- Author biographies are omitted rather than fabricated. Verified affiliation
  and contact information remains in the author block.
- Data availability uses the public repository plus corresponding-author access;
  no DOI/archive is invented.

## Multi-seed experiment decision

The original sweep was partially run on 2026-09-02/03. DeepONetFourier seed 42
took 27,250.0 s, Transolver seed 42 took 24,252.3 s, Clifford seed 42 failed
with a CUDA out-of-memory error, and DeepONetFourier seed 7 was interrupted.
Inspection found three causes: a 2,000-epoch default, full 25,000-point training
steps, and an early-stopping call that checked a `None` return value and could
therefore never stop.

`Journal/scripts/edit_009B_multiseed_sweep.py` now defines a distinct shortened
protocol: at most 300 epochs, patience 30, 4,096 reproducibly sampled points per
training/validation step, full-grid evaluation, test batch size 1, explicit
protocol manifests, and resumable non-overwriting records. Results from this
protocol must not be pooled with the original 2,000-epoch records.
The authors have closed this experiment without further training because even
the shortened sweep exceeds the available local compute and thermal budget.
The repaired script is retained for future reproducibility work, but it is not
an active submission task and no paper claim depends on running it. The
manuscript instead reports the available fixed artifacts under an explicit
statistical-scope paragraph, avoids significance claims and cross-objective
rankings, and positions repeated matched-budget training as a larger follow-up
study.

## Current state

- [x] Working folder and local rules created.
- [x] Tracked manuscript and figures copied without overwriting prior versions.
- [x] Current literature and NPPAD provenance verified through public primary
  or peer-reviewed sources.
- [x] All manuscript TODO/input/verification markers resolved or removed.
- [x] Scientific, source-code, metric, citation, and language audits completed.
- [x] LaTeX compiled; no missing/undefined references, package warnings,
  overfull boxes, or missing figures remain.
- [x] Fifteen-page PDF visually reviewed; all fonts are embedded.

## Final verified artifact summary

- Audit command:
  `conda run -n minor_proj python Journal/scripts/edit_013_completed_paper_audit.py`
- Dataset shapes: train/validation/test = 1,400/300/300 cases, each with
  25,000 points and four retained scalar fields.
- Load-compatible checkpoint parameter counts:
  baseline 3,162,116; DeepONetFourier 1,451,012; Transolver-inspired operator
  3,244,872; Clifford-algebra operator 37,380.
- Stored classifier metrics: accuracy 0.977374, precision 0.987281, recall
  0.989063, F1 0.988172, ROC-AUC 0.990554.
- Stored seed-42 Transolver mean field R²: 0.990172; per-field range:
  0.980636--0.995113 on the analytic synthetic full test set.
- The artifact audit confirms that the archived and original Transolver JSON
  files parse to the same key-value record. Their byte hashes differ only
  because the archival copy has a terminating newline.
- Manuscript dependency audit: 23 cited keys, 23 bibliography entries, five
  present figures, and no unresolved markers.
- Build command:
  `cd Journal/completed_todo && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex`

## Literature audit

The public-source search was completed on 2026-09-02 with combinations of
`Transolver nuclear`, `Clifford neural operator reactor`,
`neural operator LOCA`, and `cold-leg LOCA neural operator`. It
identified adjacent work on nuclear virtual sensing (Hossain et al., 2025) and
an SMR steam-generator surrogate (Lee et al., 2026), but no directly matching
multi-operator cold-leg-surrogate-to-classifier workflow. The paper states this
as a scoped knowledge claim, not proof of universal absence.

NPPAD provenance and licensing were verified against the originating
*Scientific Data* article and Figshare collection. Three inherited citations
whose exact bibliographic records could not be verified were removed and
replaced, where needed, by a verifiable peer-reviewed nuclear-AI review.

## Reproducibility caveat

The archived classifier pickle was serialized with scikit-learn 1.8.0, while
the current `minor_proj` environment reports 1.7.2. The stored metrics are
readable and are explicitly reported as artifact-stored values. For executable
reuse, use the originating version or retrain and archive a fresh estimator.

## Author-owned submission checks

- Verify author names, e-mail addresses, affiliation, and corresponding-author
  designation.
- Confirm whether the selected journal requires author biographies; they were
  omitted rather than fabricated.
- Optionally create a tagged persistent/DOI archive for the exact split,
  checkpoints, metric JSON, and figure-generation code.
