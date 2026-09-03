# Completion Status

**Started:** 2026-09-02  
**Target:** submission-oriented tracked manuscript with new text in blue and no
unresolved TODO markers.

## Evidence already available

- Exact pooled DeepONetFourier metrics for 300 held-out analytic synthetic cases:
  `Journal/scripts/edit_010_table_vii_metrics.json`.
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
  equivariance, label-independent detection, multi-seed operator comparison,
  calibration/UQ, and temporal modeling are not claimed as completed results.
  They are expressed as bounded limitations or future experiments without
  action-note placeholders.
- Transolver and Clifford exact accuracy cells are removed because the available
  evidence is insufficient for a defensible comparative ranking.
- Author biographies are omitted rather than fabricated. Verified affiliation
  and contact information remains in the author block.
- Data availability uses the public repository plus corresponding-author access;
  no DOI/archive is invented.

## Long run not started

The full five-seed operator/classifier sweep is expected to take several hours,
so it is outside the agreed 20–30 minute autonomous-run limit. The prepared
command is:

```bash
conda run -n minor_proj python Journal/scripts/edit_009B_multiseed_sweep.py
```

This run is optional for the present honestly scoped paper, because unsupported
multi-seed comparison claims are removed. Its outputs would strengthen a later
revision and must be reviewed before insertion.

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
