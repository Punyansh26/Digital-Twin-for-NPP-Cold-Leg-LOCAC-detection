# Edit 013 — Complete TODO Resolution and Submission-Oriented Manuscript

**Date**: 2026-09-02  
**Requested by**: Authors (complete all paper TODOs)  
**Paper section**: Whole manuscript  
**Edit type**: integrated revision + evidence audit + reference correction  
**Status**: ready-for-review

---

## Context

The working revision retained unresolved experiment, literature, data,
calibration, runtime, and author-biography placeholders. Several surrounding
claims also exceeded the available evidence: the evaluated field data were
described as CFD-like evidence, the scalar velocity regularizer was described
as divergence, the custom Clifford readout was treated as equivariant, and
unlogged latency estimates were presented quantitatively.

The authors requested a self-contained `Journal/completed_todo/` deliverable
with replacement text in blue, completed placeholders removed, and experiments
expected to exceed 20--30 minutes left as explicit author-run commands.

---

## Evidence from Codebase and Literature

- `Journal/completed_todo/artifact_audit.json` records dataset shapes,
  checkpoint hashes/load compatibility, model parameter counts, NPPAD row
  counts, stored classifier metrics, figure properties, citation completeness,
  and absence of unresolved manuscript markers.
- `Journal/scripts/edit_010_table_vii_metrics.json` provides exact pooled and
  per-case DeepONetFourier metrics on all 300 held-out analytic synthetic cases.
- `results/models/baseline_deeponet_results.json` provides the matched-dataset
  unmodified DeepONet result and training protocol.
- `Journal/scripts/edit_008d_ablation_results.json` provides the five-seed
  translation-path diagnostic and exposes its label-conditioned input.
- Source inspection of `scripts/generate_mock_data.py`,
  `src/deeponet/sobolev_loss.py`, `scripts/train_deeponet.py`,
  `src/feature_translation/translator.py`, and the operator modules establishes
  the actual analytic generator, effective loss, scalar output interface,
  severity mapping, token-attention implementation, and dense Clifford readout.
- The NPPAD source was verified as the PCTRAN-simulated open dataset of Qi
  *et al.*, *Scientific Data* 9, 766 (2022), DOI
  `10.1038/s41597-022-01879-1`.
- The September 2, 2026 public-source novelty search found adjacent
  nuclear-neural-operator studies by Hossain *et al.* (2025) and Lee *et al.*
  (2026), but no directly matching Transolver/Clifford cold-leg
  surrogate-to-classifier study. The manuscript limits novelty accordingly.
- Three inherited nuclear-ML/digital-twin references whose exact records could
  not be verified were removed. A verifiable peer-reviewed nuclear-AI review
  (Huang *et al.*, *Heliyon*, 2023) replaces their broad background role.

---

## Integrated Resolution

The completed tracked manuscript is:

`Journal/completed_todo/paper.tex`

The compiled review copy is:

`Journal/completed_todo/paper.pdf`

All requested changes are integrated directly in blue. The resulting paper:

1. removes all unresolved TODO/input/verification markers;
2. reports exact artifact-backed values only;
3. distinguishes analytic synthetic fields, PCTRAN-simulated NPPAD records,
   unevaluated Fluent automation, and future full-fidelity evidence;
4. removes unsupported latency and speedup values;
5. treats Transolver and Clifford panels as qualitative, matched-scale
   checkpoint evidence rather than an accuracy ranking;
6. corrects the effective DeepONetFourier objective and the translator's
   sigmoid severity equation;
7. limits the Clifford claim to a grade-structured inductive bias;
8. identifies split/scaler leakage, class-conditioned translation, and lack of
   calibration without presenting classifier scores as uncertainty;
9. omits unverified biographies rather than inventing credentials; and
10. supplies a public-code and NPPAD data-availability statement.

---

## Validation

Commands completed successfully:

    conda run -n minor_proj python Journal/scripts/edit_013_completed_paper_audit.py
    cd Journal/completed_todo
    latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

The audit found 23 cited keys, 23 matching bibliography entries, five present
figure dependencies, no forbidden markers, and load-compatible state
dictionaries for the baseline, DeepONetFourier, Transolver-inspired, and
Clifford-algebra checkpoints. The final LaTeX log contains no undefined
references, missing citations, package warnings, or overfull boxes. All PDF
fonts are embedded.

---

## Long Run Not Started

The prepared five-seed operator/classifier sweep is expected to require several
hours and was not started under the agreed runtime limit:

    conda run -n minor_proj python Journal/scripts/edit_009B_multiseed_sweep.py

Its output is not required by the present evidence-bounded wording. If run, it
must be reviewed before any cross-architecture ranking is added.

---

## Notes / Caveats

1. The archived classifier pickle was created with scikit-learn 1.8.0 and was
   inventoried under 1.7.2; stored metrics remain readable, but executable
   reuse should use the originating version or retrain and archive a fresh
   artifact.
2. Before external submission, the authors must verify names, e-mail addresses,
   corresponding-author designation, and any journal-specific biography
   requirement. No personal facts were inferred.
3. A persistent DOI release would improve reproducibility but was not invented
   or presented as already available.
