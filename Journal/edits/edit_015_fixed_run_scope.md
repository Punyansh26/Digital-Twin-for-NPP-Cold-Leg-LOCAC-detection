# Edit 015 — Fixed-Run Evidence Scope

**Date**: 2026-09-04  
**Requested by**: Authors (close the computationally infeasible repeated-seed sweep)  
**Paper section**: Abstract; Sec. VIII — Experimental Results; Sec. IX-E — Limitations; Conclusion  
**Edit type**: replacement + clarification  
**Status**: ready-for-review

---

## Context

The authors elected not to continue the five-seed neural-operator sweep because
the completed full-grid runs required approximately seven hours per
architecture and the shortened protocol still exceeded the available local
compute and thermal budget. This edit closes that experiment as a submission
task and removes wording that repeatedly presents it as unfinished work.

The revision does not imply that repeated training was performed. Instead, it
defines the paper as an artifact-reproducible proof-of-concept study: exact
metrics are tied to fixed archived runs and a fixed held-out split, and no
confidence interval, significance test, or cross-objective architecture ranking
is claimed.

---

## Evidence from Codebase / Literature

- `Journal/scripts/edit_010_table_vii_metrics.json` records exact
  DeepONetFourier metrics over all 300 held-out analytic synthetic cases.
- `Journal/completed_todo/evidence/seed_42_transolver_metrics.json` records the
  corresponding seed-42 Transolver-inspired full-test metrics.
- `results/models/baseline_deeponet_results.json` records the independently
  trained dataset-matched DeepONet baseline.
- `Journal/scripts/edit_008d_ablation_results.json` is already a five-seed
  translation-path diagnostic; its class-conditioned-input limitation remains
  explicit.
- The DeepONetFourier objective contains additional serialized-node proxy terms,
  while Transolver and Clifford use MSE. A strict numerical leaderboard would
  therefore remain confounded even if more seeds were available.
- The manuscript retains exact seeds, split scope, metric definitions,
  architecture sizes, objective differences, and data provenance. These are
  sufficient for fixed-run proof-of-concept reporting, while stochastic
  uncertainty is stated once as the scope of a larger comparative study.

No new numerical result is introduced by this edit.

---

## Proposed Text

<Section: Sec. VIII — Result Provenance and Scope>

> [DROP-IN REPLACEMENT BEGINS]

\textbf{Result provenance and scope:} Table~VII gives exact pooled metrics
recomputed from the archived DeepONetFourier checkpoint over all 300 held-out
analytic synthetic cases. Table~VIII reports the separately archived seed-42
Transolver-inspired metric record over the same full test set. Table~X gives
exact values stored with the fixed classifier artifact, and Table~XI provides a
five-seed translation-path diagnostic subject to its class-conditioned-input
limitation. Each numerical result is tied to an archived artifact, explicit
split, and stated metric definition. The architecture table therefore
summarizes verified fixed-protocol evidence without implying a statistically
controlled ranking across non-identical objectives.

> [DROP-IN REPLACEMENT ENDS]

<Section: Sec. IX-E — Limitations and Future Work>

> [DROP-IN REPLACEMENT BEGINS]

\textbf{Statistical scope}: The study prioritizes end-to-end method integration
and artifact-level reproducibility; field metrics therefore characterize fixed
archived runs, while the constrained translation-path diagnostic uses five
seeds. The reported values support feasibility on the prescribed split but are
not confidence intervals or tests of statistically significant differences
between architectures. Repeated training under matched objectives and compute
budgets is reserved for a larger comparative study.

> [DROP-IN REPLACEMENT ENDS]

The abstract, Transolver discussion, methodological-contribution paragraph, and
conclusion were harmonized with the same framing in
`Journal/completed_todo/paper.tex`. All replacement prose remains blue.

---

## Notes / Caveats

1. Fixed-run reporting is defensible because the evidence and evaluation scope
   are explicit; it must not be described as stochastic stability evidence.
2. Do not add standard deviations, confidence intervals, or significance claims
   without new repeated-run artifacts.
3. The optional sweep script remains available for a future larger-compute
   study, but it is not required to complete the present manuscript.
4. Full-fidelity CFD or experimental validation remains the materially more
   important limitation for any safety-relevant interpretation.
