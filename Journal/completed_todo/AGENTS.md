# AGENTS.md — Completed Journal Manuscript

> Scope: all files in `Journal/completed_todo/`. This directory contains the
> submission-oriented manuscript assembled from the canonical paper and the
> evidence-backed tracked revision. Root and `Journal/AGENTS.md` rules remain
> binding.

## Deliverable

- `paper.tex` is the editable final tracked manuscript.
- `paper.pdf` is the compiled review copy.
- `figures/` contains local, lossless figure dependencies so the manuscript is
  self-contained.
- `evidence/` contains exact metric records newly incorporated into the paper.
- `COMPLETION_STATUS.md` records evidence, resolved TODOs, excluded claims, and
  experiments deferred beyond the submitted proof-of-concept scope.

## Revision presentation

- Keep unchanged inherited prose black.
- Wrap new or replacement prose in `\added{...}`, which renders blue.
- Delete superseded wording and all completed placeholder/TODO markup.
- Do not add red, green, strikethrough, `[VERIFY]`, `[Input ...]`, or `TODO`
  text to the deliverable.
- Blue is a review aid; retain it until the authors accept the revision.

## Evidence and scientific framing

- Every numerical claim must name its provenance and scope: analytic synthetic
  fields, PCTRAN-simulated NPPAD records, or another explicit source.
- Never describe the analytic mock fields as Fluent CFD or plant data.
- Transolver accuracy may be reported only from the content-identical archived record
  `evidence/seed_42_transolver_metrics.json` (copied from
  `results/multiseed_sweep/20260902_231150/seed_42_transolver_metrics.json`), as
  a single-seed, MSE-only, full-test result on analytic synthetic data. Do not
  present it as a stable architecture ranking. Clifford full-test accuracy
  remains unevaluated after the recorded CUDA out-of-memory failure.
- Do not claim exact Clifford equivariance, mass conservation, calibrated
  probability, predictive uncertainty, or a matched CFD speedup.
- Prefer removing an unsupported ranking or recasting it as future work over
  preserving a blocking experiment placeholder.
- The paper is a proof-of-concept methodology, not a certified plant system.

## Experiment policy

- Paper-supporting Python must run as
  `conda run -n minor_proj python <script>` from the repository root.
- Do not start work expected to exceed 20–30 minutes. Preserve optional scripts,
  but do not leave a deferred experiment as an active submission requirement
  unless the authors explicitly elect to run it.
- Do not overwrite checkpoints or result artifacts. New outputs require a
  distinct, descriptive directory.

## Validation

- Compile from this directory so figure paths are local.
- Resolve LaTeX errors introduced by this revision.
- Before handoff, confirm that the source contains no unresolved TODO or input
  placeholders and that citations and references resolve on a second pass.

## Verified state

- `paper.tex` contains no unresolved TODO/input/verification markers.
- `artifact_audit.json` records the evidence inventory and dependency checks.
- `paper.pdf` compiles to 15 pages with embedded fonts and no undefined
  references, missing figures, package warnings, or overfull boxes.
- The repeated-seed sweep is closed without further training because it exceeds
  the available compute and thermal budget. The repaired script remains an
  optional future-study tool, but no manuscript claim depends on executing it.
  The paper reports fixed archived runs as artifact-traceable feasibility
  evidence and makes no significance or stochastic-stability claim.

*Created: 2026-09-02 by Codex. Updated: 2026-09-04 after closing the optional
repeated-seed experiment.*
