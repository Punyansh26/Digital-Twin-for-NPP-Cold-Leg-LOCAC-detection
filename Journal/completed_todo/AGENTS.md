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
- `COMPLETION_STATUS.md` records evidence, resolved TODOs, excluded claims, and
  any work that still requires an author-run experiment.

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
- Do not report unevaluated Transolver/Clifford accuracy, exact equivariance,
  mass conservation, calibrated probability, predictive uncertainty, or a
  matched CFD speedup.
- Prefer removing an unsupported ranking or recasting it as future work over
  preserving a blocking experiment placeholder.
- The paper is a proof-of-concept methodology, not a certified plant system.

## Experiment policy

- Paper-supporting Python must run as
  `conda run -n minor_proj python <script>` from the repository root.
- Do not start work expected to exceed 20–30 minutes. Preserve the script and
  put the exact author command and expected outputs in `COMPLETION_STATUS.md`.
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
- The optional multi-seed sweep remains author-run because it exceeds the
  agreed 20--30 minute limit; its exact command is in
  `COMPLETION_STATUS.md`.

*Created and finalized: 2026-09-02 by Codex. Update this file if the manuscript
or evidence set changes.*
