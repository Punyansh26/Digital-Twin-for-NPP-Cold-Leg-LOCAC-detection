# Edit 012 — UQ Future-Work Clarification

**Date**: 2026-08-31
**Requested by**: Professor (Section IX-E placeholder confirmation)
**Paper section**: Sec IX-E — Limitations and Future Work
**Edit type**: replacement + clarification
**Status**: ready-for-review

---

## Context

Section IX-E currently states that the pipeline produces point predictions and
then includes the following student-confirmation request:

> "[Input/confirmation required from the students: same as above — report a
> concrete UQ result if one exists, otherwise state this as unimplemented
> future work rather than ‘implemented.’]"

This edit resolves Open Item #12.  The codebase contains a prototype DDPM
component, but the repository contains no trained DDPM checkpoint, recorded
sampling result, predictive interval, calibration/coverage metric, or
uncertainty-propagated LOCA decision.  Therefore, no concrete UQ result can be
reported in the paper.

---

## Evidence from Codebase / Literature

- `configs/model_config.yaml:23-25` sets `enable_diffusion: false`; the second
  diffusion flag is also disabled at `configs/model_config.yaml:112-119`
  (`diffusion_model: false`).  DDPM is consequently not active in the reported
  default pipeline.
- `src/generative/diffusion_turbulence_model.py:222-332` defines a
  `DiffusionTurbulenceModel` prototype that can draw stochastic residual-field
  samples.  This establishes development intent, not an evaluated UQ result.
- `scripts/train_diffusion.py:130-205` supplies a training path and would save
  a model as `results/models/diffusion_model.pth`; its optional demonstration
  only prints sample tensor shape and aggregate standard deviation
  (`scripts/train_diffusion.py:209-227`).  It does not calculate predictive
  intervals, empirical coverage, calibration, or decision-level uncertainty.
- `scripts/run_inference.py:232-245` optionally returns raw
  `turbulence_samples`, but feature extraction and LOCA classification continue
  to use the original point-predicted `fields` (`scripts/run_inference.py:244-304`).
  Thus the prototype does not propagate uncertainty to the screening decision.
- Repository inspection on 2026-08-31 found no
  `results/models/diffusion_model.pth`, DDPM metric artifact, or UQ result log;
  `results/models/` contains only the DeepONet, Transolver, Clifford, and LOCA
  detector checkpoints.

---

## Proposed Text

<Section: Sec IX-E — Limitations and Future Work>

> [DROP-IN REPLACEMENT BEGINS]

\textbf{Uncertainty quantification}: The present pipeline produces point
predictions only and does not report predictive intervals, confidence bounds,
or coverage and calibration metrics. This is a significant limitation for a
screening tool intended to support safety-relevant decisions. Diffusion-based
stochastic turbulence realization and predictive uncertainty estimation are
unimplemented future work in the evaluated pipeline. Although an unevaluated
DDPM prototype is retained in the development codebase, no DDPM model has been
trained or evaluated on the present synthetic benchmark, and its samples have
not been converted into validated field-level or decision-level uncertainty
bounds. Future work should train and evaluate this component, propagate its
uncertainty through the feature-translation and LOCA-classification stages, and
report interval coverage and calibration before making confidence-bounded
screening claims.

> [DROP-IN REPLACEMENT ENDS]

**Integration instruction**: Replace the complete existing
`\textbf{Uncertainty quantification}:` paragraph, including its red
`[Input/confirmation required ...]` placeholder, with the text above.

---

## Notes / Caveats

1. This wording accurately distinguishes the unevaluated development prototype
   from a UQ capability in the reported pipeline; it makes no claim of an
   implemented or validated uncertainty bound.
2. The proposed future evaluation must use an independent test split and report
   at least nominal interval level(s), empirical coverage, interval width, and
   calibration.  Results remain subject to the paper's synthetic-benchmark
   limitation until replicated against high-fidelity or validated data.
3. If the students locate a trained `diffusion_model.pth` and reproducible UQ
   outputs elsewhere, this edit should be replaced by a results-backed revision
   that reports the exact protocol and measured values.
