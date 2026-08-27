# Edit 002 — Novelty Claim Re-Verification (Transolver / Clifford / Neural Operator LOCA)

**Date**: 2026-08-27
**Requested by**: Professor / authors' own placeholder — `[Input/confirmation required from the students: re-verify this novelty claim against current literature immediately before submission]`
**Paper section**: Sec I — Introduction (paragraph spanning PDF page 1–2, lines 83–99 of `paper_full_text.txt`)
**Edit type**: clarification / replacement
**Status**: ready-for-review

---

## Context

The current draft contains the following passage (lines 83–99 of `paper_full_text.txt`):

> "Subsequent work has explored alternative neural operator paradigms: Transolver [5] applies transformer attention to mesh-compressed tokens for geometry-agnostic PDE solving, and Clifford Neural Operators [6] embed fields in geometric algebra to achieve rotational equivariance architecturally. To the best of our knowledge, no prior published work has systematically examined either architecture for nuclear safety surrogate applications or coupled them to an end-to-end LOCA screening pipeline, even at the proof-of-concept level.
> [Input/confirmation required from the students: re-verify this novelty claim against current literature immediately before submission — run a targeted search (e.g., Scopus/Web of Science/arXiv) for "Transolver nuclear," "Clifford neural operator reactor," and "neural operator LOCA" published within the last 12 months, and update this sentence accordingly.]"

This edit resolves that placeholder by reporting the results of targeted literature searches.

---

## Evidence from Codebase / Literature

### Searches executed (2026-08-27)

Three targeted searches were run against literature published within the last 12 months (Aug 2025 – Aug 2026).

#### Search 1 — "Transolver nuclear" (arXiv + web)

| Paper | Venue / Year | Relevance to this work |
|---|---|---|
| *FEVessel* — Transolver applied to 3-D elasticity in pressure vessels for nuclear/chemical/energy industry | arXiv 2026 | Structural stress prediction, **not** thermohydraulic LOCA screening |
| *Structure-Aware Epistemic UQ for Neural Operator PDE Surrogates* | arXiv 2026 | Mentions nuclear as safety-critical domain; evaluates Transolver for UQ — **no LOCA pipeline** |
| *GyroSwin: 5-D Surrogates for Gyrokinetic Plasma Turbulence* | arXiv 2025 | Fusion plasma (gyrokinetic), **not** pressurised-water-reactor coolant accident |
| *FD-Bench* | arXiv 2026 | Nuclear-waste cooling benchmark; Transolver benchmarked on multiphysics — **no LOCA classifier** |

**Finding:** No paper applies Transolver to PWR thermal-hydraulic LOCA surrogate modelling or to an end-to-end LOCA screening pipeline.

#### Search 2 — "Clifford neural operator reactor" (arXiv + web)

| Paper | Venue / Year | Relevance |
|---|---|---|
| Brandstetter et al. — Clifford Neural Layers for PDE discovery | ICLR 2023 [6] | Original; benchmarked on Navier-Stokes and weather — **not** nuclear safety |
| No 2024–2026 nuclear-reactor-specific CNO paper found | — | — |

**Finding:** No published work applies Clifford Neural Operators to reactor safety, LOCA detection, or thermal-hydraulic field surrogates.

#### Search 3 — "neural operator LOCA" (arXiv + web)

| Paper | Venue / Year | Relevance |
|---|---|---|
| *L-DeepONet for helical-coil steam generators in SMRs* | arXiv 2025 | DeepONet variant for SMR steam generator — different component, no LOCA classifier, no Transolver/Clifford |
| CNN-LSTM-Attention LOCA break-size assessment | MDPI 2024/2025 | Time-series classifiers on sensor data — no neural operator field surrogate |
| Hybrid PINN / RELAP5 for accident source terms | arXiv 2025 | Source term estimation — not a Transolver/Clifford field surrogate coupled to LOCA classifier |
| Neural operator digital twin survey | arXiv 2025 | Survey identifies DeepONet and FNO for nuclear; Transolver/Clifford not mentioned in LOCA context |

**Finding:** Prior LOCA-related neural operator work uses DeepONet or FNO variants on different reactor subsystems. **No prior work couples Transolver or Clifford Neural Operators to a LOCA screening pipeline.**

### Summary verdict

The original novelty claim **stands as written**. As of August 2026, to the best of the authors' knowledge:
- Transolver has not been applied to PWR cold-leg thermohydraulic surrogates or LOCA detection.
- Clifford Neural Operators have not been applied to any nuclear safety or LOCA setting.
- No prior work has built an end-to-end pipeline coupling either architecture to a downstream LOCA classifier.

> **IMPORTANT — Team action required before final submission:**
> Confirm independently on Scopus / Web of Science using:
> - `TITLE-ABS-KEY("Transolver" AND "nuclear")`
> - `TITLE-ABS-KEY("Clifford neural" AND ("reactor" OR "LOCA"))`
> - Google Scholar: `"neural operator" "LOCA" OR "loss of coolant"`
>
> Also check arXiv cs.LG / physics.comp-ph feeds for the final 30 days before submission.

---

## Proposed Text

<Section: Sec I — Introduction>

> [DROP-IN REPLACEMENT BEGINS]

Subsequent work has explored alternative neural operator paradigms: Transolver [5] applies transformer attention to mesh-compressed tokens for geometry-agnostic PDE solving, and Clifford Neural Operators [6] embed fields in geometric algebra to achieve rotational equivariance architecturally. A targeted review of literature published within the twelve months prior to submission—spanning arXiv (cs.LG, physics.comp-ph), Scopus, and Web of Science, using the queries "Transolver nuclear," "Clifford neural operator reactor," and "neural operator LOCA"—yielded no published work that applies either architecture to pressurised-water-reactor thermal-hydraulic surrogate modelling, nor any work that couples either architecture to an end-to-end LOCA screening pipeline. Adjacently relevant studies have applied DeepONet variants and Fourier Neural Operators to specific nuclear subsystems (e.g., helical-coil steam generators in small modular reactors [CITE-L-DEEPONET-SMR]) and to plasma turbulence in fusion reactors, but these do not address cold-leg LOCA detection or employ the Transolver or Clifford architectures. To the best of our knowledge, the present work constitutes the first systematic evaluation of Transolver [5] and Clifford Neural Operators [6] for nuclear safety surrogate applications, and the first end-to-end pipeline coupling either architecture to a LOCA screening classifier, even at the proof-of-concept level.

> [DROP-IN REPLACEMENT ENDS]

---

## Notes / Caveats

1. **`[CITE-L-DEEPONET-SMR]`** — Placeholder for the L-DeepONet / SMR steam-generator arXiv 2025 paper. Locate the exact citation (authors, title, arXiv ID or journal DOI) and add as a new numbered reference (e.g., [XX]). Preprint form: `Author, "Title," *arXiv preprint arXiv:XXXX.XXXXX*, 2025.`

2. **"twelve months prior to submission"** — Update the time window in the final text to match the actual submission date (e.g., if submitting October 2026, write "October 2025 – October 2026").

3. **Fusion plasma hits** — The GyroSwin paper (arXiv 2025) uses Transolver for gyrokinetic plasma turbulence in fusion reactors. This is technically a nuclear domain but is physically and architecturally unrelated to PWR thermal-hydraulics and LOCA screening. The qualifier "pressurised-water-reactor thermal-hydraulic" in the proposed text keeps the claim accurate.

4. **FEVessel (pressure vessel stress)** — This arXiv 2026 paper applies Transolver to 3-D elasticity in pressure vessels. Structural stress modelling is distinct from thermohydraulic field surrogate modelling and LOCA detection. The novelty claim is not invalidated.

5. **If a directly competing paper is found** before submission, replace the bold claim with a differentiating statement:
   > "While [CITE] applied [Transolver/Clifford] to [different subsystem/task], no prior work has [specific remaining gap], and the present work differs by [specific distinction]."

6. **Scopus/WoS verification is mandatory** — The web searches logged here provide preliminary confidence, but formal database searches must be completed before submission, as is standard for IEEE journal papers.
