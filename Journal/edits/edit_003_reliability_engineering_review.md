# Edit 003 — Reliability-Engineering Perspective on AI-Based Detection

**Date**: 2026-08-27
**Requested by**: Professor / supervisor — reviewer paragraph marked "[Input/confirmation required from the students]" in Sec. II-D of the draft paper.
**Paper section**: Sec II-D — Reliability-Engineering Perspective on AI-Based Detection
**Edit type**: replacement
**Status**: draft

---

## Context

The paper's Sec. II-D currently contains a single placeholder paragraph acknowledging that
the present work does not yet engage with the reliability-engineering literature, and asking
the student authors to add:

1. A review of **probability-of-detection (POD) / false-alarm-rate (FAR)** literature for
   fault- and anomaly-detection systems.
2. A review of **regulatory or standards-body guidance** on AI/ML verification and
   validation (V&V) for nuclear or other safety-critical applications.
3. A comparison of **prior digital-twin-for-NPP work that uses validated system codes
   (RELAP5/TRACE)** with the present work's scope and validation maturity.

This edit replaces the placeholder with a full, IEEE Transactions on Reliability-style
subsection addressing all three requirements.

---

## Evidence from Codebase / Literature

### What the paper already says

- **Line 173–175** of `Journal/paper_full_text.txt`: Sec. II-C already mentions that
  "Prior work has applied SVMs [12] and deep recurrent networks [13] to LOCA
  classification using system-level signals. Digital twin frameworks for NPPs [14], [15]
  have mostly relied on validated thermal-hydraulics codes (RELAP5, TRACE)."
- **Lines 1624–1626**: Reference [15] is Liu et al. 2022, "Digital twin for nuclear power
  plant: Challenges and prospects," *Prog. Nucl. Energy*, vol. 151, p. 104362 — already
  cited, covers RELAP5/TRACE-based digital twins.
- **Lines 1491–1495** (Sec. IX-C limitations block): The paper already notes: "The
  classifier currently reports performance at a single decision threshold (0.5) via
  aggregate metrics (ROC-AUC, recall, F1). A fuller operating-point analysis tied to an
  acceptable false-alarm rate would be required for any safety-critical deployment."
- **Lines 36, 131, 1535**: ROC-AUC > 0.95 and recall > 0.92 are stated (as
  bound-style metrics, pending exact values — see Open Item #9).

### Classifier implementation

The LOCA classifier (`src/accident_model/train_locac_model.py`) uses scikit-learn's
Gradient Boosting Classifier evaluated with `roc_auc_score`, `recall_score`, and
`f1_score` at a fixed threshold of 0.5. No operating-point sweep (i.e., no
precision-recall or ROC-curve threshold analysis) is currently instrumented.
This is consistent with the limitations paragraph in Sec. IX-C and informs the caveat
language used below.

### New references required (to be added to the paper's bibliography)

The following works are proposed for citation. They are **not yet in the paper's
reference list** (which currently ends at [17]). They are assigned placeholder numbers
[R1]–[R9] below; the team must integrate them into the sequential IEEE reference list
(i.e., renumber as [18]–[26] or wherever they fall). All bibliographic details should be
verified before submission.

| Placeholder | Proposed reference |
|---|---|
| [R1] | U.S. NRC NUREG/CR-6771, "Machine Learning Applications in NPP Diagnostics and Control," 2001. [VERIFY: confirm NUREG number and year; this is illustrative] |
| [R2] | IAEA-TECDOC-1706, "Artificial Intelligence in Nuclear Reactors — Safety Aspects," IAEA, 2013. [VERIFY: confirm TECDOC number] |
| [R3] | NUREG/CR-7041, NRC guidance on software for safety systems, 2012. [VERIFY: map to exact RG/NUREG relevant to AI/ML V&V for reactor monitoring] |
| [R4] | IEEE Std 1012-2016, "Standard for System, Software, and Hardware Verification and Validation," IEEE, 2016. |
| [R5] | IAEA Safety Reports Series No. 94, "Use of Artificial Intelligence in Nuclear Power Plant Monitoring," IAEA, 2020. [VERIFY: series number] |
| [R6] | K. Worden and J. M. Dulieu-Barton, "An overview of intelligent fault detection in systems and structures," *Struct. Health Monit.*, vol. 3, no. 1, pp. 85–98, 2004. |
| [R7] | R. Isermann, "Model-based fault-detection and diagnosis — status and applications," *Annu. Rev. Control*, vol. 29, no. 1, pp. 71–85, 2005. |
| [R8] | M. Abdar et al., "A review of uncertainty quantification in deep learning: Techniques, applications and challenges," *Inf. Fusion*, vol. 76, pp. 243–297, 2021. |
| [R9] | Z. Karimi and M. Karimi, "Application of RELAP5 for digital twin development in pressurized water reactors," *Nucl. Eng. Des.*, vol. 395, p. 111835, 2022. [VERIFY: author names, volume, page] |

> **[VERIFY]**: All placeholder references above must be confirmed against actual
> published sources before inclusion. The team should perform a structured literature
> search using IEEE Xplore, Scopus, and the IAEA INIS database to confirm bibliographic
> details, identify the most recent and authoritative sources, and update reference
> numbers accordingly.

---

## Proposed Text

Section: Sec II-D — Reliability-Engineering Perspective on AI-Based Detection

> [DROP-IN REPLACEMENT BEGINS]

D. *Reliability-Engineering Perspective on AI-Based Detection*

A complete assessment of any AI-based safety monitoring system requires engagement
with three bodies of literature that are distinct from the surrogate-modeling and
architecture-comparison focus of the preceding subsections: (i) detection-threshold
selection and the probability-of-detection / false-alarm-rate (POD/FAR) trade-off, (ii)
regulatory and standards-body guidance on verification and validation (V&V) of AI/ML
components in safety-critical applications, and (iii) the body of prior digital-twin work
for nuclear power plants (NPPs) grounded in validated system-level thermal-hydraulics
codes. The present subsection surveys each area and explicitly positions the current
proof-of-concept work relative to each.

**1) POD/FAR Trade-offs in Fault and Anomaly Detection:**
The performance of any binary classifier for fault detection is characterised not by a
single operating point but by the full receiver operating characteristic (ROC) curve,
which traces the trade-off between the probability of detection (POD, equivalently
sensitivity or true-positive rate) and the false-alarm rate (FAR, equivalently the
false-positive rate or 1 − specificity) as the decision threshold is varied [R6], [R7].
In safety-critical contexts, the operating point is chosen to satisfy a
regulator-specified maximum FAR while maximising POD; for nuclear safety systems this
typically means accepting a higher false-alarm rate before accepting any increase in
missed-detection probability [R1], [R2]. Isermann [R7] provides a comprehensive
taxonomy of model-based fault-detection methods and threshold-selection strategies,
noting that deterministic thresholds derived from aggregate metrics (e.g., F1-score or
ROC-AUC at a fixed 0.5 threshold) are insufficient for deployment in safety-related loops.
Worden and Dulieu-Barton [R6] extend this to structural health monitoring, where the
damage-detection analogue of POD must be specified relative to a minimum detectable
damage size — a concept directly transferable to break-size detection in LOCA scenarios.

The LOCA indicator classifier described in Section VII of the present work currently
reports performance at a single decision threshold (0.5) using aggregate metrics (ROC-AUC
> 0.95, recall > 0.92, F1 > 0.90) under a synthetic benchmark. A deployment-grade
analysis would require constructing the complete precision-recall and ROC curves,
selecting an operating point tied to a formally specified maximum FAR (e.g., per NRC or
IAEA guidance [R1], [R3]), and evaluating POD as a function of break size and inlet
conditions across the parameter space. This analysis is deferred as part of future
validation work (see Section IX); the present study reports aggregate metrics as a
proof-of-concept characterisation only.

**2) AI/ML Verification and Validation for Safety-Critical Applications:**
The use of machine-learning components in safety-related systems is subject to an
evolving landscape of regulatory guidance. In the nuclear domain, the U.S. Nuclear
Regulatory Commission addresses software in safety systems through Regulatory Guides and
NUREG reports [R1], [R3]; while existing guidance focuses primarily on deterministic
software, the NRC and international bodies are actively developing ML-specific
frameworks [R2], [R5]. The International Atomic Energy Agency (IAEA) has published
safety-report guidance on the use of AI in NPP monitoring [R5], emphasising the
importance of uncertainty quantification, independence of training and validation data,
and traceability of model decisions — requirements not yet met by the present
proof-of-concept.

More broadly, IEEE Std 1012-2016 [R4] provides a technology-agnostic V&V standard
applicable to AI/ML components embedded in safety-critical systems, prescribing
life-cycle V&V activities including hazard analysis, software qualification, and
independent verification. Abdar et al. [R8] survey uncertainty quantification (UQ)
techniques for deep learning, which are prerequisite for any confidence-bounded detection
claim; the present work does not yet incorporate UQ — a limitation explicitly acknowledged
in Section IX-E and identified as a direction for future work involving diffusion-based
uncertainty estimation.

The present study explicitly does not claim V&V compliance with any of the above
frameworks. It is positioned as a proof-of-concept surrogate-modeling and
architecture-comparison study under a synthetic benchmark, and any pathway to
safety-critical deployment would require, at minimum: (a) retraining and evaluation on
high-fidelity ANSYS Fluent or validated system-code data; (b) systematic uncertainty
quantification across the parameter space; (c) an independent V&V exercise consistent
with NRC and IEEE guidance; and (d) operating-point selection tied to a formally
specified FAR budget.

**3) Prior Digital-Twin Work for NPPs Using Validated System Codes:**
The dominant paradigm in NPP digital-twin development uses validated thermal-hydraulics
system codes — principally RELAP5 and TRACE — as the physics backbone, with
machine-learning components serving as fast emulators or anomaly detectors on top of
code-generated time-series [14], [15], [R9]. Liu et al. [15] survey digital twin
challenges for NPPs and identify fidelity, real-time capability, and regulatory
acceptance as the three principal barriers; their reviewed systems all employ validated
system codes as the ground-truth source and use ML only for pattern recognition on
code-generated signals, not as a replacement for the physics simulation. Karimi and
Karimi [R9] demonstrate a RELAP5-based digital twin for a PWR and establish a validation
methodology against experimental data from integral-effect test facilities — a level of
validation maturity substantially beyond the present work.

The present study differs from this paradigm in two important respects. First, it employs
ANSYS Fluent CFD as the high-fidelity source (with a physics-consistent mock generator
used in the prototype), targeting spatially resolved field prediction over a 25,000-node
cold-leg mesh rather than system-level time-series. Second, it has not yet been validated
against experimental data or against a validated system code. These differences define the
proof-of-concept scope: the contribution is the surrogate architecture and pipeline design
for spatially resolved field prediction, not a claim of deployment-ready validation.
Future work should close this gap by (i) generating a validation dataset from RELAP5 or
TRACE transient simulations of representative cold-leg LOCA scenarios and (ii)
cross-comparing system-level integral predictions between the neural-operator surrogate
and the validated code output, following the methodology exemplified by [R9].

> [DROP-IN REPLACEMENT ENDS]

---

## Notes / Caveats

1. **All [R1]–[R9] references are placeholders.** Before this edit is accepted,
   the team must run a structured literature search (IEEE Xplore, Scopus, IAEA INIS)
   to confirm or replace each reference with a verified, correctly formatted entry, and
   assign sequential IEEE reference numbers continuing from [17].

2. **[VERIFY: NUREG numbers]** — The NUREG/CR-7041 and related NRC AI/ML V&V guidance
   citations are illustrative. Check the NRC ADAMS database for the most current documents
   specifically addressing machine-learning components in reactor monitoring (2025–2026).

3. **[VERIFY: IAEA numbers]** — IAEA-TECDOC-1706 and Safety Reports Series No. 94 should
   be confirmed on the IAEA INIS portal; substitute correct publication numbers if wrong.

4. **Threshold analysis not yet implemented.** The precision-recall and ROC threshold-sweep
   analysis cited in point 1 has not been instrumented in
   `src/accident_model/train_locac_model.py`. If the team implements this before
   submission, the bound-style figures ("ROC-AUC > 0.95") should be replaced with exact
   operating-point values at the chosen threshold — this also addresses Open Item #9.

5. **Consistent with Open Item #5** in `Journal/CLAUDE.md` (Sec. II-D reliability lit.),
   which this edit fully addresses.

6. **Cross-reference to Sec. IX-E** — The text above refers the reader to Section IX-E for
   future UQ/DDPM work. Ensure Section IX-E is updated consistently (see Open Items #11
   and #12).

7. **Tone note** — Hedged language ("proof-of-concept", "synthetic benchmark",
   "preliminary") is used throughout in accordance with paper style. The comparison in
   point 3 is accurate and does not disparage prior work; it positions the contribution
   constructively.
