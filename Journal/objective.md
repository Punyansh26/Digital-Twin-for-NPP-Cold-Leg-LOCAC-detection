# Objective — Journal Paper Edit Workflow

> **Project**: A Multi-Architecture Neural Operator Digital Twin for Cold-Leg LOCA Screening in AP1000 Nuclear Reactors  
> **Venue**: IEEE Transactions on Reliability (2026)  
> **Purpose of this file**: Document the complete task structure, workflow, and rules for handling professor-requested edits and additions to the journal paper.

---

## What This Task Is About

The professor (advisor/guide) reviews the paper and provides paragraph-form feedback — either a specific reviewer comment, a pointed question about the methodology, a request for a new section, or a clarification of an existing claim. The AI assistant's job is to:

1. Read the professor's input carefully.
2. Search the project codebase for implementation evidence.
3. Run any needed scripts (saved inside `Journal/scripts/`).
4. Produce a well-formatted, IEEE-style edit ready for the team to drop into the LaTeX source.
5. Save that edit in `Journal/edits/` with a sequential number and short name.

---

## The Paper at a Glance

| Item | Detail |
|---|---|
| **Paper** | Ver_2___A_Multi_Architecture_Neural_Operator_Digital_Twin_for_Real_Time_Cold_Leg_LOCAC_Detection_in_AP1000_Nuclear_Reactors__Copy_.pdf |
| **Domain** | Nuclear safety / Physics-ML / Scientific computing |
| **Core claim** | Three neural operator architectures (DeepONetFourier, Transolver++, Clifford) replace ANSYS Fluent CFD (1-4 h) with a surrogate that runs in < 20 ms on GPU |
| **Venue style** | IEEE Transactions on Reliability — formal academic, hedged, third-person |
| **Benchmark** | Synthetic, physics-consistent (NOT validated against real Fluent or plant data) |
| **Key limitation** | Results are upper-bound on a synthetic benchmark; full-fidelity validation is required before any deployment claim |

---

## Folder Layout for This Workflow

```
Journal/
|-- CLAUDE.md            <- AI context file (paper summary, style guide, workflow)
|-- objective.md         <- THIS FILE
|-- paper_full_text.txt  <- Plain-text extract of the PDF (for search/grep)
|
|-- scripts/             <- All helper Python/bash scripts created during edit work
|   |-- edit_001_*.py    <- Script for edit 001
|   |-- edit_002_*.py    <- Script for edit 002
|   `-- ...
|
`-- edits/               <- One .md file per professor-requested edit
    |-- edit_001_<name>.md
    |-- edit_002_<name>.md
    `-- ...
```

---

## Step-by-Step Workflow

### When the professor gives a paragraph/request:

**Step 1 — Parse the request**
- Identify the target section of the paper (e.g., Section II-D, Section V-A, Conclusion).
- Classify the edit type:
  - `addition` — new text must be written from scratch
  - `replacement` — existing paragraph must be rewritten
  - `clarification` — a specific claim needs more detail/evidence
  - `figure` — image needs regeneration or new plot
  - `metric` — experiment needs to be run and exact numbers reported
  - `reference` — new citations need to be added

**Step 2 — Search the codebase**
- Use `grep`, `find`, or read specific source files to locate evidence.
- Key paths to search (relative to project root):
  - `src/deeponet/` — DeepONetFourier model, loss, training
  - `src/operators/` — Transolver++ and Clifford operator
  - `src/deeponet/sobolev_loss.py` — Sobolev + divergence loss
  - `src/feature_translation/translator.py` — NPPAD feature mapping
  - `src/accident_model/train_locac_model.py` — LOCA classifier
  - `src/inference/run_inference.py` — end-to-end inference
  - `configs/config.yaml`, `configs/model_config.yaml` — hyperparameters
  - `results/` — saved metrics, plots
  - `data/nppad/operation_csv_data/` — NPPAD sensor data

**Step 3 — Write/run a script (if needed)**
- If evidence requires computation (e.g., running an ablation, computing a statistic, regenerating a figure), write a Python script.
- Save it as `Journal/scripts/edit_NNN_<descriptive_name>.py`.
- Run it from the project root; capture stdout/stderr.
- Include the script output in the evidence section of the edit file.

**Step 4 — Write the edit**
- IEEE style: formal, third-person, hedged appropriately.
- If adding text near a `[Input/confirmation required from the students: ...]` placeholder, quote that placeholder in the Context section and write text to replace it.
- Include all equations, table entries, or figure captions as needed.
- If the edit depends on values the team must verify, flag them clearly with `[VERIFY: ...]` inline.

**Step 5 — Save the edit file**
- Filename: `Journal/edits/edit_NNN_<short_name>.md` where NNN is zero-padded (001, 002, ...).
- Follow the template exactly (see Section 6 of CLAUDE.md).
- Set `Status: draft` initially.

**Step 6 — Report to the team**
- Provide a short summary: what was requested, what was found, what was written, what still needs team verification.

---

## Known Open Items in the Paper (Pre-existing)

These are the `[Input/confirmation required]` placeholders already in the paper that have NOT yet been addressed. Any professor request may relate to one of these:

| # | Section | Item |
|---|---|---|
| 1 | Sec I | Reconcile speedup denominator: is it exactly 1 h or "low end of 1-4 h range"? |
| 2 | Sec I | Re-verify novelty claim for Transolver/Clifford in nuclear domain (run Scopus/WoS search) |
| 3 | Sec III | Confirm that branch-trunk internally predicts (vx, vy, vz) vector components before reducing to scalar magnitude; if not, reformulate divergence penalty Eq.15 |
| 4 | Sec III-A | Specify which gradient reconstruction scheme (Green-Gauss, least-squares, k-NN finite difference) was used to compute target-field gradients for Sobolev loss |
| 5 | Sec II-D | Add dedicated review: (i) POD/FAR literature for fault/anomaly detection, (ii) AI V&V regulatory guidance for nuclear/safety-critical systems, (iii) prior NPP digital twins using RELAP5/TRACE |
| 6 | Sec V-A (Table I) | Either retrain unmodified DeepONet baseline under matched protocol OR retitle column "Baseline DeepONet (reference, not retrained)" |
| 7 | Sec V-C | Confirm MultivectorToFields is grade-selective (scalar outputs from grade-0, velocity from norm of grade-1); if not, soften Eq.28 equivariance claim |
| 8 | Sec VI | Run field-only ablation (eta_eff = eta_CFD only, zero weight on input break size) and report ROC-AUC, recall, F1 |
| 9 | Sec VIII | Replace bound-style figures ("> 0.90") with exact measured values; repeat all three architectures + classifier over >= 5 random seeds; report mean +/- std |
| 10 | Sec VIII-B | Regenerate Transolver++ and Clifford field panels as lossless PNG/PDF/EPS at matched resolution and color scale |
| 11 | Sec IX-E | Mamba temporal operator: report at least one concrete result OR move to forward-looking future-work statement (remove "implemented" language if not evaluated) |
| 12 | Sec IX-E | DDPM uncertainty quantification: same — report a result or state as unimplemented future work |
| 13 | Data Availability | Write data availability statement (GitHub repo link or "available from corresponding author") + NPPAD redistribution terms |
| 14 | Acknowledgment | Add NPPAD originating publication as formal bibitem and cite it at first use in Sec VI |
| 15 | Biographies | Write 2-4 sentence bios for Punyansh Thakur, Anushka Paul, Ayush Kumar Bhadra |

---

## Edit Numbering Convention

```
edit_001_<name>.md   — First edit received
edit_002_<name>.md   — Second edit received
...
edit_NNN_<name>.md   — N-th edit received
```

Names should be short (2-4 words, underscores, lowercase), e.g.:
- `edit_001_speedup_reconciliation.md`
- `edit_002_reliability_review.md`
- `edit_003_field_ablation_results.md`

---

## Rules & Constraints

- **Do not** modify the PDF or LaTeX source directly unless explicitly told to.
- **Do not** fabricate experimental results. If a metric is unknown, flag it with `[VERIFY: run experiment]`.
- **Always** cite with IEEE numeric format when adding references.
- **Always** qualify synthetic-benchmark results with the appropriate caveat ("under the synthetic benchmark", "proof-of-concept", etc.).
- **Store all scripts** in `Journal/scripts/` — never in tmp or Desktop.
- **Store all edits** in `Journal/edits/` — never anywhere else.
- **Name files** using the `edit_NNN_` convention — never free-form names.

---

*Created: 2026-08-27 by Antigravity AI assistant*
