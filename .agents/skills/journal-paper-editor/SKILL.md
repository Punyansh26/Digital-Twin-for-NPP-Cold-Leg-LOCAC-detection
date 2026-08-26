---
name: journal-paper-editor
description: >-
  Skill for handling professor/reviewer-requested edits and additions to the
  IEEE journal paper on the AP1000 Digital Twin LOCA detection project.
  Use this skill whenever a paragraph or request is given that relates to
  modifying, extending, or clarifying the journal paper. It provides the
  complete workflow, file templates, paper context, writing-style rules, and
  codebase search strategies needed to produce a numbered edit file.
---

# Journal Paper Edit Skill

## When to activate this skill

Activate this skill when:
- The user provides a paragraph/request from their professor or reviewer to add, change, or clarify something in the journal paper.
- The user asks you to answer a specific section's "[Input/confirmation required]" placeholder.
- The user asks you to run an experiment or extract a metric for inclusion in the paper.
- The user asks you to write a new subsection, table, figure caption, or reference entry for the paper.

---

## Project Context

**Paper title**: A Multi-Architecture Neural Operator Digital Twin: A Proof-of-Concept Surrogate Methodology for Cold-Leg LOCA Screening in AP1000 Nuclear Reactors  
**Venue**: IEEE Transactions on Reliability (2026)  
**Authors**: Punyansh Thakur, Anushka Paul, Ayush Kumar Bhadra, Vinay Kumar — IIIT Naya Raipur, India  
**PDF**: `Journal/Ver_2___A_Multi_Architecture_Neural_Operator_Digital_Twin_for_Real_Time_Cold_Leg_LOCAC_Detection_in_AP1000_Nuclear_Reactors__Copy_.pdf`  
**Plain-text extract**: `Journal/paper_full_text.txt` (use grep/search on this file)  
**Detailed AI context**: `Journal/CLAUDE.md` — READ THIS FILE for full paper summary, architecture details, open items, and writing style.

---

## Complete Workflow

### Step 1 — Understand the professor's request
- Read the input paragraph carefully.
- Identify: target paper section, edit type (addition/replacement/clarification/figure/metric/reference), and any open "[Input/confirmation required]" placeholder it addresses.
- Check `Journal/CLAUDE.md` Section 2 (Open Items table) to see if this matches a known item.

### Step 2 — Search the codebase for evidence
Use `grep`, `find`, and `view_file` on these key paths:

| What | Where |
|---|---|
| DeepONetFourier model | `src/deeponet/deeponet_fourier.py` |
| Transolver++ model | `src/operators/transolver_operator.py` |
| Clifford model | `src/operators/clifford_operator.py` |
| Sobolev + divergence loss | `src/deeponet/sobolev_loss.py` |
| Training loop | `src/deeponet/train.py` |
| Feature translation | `src/feature_translation/translator.py` |
| LOCA classifier | `src/accident_model/train_locac_model.py` |
| Inference | `src/inference/run_inference.py` |
| Configs | `configs/config.yaml`, `configs/model_config.yaml` |
| Results | `results/` |
| Mock data | `scripts/generate_mock_data.py` |
| NPPAD data | `data/nppad/operation_csv_data/` |

Also search `Journal/paper_full_text.txt` to locate the exact paragraph being addressed.

### Step 3 — Write a helper script (if needed)
- If the edit requires running code (ablation, metric extraction, plot regeneration), write a Python script.
- Save it as: `Journal/scripts/edit_NNN_<description>.py`
- Run it from the project root (`/run/media/rtx/Files/Code/Minor/minorProjimproved`).
- Capture all output and include it in the edit evidence.

### Step 4 — Draft the edit text
Follow the IEEE Transactions on Reliability writing style (see `Journal/CLAUDE.md` Section 3):
- Formal academic prose, third-person.
- Hedged language: "under the synthetic benchmark", "preliminary", "proof-of-concept", "to the best of our knowledge".
- Never overstate: always qualify synthetic results vs. full-fidelity validation.
- If values are unknown/unverified, flag inline as `[VERIFY: <what needs checking>]`.
- Use IEEE numeric citations: [1], [4], [5], etc.

### Step 5 — Save the edit file
**Path**: `Journal/edits/edit_NNN_<short_name>.md`
**Numbering**: NNN is zero-padded sequential (001, 002, 003 ...).
**Short name**: 2-4 words, lowercase, underscores (e.g., `speedup_reconciliation`, `reliability_review`).

Use this exact template:

```
# Edit NNN — <Short Descriptive Title>

**Date**: YYYY-MM-DD
**Requested by**: <Professor / Reviewer comment source>
**Paper section**: Sec X.Y — <Section Title>
**Edit type**: [addition | replacement | clarification | figure | metric | reference]
**Status**: [draft | ready-for-review | approved]

---

## Context

<What was asked, and why this edit is needed. Quote the professor's request and
the "[Input/confirmation required]" placeholder if applicable.>

---

## Evidence from Codebase / Literature

<What was found in the code, experiment logs, or literature.
Include file paths, line numbers, and script output.>

---

## Proposed Text

<Section: Sec X.Y — Subsection Name>

> [DROP-IN REPLACEMENT / ADDITION BEGINS]

<Exact IEEE-style text, ready to paste into LaTeX.>

> [DROP-IN REPLACEMENT / ADDITION ENDS]

---

## Notes / Caveats

<Assumptions, open questions, items the team must verify.>
```

### Step 6 — Report back
Provide a concise summary:
- What was requested.
- What was found in the code.
- What edit was written.
- What the team still needs to verify or run.

---

## Writing Style Quick-Reference

| Rule | Detail |
|---|---|
| Voice | Third-person academic ("We demonstrate...", "The results show...") |
| Hedging | Always scope to "synthetic benchmark", "proof-of-concept" |
| Equations | Numbered right-aligned (1), (2)... in LaTeX |
| Sections | ALL CAPS Roman: "II. R ELATED W ORK"; subsections "A. Name"; sub-subs "1) Name:" |
| Figures | "Fig. N: Caption sentence case." |
| Tables | Caption above, sentence case |
| Citations | IEEE numeric [N] at end of clause before period |
| Placeholders | Preserve "[Input/confirmation required...]" verbatim when writing near them |

---

## Open Items Quick-Reference

(From `Journal/CLAUDE.md` Section 2 — Open Items)

| # | Location | Action needed |
|---|---|---|
| 1 | Sec I | Reconcile speedup denominator (1 h vs 1-4 h) |
| 2 | Sec I | Novelty re-verification (literature search) |
| 3 | Sec III | Confirm (vx,vy,vz) internal prediction for divergence penalty |
| 4 | Sec III-A | Specify gradient reconstruction scheme for Sobolev loss |
| 5 | Sec II-D | Add reliability/POD/FAR/V&V/RELAP5 review |
| 6 | Sec V-A | Retrain baseline OR retitle column |
| 7 | Sec V-C | Confirm grade-selective output projection for equivariance |
| 8 | Sec VI | Run field-only ablation (eta_CFD only) |
| 9 | Sec VIII | Replace bound-style metrics with exact values; multi-seed |
| 10 | Sec VIII-B | Regenerate figures as lossless PNG |
| 11 | Sec IX-E | Mamba result or future-work statement |
| 12 | Sec IX-E | DDPM result or future-work statement |
| 13 | End | Data availability statement |
| 14 | Acknowledgment | NPPAD formal citation |
| 15 | Biographies | Student author bios |

---

## Constraints (do not violate)

- Never modify the PDF or LaTeX source directly without explicit instruction.
- Never fabricate experiment results — flag unknowns with [VERIFY: ...].
- All scripts go in `Journal/scripts/` only.
- All edit outputs go in `Journal/edits/` only.
- Always use `edit_NNN_` prefix for file names.
- Qualify all synthetic-benchmark results with appropriate caveats.
