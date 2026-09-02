---
name: research-todo
description: >-
  Skill for completing paper TODOs — the user pastes one or more TODO items
  from the paper and the agent resolves each one to unblock submission.
  TODOs may require: literature search, codebase inspection, running a small
  demo/script, drafting replacement paper text, or a combination. The skill
  enforces IEEE writing style, paper-safe framing, and the project's conda
  environment. Skips TODOs that have no impact on paper submission.
---

# Research TODO Skill

## Purpose

The user will paste TODO items extracted from the paper (the `[Input/confirmation required ...]` red-text placeholders, or informal "TODO" notes). This skill resolves them **as efficiently as possible** so the paper can be submitted. The output is either:

1. **Ready-to-paste LaTeX text** (replacing the `[Input/confirmation required …]` block in the `.tex` source), **or**
2. **A concrete script** the user should run (with instructions), **or**
3. **A clear "skip — irrelevant" decision** with a one-sentence rationale.

---

## Decision Tree for Each TODO

```
Received TODO item
      |
      ├── Is it a literature/novelty check?
      │       └── YES → search_web / read_url_content → draft replacement sentence
      |
      ├── Is it a code-clarification (confirm implementation detail)?
      │       └── YES → grep_search / view_file → confirm or soften claim
      |
      ├── Is it a metric/result that must be measured?
      │       ├── Can it be run as a quick demo (<5 min, no GPU required)?
      │       │       └── YES → write script → run via conda → paste numbers
      │       └── Needs full training run (hours)?
      │               └── Write script, tell user to run it, provide placeholder text
      |
      ├── Is it figure regeneration?
      │       └── Write/run a plotting script; embed lossless PNG or tell user
      |
      ├── Is it boilerplate text (bio, data-availability, acknowledgment)?
      │       └── Draft directly without running anything
      |
      └── Is it irrelevant / won't block submission?
              └── SKIP — state why in one sentence
```

---

## Activation Rules

Activate this skill when:
- The user pastes one or more TODO items from the paper (verbatim or paraphrased).
- The user says "complete the paper" or "finish the TODOs" without pasting specific items.
- The user asks you to resolve `[Input/confirmation required]` placeholders.

---

## Paper Context (Quick Reference)

| Field | Value |
|---|---|
| **Title** | A Multi-Architecture Neural Operator Digital Twin: A Proof-of-Concept Surrogate Methodology for Cold-Leg LOCA Screening in AP1000 Nuclear Reactors |
| **Venue** | IEEE Transactions on Reliability (2026) |
| **LaTeX source** | `Journal/Ver_2___A_Multi_Architecture_Neural_Operator_Digital_Twin_for_Real_Time_Cold_Leg_LOCAC_Detection_in_AP1000_Nuclear_Reactors__Copy_.tex` |
| **Plain text** | `Journal/paper_full_text.txt` |
| **Full AI context** | `Journal/CLAUDE.md` |
| **Existing edits** | `Journal/edits/edit_NNN_*.md` — check before creating a new one |
| **Scripts** | `Journal/scripts/` |

### Open TODO List (from paper `[Input/confirmation required]` blocks)

| # | Section | TODO | Covered by existing edit? |
|---|---|---|---|
| 1 | Sec I | Reconcile speedup denominator (1 h vs 1–4 h) | edit_001 ✓ |
| 2 | Sec I | Re-verify novelty (Transolver/Clifford in nuclear domain) | edit_002 ✓ |
| 3 | Sec III | Confirm branch-trunk predicts (vx, vy, vz) internally | edit_004 ✓ |
| 4 | Sec III-A | Specify gradient reconstruction scheme for Sobolev loss | edit_006 ✓ |
| 5 | Sec II-D | Add reliability/POD/FAR/V&V/RELAP5 review | edit_003 ✓ |
| 6 | Sec V-A | Retrain baseline OR retitle column | edit_005 ✓ |
| 7 | Sec V-C | Confirm grade-selective output in MultivectorToFields | edit_007 ✓ |
| 8 | Sec VI | Run eta_CFD-only ablation + report ROC-AUC/recall/F1 | edit_008 ✓ |
| 9 | Sec VIII | Replace bound-style metrics; multi-seed runs | edit_009 / edit_009B ✓ |
| 10 | Sec VIII-B | Regenerate Transolver++/Clifford panels as lossless PNG | edit_010 / edit_011 ✓ |
| 11 | Sec IX-E | Mamba result or future-work statement | edit_012 (partial) |
| 12 | Sec IX-E | DDPM result or future-work statement | edit_012 ✓ |
| 13 | Sec X | Data availability statement | **OPEN** |
| 14 | Acknowledgment | NPPAD formal citation | **OPEN** |
| 15 | Biographies | Student author bios (Punyansh, Anushka, Ayush) | **OPEN** |

---

## Workflow

### Step 1 — Read the TODO carefully
- If it matches an existing `edit_NNN_*.md`, check whether that edit already resolves it fully.
- If yes: point the user to the edit file and confirm it covers the TODO.
- If not or only partially: proceed with a new / supplementary edit.

### Step 2 — Gather evidence
Use the appropriate method:

**Literature search** (novelty, regulatory references, related work):
- Use `search_web` with targeted queries.
- Use `read_url_content` to fetch abstracts or citations.
- Summarize findings in 2–4 sentences, IEEE-cited.

**Code inspection** (implementation confirmation):
- Use `grep_search` and `view_file` on `src/`, `configs/`.
- Quote the exact file + line number as evidence.

**Metric extraction / demo run**:
- Write script to `Journal/scripts/todo_NNN_<description>.py`.
- Run with `conda run -n minor_proj python <script>` from project root.
- Paste captured output into edit file.

**Boilerplate text** (bios, data availability):
- Draft directly based on known information.
- Flag any specific items (GitHub URL, ORCID) with `[VERIFY: …]`.

### Step 3 — Draft replacement LaTeX

Write the exact text that replaces the `\textcolor{red}{[Input/confirmation required …]}` block. Text must be:
- Formal third-person academic.
- Hedged: "under the synthetic benchmark", "proof-of-concept", "preliminary".
- Never fabricated: unknown values → `[VERIFY: …]`.
- IEEE numeric citations: [N].
- Ready to paste into the `.tex` source.

### Step 4 — Save the edit file

**Path**: `Journal/edits/edit_NNN_<short_name>.md`
**NNN**: next sequential number after the highest existing edit number.

Use this template:

```
# Edit NNN — <Short Descriptive Title>

**Date**: YYYY-MM-DD
**Triggered by**: Research TODO skill — TODO item #N
**Paper section**: Sec X.Y — <Section Title>
**Edit type**: [addition | replacement | clarification | figure | metric | reference]
**Status**: [draft | ready-for-review | approved]

---

## Context

<What the TODO asked for. Quote the [Input/confirmation required] block.>

---

## Evidence

<Codebase search results, web search findings, script output. Include file paths + line numbers.>

---

## Proposed LaTeX Text

<Section: Sec X.Y — Subsection Name>

> [REPLACES: \textcolor{red}{[Input/confirmation required ...]}]

\`\`\`latex
<Exact replacement text, ready to paste.>
\`\`\`

---

## Notes / Caveats

<Assumptions, items team must verify before accepting.>
```

### Step 5 — Report back concisely

For each TODO, output:
- ✅ **Resolved** — what was found, what edit was written, file path.
- ⏳ **Script ready** — script path, command to run, placeholder text provided.
- ⏸️ **Partial** — what was done, what remains.
- ⏭️ **Skipped** — one-sentence reason.

---

## Writing Style Rules

| Rule | Detail |
|---|---|
| Voice | Third-person ("We demonstrate…", "The results indicate…") |
| Hedging | "under the synthetic benchmark", "proof-of-concept", "preliminary" |
| Equations | Numbered right-aligned: (1), (2)… |
| Figures | "Fig. N: Caption sentence case." |
| Tables | Caption above, sentence case |
| Citations | IEEE numeric [N] at end of clause |
| Unknown values | `[VERIFY: <what to check>]` inline |
| Tone | Never sabotage the paper — frame limitations as "scope of current work" |

---

## Hard Constraints

1. **Never fabricate numbers** — if a metric isn't measured, write `[VERIFY: run experiment]` or provide a script.
2. **Never modify the PDF** — all output goes to `Journal/edits/`.
3. **Always use `minor_proj` conda env** for any Python execution: `conda run -n minor_proj python <script>`.
4. **All scripts go to `Journal/scripts/`** only.
5. **Skip irrelevant TODOs** — if a TODO has zero impact on what a reviewer would block on, state that clearly and move on.
6. **Strategic framing** — present every result in the light most favorable to the paper's contribution. If a simpler model has a higher raw metric, contextualize (different loss, different objective complexity). Do not self-sabotage.
7. **Qualify all synthetic results** — "under the synthetic benchmark", every time.
8. **Check existing edits first** — never re-do work that edit_001 through edit_012 already resolved.

---

## Key Codebase Paths

| What | Where |
|---|---|
| DeepONetFourier | `src/deeponet/deeponet_fourier.py` |
| Transolver++ | `src/operators/transolver_operator.py` |
| Clifford Operator | `src/operators/clifford_operator.py` |
| Sobolev + Div loss | `src/deeponet/sobolev_loss.py` |
| Training loop | `src/deeponet/train.py` |
| Feature translation | `src/feature_translation/translator.py` |
| LOCA classifier | `src/accident_model/train_locac_model.py` |
| Inference pipeline | `src/inference/run_inference.py` |
| Configs | `configs/config.yaml`, `configs/model_config.yaml` |
| Results/plots | `results/` |
| Mock data | `scripts/generate_mock_data.py` |
| NPPAD data | `data/nppad/operation_csv_data/` |

---

## Common Patterns for Remaining Open TODOs

### TODO 13 — Data Availability Statement
Draft directly. Template:
> The synthetic dataset used in this study was generated using the physics-consistent mock generator provided in the project codebase, which is available at [VERIFY: GitHub URL]. The NPPAD (Nuclear Power Plant Accident Database) data used for LOCA classifier training is publicly available [VERIFY: cite NPPAD reference]. All trained model checkpoints and generated figures are available from the corresponding author upon reasonable request.

### TODO 14 — NPPAD Formal Citation
Search for the NPPAD dataset originating publication: "NPPAD nuclear power plant accident dataset".
Add a `\bibitem` in the references and cite at first use of NPPAD in Sec VI.

### TODO 15 — Student Author Biographies
Draft 2–4 sentence bios for:
- **Punyansh Thakur** — lead student author, CSE, IIIT Naya Raipur
- **Anushka Paul** — co-author, CSE, IIIT Naya Raipur
- **Ayush Kumar Bhadra** — co-author, CSE, IIIT Naya Raipur
Flag specific details (graduation year, research interests) with `[VERIFY: …]`.

### TODO 11 — Mamba Temporal Operator
Check `src/temporal/mamba_operator.py`. If no checkpoint exists in `results/models/`, draft a forward-looking statement removing "implemented" language. Example:
> A Mamba-style temporal operator [CITE] for transient sequence modelling at O(T) complexity has been prototyped in the codebase but not yet evaluated under the present benchmark. Its integration into the end-to-end pipeline, along with quantitative comparison against the steady-state surrogate baseline, constitutes a primary direction for future work.

---

*Last updated: 2026-09-02 by Antigravity AI assistant*
