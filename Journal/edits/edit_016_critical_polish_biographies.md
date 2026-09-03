# Edit 016 — Critical Polish and Author Biographies

**Date**: 2026-09-04
**Requested by**: Authors (critical whole-manuscript review and final biography insertion)
**Paper section**: Whole manuscript; end matter — Author Biographies
**Edit type**: replacement + clarification + addition
**Status**: ready-for-review

---

## Context

The authors requested a critical review of
`Journal/completed_todo/paper.tex`, a more confident journal tone, black final
text, corrected formatting and spacing, and completed no-photo biographies for
Punyansh Thakur, Anushka Paul, Ayush Kumar Bhadra, and Vinay Kumar. The supplied
student facts identify Punyansh Thakur and Anushka Paul as third-year Data
Science and Artificial Intelligence students and Ayush Kumar Bhadra as a
third-year Electronics and Communication Engineering student at IIIT Naya
Raipur.

---

## Evidence from Codebase / Literature

- The author block previously assigned every author to Computer Science and
  Engineering. It is corrected at `Journal/completed_todo/paper.tex:46`--49 to
  distinguish Data Science and Artificial Intelligence, Electronics and
  Communication Engineering, and Computer Science and Engineering.
- The Fourier implementation stores the random projection as a
  $3\times m$ matrix and evaluates `x @ B`
  (`src/deeponet/fourier_encoding.py:50`, `:71`--72). The manuscript equation
  is therefore corrected to $\bm B^{\top}\bm y$ at
  `Journal/completed_todo/paper.tex:357`; the former $\bm B\bm y$ expression
  was dimensionally inconsistent.
- Cl(3,0) has eight basis components spanning four grades. The corrected
  description and non-physical latent-grade labels appear at
  `Journal/completed_todo/paper.tex:464`--483. The previous wording incorrectly
  described eight grades and identified every latent bivector with vorticity.
- The standard Tier-2 CLI and constructor default to batch size 4
  (`scripts/train_operator.py:69`, `:264`), not 16. In addition,
  `EarlyStopping.__call__` updates state but returns no Boolean value
  (`src/deeponet/train.py:79`--89), whereas the original Tier-2 driver checks
  that return value (`scripts/train_operator.py:199`). The training discussion
  and table are corrected at `Journal/completed_todo/paper.tex:229` and
  `:581`--609.
- Exact baseline test metrics are preserved in
  `results/models/baseline_deeponet_results.json:40`--53. They are now reported
  in a complete table at `Journal/completed_todo/paper.tex:628`--650 instead of
  being mentioned without the supporting four-field record.
- The translator assigns 80% of effective severity to the prescribed break
  input (`src/feature_translation/translator.py:105`--117). This affects the
  coupled translation path, not the classifier-only NPPAD test. The distinction
  is corrected at `Journal/completed_todo/paper.tex:780`.
- The tracked-draft `\added` macro is now an identity macro, colored hyperlink
  styling is hidden, and explicit blue equation scopes were removed. The PDF
  therefore renders all manuscript content in black.
- IEEEtran's no-photo biography environment inserts infinitely stretchable
  vertical space. A narrow patch replaces that stretch with fixed two-baseline
  separation at `Journal/completed_todo/paper.tex:23`--31, retaining the IEEE
  environment while preventing excessive gaps between short biographies.

---

## Proposed Text

<Section: End matter — Author Biographies>

> [DROP-IN ADDITION BEGINS]

```latex
\begin{IEEEbiographynophoto}{Punyansh Thakur}
is a third-year undergraduate student in the Data Science and Artificial
Intelligence program at IIIT Naya Raipur, India. His academic work in this
project focuses on neural-operator surrogate modeling, scientific machine
learning, and digital-twin workflows for engineering systems.
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Anushka Paul}
is a third-year undergraduate student in the Data Science and Artificial
Intelligence program at IIIT Naya Raipur, India. Her academic work in this
project focuses on data-driven thermohydraulic analysis and machine-learning
methods for reliability-oriented accident screening.
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Ayush Kumar Bhadra}
is a third-year undergraduate student in the Electronics and Communication
Engineering program at IIIT Naya Raipur, India. His academic work in this
project focuses on integrating engineering-domain knowledge with data-driven
field reconstruction and system-indicator analysis.
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Vinay Kumar}
is an Assistant Professor in the Department of Computer Science and Engineering
at IIIT Naya Raipur, India. He received his doctoral degree in computer science
and engineering from the Indian Institute of Technology (BHU), Varanasi,
India. Before joining IIIT Naya Raipur, he served as an Assistant Professor at
the National Institute of Technology Jamshedpur and Birla Institute of
Technology Mesra, Ranchi. He has authored or coauthored more than 30 papers in
peer-reviewed journals and conference proceedings and has served as a reviewer
for international journals and conferences. His research interests include
reliability, safety, and mathematical modeling.
\end{IEEEbiographynophoto}
```

> [DROP-IN ADDITION ENDS]

The manuscript also includes integrated replacements for the abstract,
contribution framing, Fourier equation, Clifford terminology, training
protocol, baseline evidence table, classifier/translation interpretation,
runtime scope, and conclusion. These replacements are already applied in
`Journal/completed_todo/paper.tex`.

---

## Notes / Caveats

1. The student program/year facts and Vinay Kumar's professional history are
   based on the information supplied by the authors. The project-focus
   sentences in the student biographies should be checked against the agreed
   author-contribution record before submission.
2. The authors should verify all three student e-mail addresses and the exact
   institutional naming preferred by IIIT Naya Raipur before submission.
3. Scientific claims remain explicitly scoped to analytic synthetic fields and
   PCTRAN-simulated NPPAD records. The stronger tone does not convert these
   results into Fluent, experimental, plant-operational, or certification
   evidence.
4. Validation build: `latexmk -pdf -interaction=nonstopmode -halt-on-error
   -outdir=/tmp/ap1000_paper_review paper.tex` from
   `Journal/completed_todo/`. The build completed as a 16-page letter-size PDF
   with embedded fonts, no undefined references or citations, no missing
   figures, and no overfull boxes. Pages 1, 8, 9, 15, and 16 were visually
   inspected for title, table, reference, and biography layout.
