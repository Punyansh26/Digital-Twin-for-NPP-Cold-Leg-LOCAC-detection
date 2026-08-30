# Edit 009 — Multi-Seed Result Stability

**Date**: 2026-08-31
**Requested by**: Paper placeholder `[Input/confirmation required from the students]` — Sec. VIII
**Paper section**: Sec. VIII — EXPERIMENTAL RESULTS (Tables VII–IX)
**Edit type**: metric | replacement
**Status**: draft

---

## Context

Section VIII currently states that its results come from single synthetic-benchmark
runs and presents several one-sided bounds, including `$R^2 > 0.90$`.  The
student-confirmation placeholder requires (i) exact values and (ii) repeated
training of DeepONetFourier, Transolver++, Clifford, and the LOCA classifier
over at least five random seeds, with results reported as mean $\pm$ standard
deviation.  This edit supplies the reporting protocol and drop-in replacements
for the affected narrative and tables.  It deliberately does **not** supply
numerical values that have not been measured.

---

## Evidence from Codebase / Existing Results

- The unresolved placeholder is in
  `Journal/Ver_2___A_Multi_Architecture_Neural_Operator_Digital_Twin_for_Real_Time_Cold_Leg_LOCAC_Detection_in_AP1000_Nuclear_Reactors__Copy_.tex:583`.
  The affected tables use bounds at lines 709–712 (Table VII), 727–733
  (Table VIII), and 764–768 (Table IX).
- `scripts/train_deeponet.py:56–69` computes per-field relative-$L_2$, $R^2$,
  MAE, and derivative-$L_2$; `:257–269` stores the metrics in the checkpoint.
  However, its CLI (`:326–344`) has no `--seed` option and its normal output
  name is not run-specific.
- `scripts/train_operator.py:157–174` and `:223–245` compute and print
  validation/test metrics for Transolver++ and Clifford, but checkpoint only
  the best validation loss (`:207–219`), not the test metrics.  Its CLI
  (`:251–270`) also has no `--seed` option.
- The classifier fixes both the estimator seed and the 80/20 split to 42
  (`src/accident_model/train_locac_model.py:38–47, 211–239`).  It therefore
  cannot currently perform the requested repeated-seed study without a small
  reproducibility extension.
- `results/models/baseline_deeponet_results.json:25–53` contains exact
  single-run metrics for the *vanilla baseline only*.  It is not evidence for
  DeepONetFourier, Transolver++, or Clifford and must not be substituted into
  Tables VII–VIII.
- `Journal/edits/edit_008_field_only_ablation.md:70–87` contains a genuine
  five-seed **classifier ablation**.  It does not retrain the three neural
  operators and therefore does not resolve the present Section VIII request.

No archived artifact was found containing five independently seeded runs for
each of the three target architectures plus the classifier.  The bound-style
values must consequently remain un-replaced until the protocol below is run.

---

## Required Measurement Protocol

Use five fixed seeds: $\{42, 7, 13, 99, 2024\}$.  Preserve the existing
synthetic dataset and its fixed 70/15/15 split (the preparation code uses
`random_state=42`); vary only stochastic training components for the three
operators.  For each seed, set Python, NumPy, PyTorch CPU, and PyTorch CUDA
RNGs before model construction; make the DataLoader shuffle generator
seeded; and record the full configuration, device, epoch selected by early
stopping, checkpoint path, and test-set metrics in a seed-specific JSON file.

Train each architecture with its current stated loss setting and all other
hyperparameters held fixed: DeepONetFourier with Sobolev and divergence terms,
and Transolver++ and Clifford with MSE.  This repeated-seed study measures
stability; it does not remove the existing non-identical-loss caveat.  Evaluate
all checkpoints on the same held-out 300-sample test partition.  Report the
arithmetic mean and **sample** standard deviation ($ddof=1$) over the five
run-level metrics; do not pool node-level errors across runs.

For the classifier, use the same five seeds for the model random state and a
seeded stratified 80/20 split.  Keep the data-preparation pipeline and all
classifier hyperparameters fixed, save the per-seed test predictions, and
report accuracy, precision, recall, F1, and ROC-AUC as five-run mean $\pm$
sample standard deviation.  State explicitly whether the classifier uses the
field-only configuration from Edit 008 or the original blended severity
configuration; the two should not be mixed in one aggregate.

Before accepting this edit, archive the five raw JSON/CSV records per model
under a dated, non-overwriting results directory and independently recompute
the aggregate table values from those records.

---

## Proposed Text

### 1. Replace the note at Sec. VIII opening (current line 583) after the runs complete

> [DROP-IN REPLACEMENT BEGINS]

\textbf{Result stability:} All results in this section are reported as the
mean $\pm$ sample standard deviation over five independently seeded runs
($s \in \{42,7,13,99,2024\}$) on the fixed synthetic-benchmark split.  For
the neural operators, the seed controls model initialization and minibatch
ordering; for the classifier, it additionally controls the stratified 80/20
train/test split.  Exact test-set metrics, rather than one-sided bounds, are
reported in Tables~\ref{tab:field_metrics}, \ref{tab:benchmark}, and
\ref{tab:locac_metrics}.  The architectural comparison remains preliminary
because DeepONetFourier uses Sobolev and divergence regularization whereas
Transolver++ and Clifford use MSE-only training.

> [DROP-IN REPLACEMENT ENDS]

### 2. Replace Table VII and its preceding sentence

> [DROP-IN REPLACEMENT BEGINS]

Table~\ref{tab:field_metrics} reports DeepONetFourier test-set field metrics
over five independently seeded training runs on the fixed 300-sample synthetic
test partition.  Values are mean $\pm$ sample standard deviation.

\begin{table}[htbp]
\caption{DeepONetFourier field prediction accuracy (synthetic test set; five seeds, mean $\pm$ standard deviation).}
\label{tab:field_metrics}
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Field} & \textbf{$R^2$} & \textbf{Rel.\,$L_2$} & \textbf{MAE (norm.)} \\
\midrule
Pressure & \textit{[VERIFY: $\bar R^2_p \pm s_p$]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Velocity magnitude & \textit{[VERIFY: $\bar R^2_{|v|} \pm s_{|v|}$]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Temperature & \textit{[VERIFY: $\bar R^2_T \pm s_T$]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Turbulence $k$ & \textit{[VERIFY: $\bar R^2_k \pm s_k$]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
\bottomrule
\end{tabular}
\end{table}

> [DROP-IN REPLACEMENT ENDS]

### 3. Replace Table VIII

> [DROP-IN REPLACEMENT BEGINS]

\begin{table*}[htbp]
\caption{Neural-operator performance on the synthetic test set (five seeds, mean $\pm$ standard deviation; non-identical loss settings).}
\label{tab:benchmark}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{DeepONetFourier} & \textbf{Transolver++} & \textbf{Clifford} \\
\midrule
Total parameters & 1,451,012 & 106,072 & 1,796 \\
Mean field $R^2$ & \textit{[VERIFY]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Pressure $R^2$ & \textit{[VERIFY]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Velocity-magnitude $R^2$ & \textit{[VERIFY]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Temperature $R^2$ & \textit{[VERIFY]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Turbulence-$k$ $R^2$ & \textit{[VERIFY]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
Model-only forward pass (ms) & \textit{[VERIFY: mean $\pm$ std]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
End-to-end latency (ms) & \textit{[VERIFY: mean $\pm$ std]} & \textit{[VERIFY]} & \textit{[VERIFY]} \\
\bottomrule
\end{tabular}
\end{table*}

> [DROP-IN REPLACEMENT ENDS]

### 4. Replace Table IX and revise the figure caption

> [DROP-IN REPLACEMENT BEGINS]

\begin{table}[htbp]
\caption{LOCA indicator classifier performance on the synthetic benchmark (five seeds, mean $\pm$ standard deviation).}
\label{tab:locac_metrics}
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Score} \\
\midrule
Accuracy & \textit{[VERIFY: mean $\pm$ std]} \\
Precision & \textit{[VERIFY: mean $\pm$ std]} \\
Recall (Sensitivity) & \textit{[VERIFY: mean $\pm$ std]} \\
F1 Score & \textit{[VERIFY: mean $\pm$ std]} \\
ROC-AUC & \textit{[VERIFY: mean $\pm$ std]} \\
\bottomrule
\end{tabular}
\end{table}

\caption{LOCA indicator detection on the synthetic benchmark.  The ROC,
precision--recall, and confusion-matrix panels correspond to the
pre-specified representative seed $s=42$; aggregate five-seed metrics are
reported in Table~\ref{tab:locac_metrics}.}

> [DROP-IN REPLACEMENT ENDS]

---

## Notes / Caveats

1. Do **not** paste the proposed tables while any `[VERIFY]` token remains.
   They are deliberately conspicuous placeholders, not estimated values.
2. A five-seed run of a classifier alone is insufficient: the reviewer request
   explicitly covers the three neural operators and the classifier.
3. The original classifier source currently works with NPPAD data when
   available and otherwise generates synthetic NPPAD-like data.  The final
   manuscript must identify which path produced Table IX and use that same
   path for every seed.
4. Mean field $R^2$ in Table VIII should be the unweighted arithmetic mean of
   the four per-field $R^2$ values for each run, then summarized over seeds.
   It must not be computed from the already-rounded table entries.
5. The current table labels and body text should be changed from “Single Run”
   and inequalities (`>`, `<`) only after the exact run records have been
   checked.  The existing non-identical-loss wording must remain.
