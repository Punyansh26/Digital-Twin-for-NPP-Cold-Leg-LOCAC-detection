# Edit 010 — Lossless Panel Figures and Exact Table VII Metrics

**Date**: 2026-08-31
**Requested by**: Professor (paper Section VIII-B placeholder)
**Paper section**: Sec VIII-B — Synthetic-Benchmark Field Reconstruction Accuracy
**Edit type**: figure + metric
**Status**: draft

---

## Context

The professor's placeholder text in Sec VIII-B reads:

> "[Input/confirmation required from the students: the DeepONetFourier panels are
> currently sourced as lossless PNG while the Transolver++ and Clifford panels are
> lossy JPEG; regenerate all panels as lossless images (PNG/PDF/EPS) at matched
> resolution and color scale before submission, since JPEG compression artifacts on
> continuous colormaps can visually distort exactly the gradient-resolution
> comparison these figures are meant to demonstrate.]"

This maps to **Open Item #10** (CLAUDE.md Sec VIII-B — Regenerate figures as lossless PNG)
and **Open Item #9** (Sec VIII — Replace bound-style metrics with exact values).

---

## Evidence from Codebase

### Current figure format
- All DeepONetFourier comparison plots in `results/plots/` are **already lossless PNG**
  (2654–2659 × 740 px), generated via `src/deeponet/visualize.py:150`:
  ```python
  plt.savefig(plot_path, dpi=150, bbox_inches='tight')
  ```
- The Transolver++ and Clifford panels in the submitted draft were produced separately
  and saved as lossy JPEG — a workflow error. The code itself defaults to `.png`.
- **Resolution note**: Existing panels use `dpi=150`. Submission quality requires `dpi=300`.
- **Matched colorscale**: Existing panels set their own `vmin/vmax` per panel. For a
  cross-architecture comparison the colorscale must be anchored to ground truth.

### Unified regeneration script
- `Journal/scripts/edit_010_lossless_figures_and_table_metrics.py`
  - Loads all three model checkpoints.
  - Runs inference on test sample #0.
  - Generates Figs. 2–5 as lossless PNG (300 dpi, matched `vmin/vmax` from GT).
  - Saves to `results/plots/paper_figures/fig_{N}_{field}_lossless.png`.
  - Computes exact R², Rel-L2, MAE(norm.) for DeepONetFourier across all 300 test
    samples and saves to `Journal/scripts/edit_010_table_vii_metrics.json`.

### Table VII — current vs. exact
Current (bound-style):
```
Pressure         R² > 0.95   Rel-L2 < 0.05   MAE < 0.010
Velocity mag.    R² > 0.93   Rel-L2 < 0.07   MAE < 0.010
Temperature      R² > 0.97   Rel-L2 < 0.03   MAE < 0.005
Turbulence k     R² > 0.90   Rel-L2 < 0.10   MAE < 0.020
```

Exact values: [VERIFY — paste from `edit_010_table_vii_metrics.json` after script run]

Context from baseline_deeponet_results.json (vanilla DeepONet, matched-condition):
- Pressure R² = 0.9933, Rel-L2 = 0.0207, MAE = 0.0129
- Velocity R² = 0.9943, Rel-L2 = 0.0404, MAE = 0.0124
- Turbulence k R² = 0.9966, Rel-L2 = 0.0373, MAE = 0.0052
- Temperature R² = 0.9960, Rel-L2 = 0.0286, MAE = 0.0126

DeepONetFourier with Sobolev + divergence loss may differ; script provides exact values.

---

## Proposed Text

### Part A — Replace [Input/confirmation required] in Sec VIII-B

<Section: Sec VIII-B — Synthetic-Benchmark Field Reconstruction Accuracy>

> [DROP-IN REPLACEMENT / ADDITION BEGINS]

Figs. 2–5 present comparative field reconstructions across the three neural
operator architectures (DeepONetFourier, Transolver++, and Clifford Operator)
for each thermohydraulic output under the synthetic benchmark. All panels are
rendered from lossless PNG exports at 300\,dpi with a matched colormap range
($v_{\min}$, $v_{\max}$) derived from the ground-truth field of each plotted
test sample, ensuring that cross-architecture gradient-resolution comparisons
are not confounded by quantization or compression artifacts.

> [DROP-IN REPLACEMENT / ADDITION ENDS]

---

### Part B — Replace Table VII with exact values (LaTeX)

> [DROP-IN REPLACEMENT / ADDITION BEGINS]

\begin{table}[!t]
\renewcommand{\arraystretch}{1.3}
\caption{TABLE VII: DeepONetFourier Field Prediction Accuracy (300-Sample Test Set,
Synthetic Benchmark). Values are mean~$\pm$~standard deviation over all test
samples; metrics are computed on min-max normalised fields.}
\label{tab:deeponet_field_accuracy}
\centering
\begin{tabular}{lccc}
\hline
\textbf{Field} & $\mathbf{R^{2}}$ & \textbf{Rel.~$\ell_{2}$} & \textbf{MAE (norm.)} \\
\hline
Pressure $p$              & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] \\
Velocity magnitude $|\mathbf{v}|$ & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] \\
Temperature $T$           & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] \\
Turbulence KE $k$         & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] & [VERIFY] $\pm$ [std] \\
\hline
\multicolumn{4}{l}{\small All results under the synthetic benchmark;
  see Section~\ref{sec:limitations} for scope.}
\end{tabular}
\end{table}

> [DROP-IN REPLACEMENT / ADDITION ENDS]

Fill in [VERIFY] cells from `Journal/scripts/edit_010_table_vii_metrics.json`
keys: `r2_mean`, `r2_std`, `rel_l2_mean`, `rel_l2_std`, `mae_mean`, `mae_std`.

---

## Notes / Caveats

1. **Colorscale anchoring**: Use `vmin/vmax` from the ground-truth field of each
   plotted sample (as implemented in the script). Never use per-panel auto-scaling
   for cross-architecture comparison figures.

2. **300 dpi is the minimum for IEEE submission**; the team may use PDF/EPS for
   vector-rendering of contour plots at essentially infinite resolution.

3. **Multi-seed requirement** (Open Item #9): This edit provides single-seed exact
   values. Edit 009B covers the ≥ 5-seed sweep.

4. **Transolver++ / Clifford exact metrics**: This edit's Table VII covers
   DeepONetFourier only (as per the paper). Equivalent tables for Transolver++ and
   Clifford are covered in the Neural Operator Comparison subsection (Sec VIII-C).
   Exact values for those architectures require their test-set inference to be run
   through the same script.
