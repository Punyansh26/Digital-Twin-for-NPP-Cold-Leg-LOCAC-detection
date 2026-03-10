# AP1000 Digital Twin — Results Explained

This document provides a detailed explanation of every result artifact (plots, metrics, and saved models) produced by the AP1000 LOCAC Digital Twin pipeline.

---

## Table of Contents

1. [Saved Models](#1-saved-models)
2. [Training History](#2-training-history)
3. [DeepONet Field Comparison Plots](#3-deeponet-field-comparison-plots)
4. [LOCAC Detection Performance Plot](#4-locac-detection-performance-plot)
5. [DeepONet Training Metrics](#5-deeponet-training-metrics)
6. [LOCAC Detection Metrics](#6-locac-detection-metrics)
7. [Inference Output Explained](#7-inference-output-explained)

---

## 1. Saved Models

### `models/best_model.pth` — DeepONet Best Checkpoint

A PyTorch checkpoint saved whenever the **validation loss improves** during training. This is the recommended model for inference.

**Contents:**

| Key | Description |
|-----|-------------|
| `epoch` | The epoch number at which this best model was saved |
| `model_state_dict` | DeepONetFourier neural network weights |
| `optimizer_state_dict` | Adam optimizer state (for resuming training) |
| `metrics` | Per-field validation metrics (R², Rel L2, MAE) at the time of saving |
| `operator` | Architecture identifier (`deeponet_fourier`) |
| `config` | Full merged configuration dictionary |

**What it represents:** The neural operator that maps input parameters (velocity, break size, temperature) to full 3D CFD fields (pressure, velocity, turbulence, temperature) across 25,000 mesh nodes — acting as a surrogate for ANSYS Fluent simulations.

---

### `models/final_model.pth` — DeepONet Final Checkpoint

Same structure as `best_model.pth`, saved after the **last training epoch** regardless of performance. Useful for:
- Comparing best vs. final model to check for overfitting
- If training was interrupted, this captures the latest state

---

### `models/locac_detector.pkl` — LOCAC Classifier

A pickled Python dictionary containing:

| Key | Description |
|-----|-------------|
| `model` | Trained scikit-learn `GradientBoostingClassifier` (200 trees, depth 4) |
| `scaler` | `StandardScaler` fitted on the 7 NPPAD features during training |
| `metrics` | Performance metrics from the test split (accuracy, precision, recall, F1, ROC-AUC) |
| `config` | Configuration used during training |

**The 7 NPPAD features it uses:**

| Feature | Full Name | Unit | Normal Range | LOCAC Range |
|---------|-----------|------|-------------|-------------|
| P | Primary Coolant Pressure | bar | ~155.5 | ~63 |
| TAVG | Average Coolant Temperature | °C | ~301 | ~262 |
| WRCA | Reactor Coolant Flow (Loop A) | kg/s | ~16,835 | varies |
| PSGA | Steam Generator A Pressure | bar | ~67 | varies |
| SCMA | Subcooling Margin A | °C | ~35.6 | varies |
| DNBR | Departure from Nucleate Boiling Ratio | — | ~5.6 | ~127 |
| DT_HL_CL | Hot-Leg minus Cold-Leg Temperature | °C | ~16 | ~1.6 |

---

### `models/training_history.json` — Epoch-by-Epoch Metrics

A JSON file with three arrays, one entry per training epoch:

| Key | Description |
|-----|-------------|
| `train_loss` | Average training loss per epoch (MSE + optional Sobolev + Divergence) |
| `val_loss` | Average validation loss per epoch (same metric, on held-out data) |
| `learning_rate` | Learning rate at each epoch (adjusted by ReduceLROnPlateau scheduler) |

**How to interpret:**
- **Decreasing train_loss** → the model is learning
- **Decreasing val_loss** → the model generalises well to unseen data
- **val_loss increasing while train_loss decreases** → overfitting (the early stopping mechanism prevents this)
- **Learning rate drops** → the scheduler detected a plateau and reduced LR by 50% to escape local minima

---

## 2. Training History

The training history tracks the DeepONet's learning progress over all epochs.

### Loss Curves

| Curve | What It Shows |
|-------|---------------|
| **Train Loss** | How well the model fits the training data. Should decrease smoothly. |
| **Validation Loss** | How well the model predicts on unseen data (15% of the dataset). This is the true performance indicator. |

**What to look for:**
- **Gap between curves**: A small gap means good generalisation. A large gap signals overfitting.
- **Convergence**: Both curves should flatten. If train loss is still dropping steeply when training ends, more epochs may help.
- **Spikes**: Occasional spikes in val_loss are normal (noisy batches). Persistent spikes may indicate learning rate issues.

### Learning Rate Schedule

The `ReduceLROnPlateau` scheduler monitors validation loss:
- If val_loss hasn't improved for 20 epochs → LR is halved
- Initial LR: 0.001 → can drop to ~1e-6
- **Staircase pattern** in the LR plot is expected and healthy

---

## 3. DeepONet Field Comparison Plots

### Files: `plots/test_{0-4}_{field}_comparison.png`

There are **20 plots** total: 5 test cases × 4 CFD fields.

**Test cases:** 5 randomly selected parameter sets from the held-out test split (never seen during training).

**Fields:** `pressure`, `velocity_magnitude`, `turbulence_k`, `temperature`

### Each Plot Contains 3 Side-by-Side Panels (18×5 inches)

#### Panel 1 (Left) — CFD Ground Truth

- **What:** The actual CFD simulation result from ANSYS Fluent (or mock data)
- **Axes:** x position (metres) vs y position (metres)
- **Colours:** Contour plot using the `jet` colormap with 20 levels
- **Purpose:** The "correct answer" that we want the DeepONet to reproduce

#### Panel 2 (Centre) — DeepONet Prediction

- **What:** The neural operator's prediction for the same input parameters
- **Axes & Colours:** Identical scale (same vmin/vmax) as the CFD panel for fair comparison
- **Purpose:** Visually assess how closely the surrogate matches CFD

**What to look for:**
- Contour shapes should match — same patterns, gradients, and hotspots
- Colour values should be similar — no drastic shifts in field magnitude
- Smooth contours indicate the model has learned the underlying physics

#### Panel 3 (Right) — Relative Error

- **What:** Pointwise relative error = |CFD − Prediction| / (|CFD| + ε)
- **Colours:** `Reds` colormap (darker = higher error)
- **Purpose:** Highlights exactly where the model struggles

**What to look for:**
- **Mostly pale/white** → excellent agreement (errors < 1%)
- **Dark spots at boundaries or elbows** → the model may need more training data in those geometric regions
- **Uniform light colour** → the error is evenly distributed (good)
- **High error at sharp gradients** → expected for neural operators; the Sobolev loss helps mitigate this

### Field-Specific Interpretation

#### Pressure Comparison (`test_X_pressure_comparison.png`)
- Normal pressure is ~15.5 MPa (AP1000 primary loop operating pressure)
- Look for: correct pressure gradient along the pipe, pressure drop at the elbow
- High error near the elbow bend is expected — this is where the flow physics are most complex

#### Velocity Magnitude Comparison (`test_X_velocity_magnitude_comparison.png`)
- Normal inlet velocity is 4–6 m/s
- Look for: boundary layer profile (slower near walls, faster at centre), acceleration at the elbow
- The velocity field is critical for LOCAC detection — break scenarios disrupt flow patterns

#### Turbulence K Comparison (`test_X_turbulence_k_comparison.png`)
- Turbulent kinetic energy (k) in the RNG k-ε model
- Look for: elevated k near the elbow (secondary flows create turbulence), low k in straight sections
- This field often has the highest relative error because turbulence is inherently harder to predict
- The optional Diffusion Turbulence Super-Resolution model can improve this

#### Temperature Comparison (`test_X_temperature_comparison.png`)
- Coolant temperature ~290–320°C
- Look for: temperature gradients along the pipe (if applicable), uniform temperature in well-mixed regions
- Temperature is typically the smoothest field and should have the lowest error

---

## 4. LOCAC Detection Performance Plot

### File: `plots/locac_detection_performance.png`

A single figure (15×5 inches) with **3 subplots** evaluating the LOCAC classifier's ability to distinguish Normal operation from Loss of Coolant Accidents.

### Subplot 1 (Left) — ROC Curve

**Axes:** False Positive Rate (x) vs True Positive Rate (y)

**What it shows:** The trade-off between detecting real LOCAs (sensitivity) and triggering false alarms (specificity) at every possible classification threshold.

| Element | Meaning |
|---------|---------|
| **Blue curve** | The classifier's performance across all thresholds (0→1) |
| **Dashed diagonal** | Random-guessing baseline (AUC = 0.5) |
| **AUC value in legend** | Area Under the Curve — closer to 1.0 is better |

**How to read it:**
- A curve that hugs the **top-left corner** is excellent
- AUC ≥ 0.95 → the classifier strongly separates Normal from LOCAC
- AUC = 0.99 → near-perfect discrimination
- The further the curve is from the diagonal, the better

**Nuclear safety context:** In reactor safety, missing a real LOCA (low recall / low TPR) is far worse than a false alarm. The top of the ROC curve (high TPR) is the critical region.

### Subplot 2 (Centre) — Precision-Recall Curve

**Axes:** Recall (x) vs Precision (y)

**What it shows:** How precision (of those flagged as LOCAC, how many are real) changes as we try to catch more LOCAs (recall).

**How to read it:**
- A curve that stays **high and flat** toward the right is ideal — meaning we can catch nearly all LOCAs without many false alarms
- A sharp drop at high recall means catching the last few LOCAs requires accepting many more false positives
- More informative than ROC when classes are imbalanced (our data has far more LOCAC rows than Normal)

### Subplot 3 (Right) — Confusion Matrix

**Axes:** Predicted label (x) vs Actual label (y)

A 2×2 heatmap showing classification counts:

|  | Predicted Normal | Predicted LOCAC |
|--|-----------------|-----------------|
| **Actual Normal** | True Negatives (TN) | False Positives (FP) — false alarms |
| **Actual LOCAC** | False Negatives (FN) — missed LOCAs | True Positives (TP) — correctly detected |

**How to read it:**
- **Diagonal (TN + TP)** should be dark blue (high counts) — these are correct predictions
- **Off-diagonal (FP + FN)** should be pale — these are errors
- **FN (bottom-left)** is the most dangerous cell in nuclear safety — missed accidents
- With our transitional training data, some FP/FN in the transition zone is expected and desirable (it means the model outputs graded probabilities rather than binary 0/1)

---

## 5. DeepONet Training Metrics

These metrics evaluate how accurately the DeepONet surrogate reproduces each CFD field.

### R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

| Value | Interpretation |
|-------|---------------|
| R² = 1.0 | Perfect prediction |
| R² > 0.95 | Excellent — the surrogate captures >95% of spatial variance |
| R² > 0.90 | Good — suitable for engineering screening |
| R² < 0.80 | Poor — predictions may miss important flow features |

**What it tells you:** How much of the variance in the CFD output the DeepONet explains. Computed per field (pressure, velocity, turbulence, temperature).

### Relative L2 Error

$$\text{Rel } L_2 = \frac{\|y - \hat{y}\|_2}{\|y\|_2}$$

| Value | Interpretation |
|-------|---------------|
| < 0.01 | Excellent — <1% normalised error |
| < 0.05 | Good — within 5% of CFD accuracy |
| < 0.10 | Acceptable for surrogate use |
| > 0.10 | The model may need more training or data |

**What it tells you:** The total prediction error normalised by the magnitude of the true field. A single number summarising spatial accuracy.

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

| Value | Interpretation |
|-------|---------------|
| < 0.001 | Excellent (in normalised space) |
| < 0.01 | Good |
| < 0.05 | Acceptable |

**What it tells you:** The average pointwise error in normalised units. Less sensitive to outliers than MSE. Indicates the "typical" error at any random mesh node.

### Derivative L2 Error (Upgraded Trainer)

$$\text{Deriv } L_2 = \frac{\|\Delta\hat{y} - \Delta y\|_2}{\|\Delta y\|_2 + \epsilon}$$

Computed from finite differences of the spatial field.

**What it tells you:** Whether the model captures sharp gradients (pressure drops, velocity boundary layers) or only smooth trends. Critical for LOCAC detection where breaks create abrupt field changes.

### Loss Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| **MSE Loss** | 1.0 | Standard data fidelity — match CFD values |
| **Sobolev Loss** | 0.1 (β) | Gradient fidelity — match spatial derivatives |
| **Divergence Penalty** | 0.01 (λ) | Physics consistency — enforce ∇·**u** ≈ 0 |

**Total loss = MSE + β × Sobolev + λ × Divergence**

---

## 6. LOCAC Detection Metrics

These metrics evaluate the LOCAC classifier's performance on the held-out test set.

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Overall correct classification rate. Our target is ≥ 90%.

**Caveat:** Accuracy can be misleading with imbalanced data. If 95% of samples are LOCAC, a model that always predicts "LOCAC" achieves 95% accuracy but is useless. That's why we also track precision, recall, and F1.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all samples the classifier flagged as LOCAC, what fraction actually are.

- **High precision** → few false alarms
- **Low precision** → many false alarms (operators get alarm fatigue)

### Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual LOCAC events, what fraction were detected.

- **High recall** → few missed accidents
- **Low recall** → dangerous — the system fails to detect real LOCAs

**In nuclear safety, recall is the most critical metric.** Missing a LOCA can lead to core damage.

### F1 Score

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean of precision and recall. Balances the trade-off between false alarms and missed detections. A single number summarising classifier quality.

| Value | Interpretation |
|-------|---------------|
| > 0.95 | Excellent |
| > 0.90 | Good for deployment |
| > 0.80 | Acceptable for screening |
| < 0.70 | Needs improvement |

### ROC-AUC (Area Under the ROC Curve)

The area under the ROC curve (subplot 1 of the performance plot). Measures overall discrimination ability across all possible classification thresholds.

| Value | Interpretation |
|-------|---------------|
| 1.0 | Perfect separation of Normal and LOCAC |
| > 0.95 | Excellent discriminator |
| > 0.80 | Good |
| 0.5 | No better than random guessing |

---

## 7. Inference Output Explained

When running inference (`python scripts/run_inference.py`), the output is organised into sections:

### Section 1: CFD-Derived Features

These come directly from the DeepONet's predicted fields:

| Feature | How Computed | What It Means |
|---------|-------------|---------------|
| `average_pressure` | Mean of pressure field (Pa) | Average coolant pressure across the pipe cross-section |
| `pressure_gradient` | Linear fit slope | Pressure change per unit length — drives coolant flow |
| `pressure_drop` | outlet − inlet pressure (Pa) | Total pressure loss through the elbow — increases with break size |
| `mass_flow_rate` | ρ × velocity × area (kg/s) | Coolant mass throughput — drops during LOCAC |
| `inlet_velocity` | Mean velocity at inlet (m/s) | Inlet bulk flow speed |
| `max_turbulence` | Max of turbulence_k field | Peak turbulent kinetic energy — rises during transients |
| `avg_turbulence` | Mean of turbulence_k field | Average turbulence level |
| `avg_temperature` | Mean of temperature field (K) | Bulk coolant temperature |
| `temperature_difference` | Max − min temperature (K) | Temperature inhomogeneity |
| `velocity_std` | Std dev of velocity field | Flow uniformity — higher = more disturbed flow |
| `pressure_std` | Std dev of pressure field | Pressure uniformity |

### Section 2: NPPAD-Mapped Signals

System-level reactor parameters mapped from input parameters + CFD anomaly signals using a sigmoid severity model:

| Signal | Mapped From | Physical Meaning |
|--------|-------------|-----------------|
| Primary Pressure (bar) | break_size + CFD anomalies | RCS depressurisation during LOCA |
| Avg Coolant Temp (°C) | temperature input + severity | Core cooling effectiveness |
| RCS Flow Loop-A (kg/s) | velocity + severity | Reactor coolant pump discharge |
| SG-A Pressure (bar) | severity | Steam generator secondary-side response |
| Subcooling Margin (°C) | severity | Margin above saturation — safety buffer |
| DNBR | severity | Closest approach to boiling crisis — critical safety parameter |
| HL-CL ΔT (°C) | severity | Temperature difference between hot and cold legs — collapses during LOCAC |

### Section 3: Severity Breakdown

| Value | Source | Meaning |
|-------|--------|---------|
| Input severity | Sigmoid of break_size | How severe the accident is based on pipe break alone (0%→~0.02, 5%→0.50, 10%→~0.98) |
| CFD anomaly | Turbulence + pressure + flow deviations | How abnormal the DeepONet's predicted fields look compared to nominal |
| Effective severity | 80% input + 20% CFD | Combined severity used to map NPPAD features |

### Section 4: Decision

| Output | Meaning |
|--------|---------|
| LOCAC Probability | Classifier confidence that this scenario is a Loss of Coolant Accident (0.0 = certainly Normal, 1.0 = certainly LOCAC) |
| Decision | `✓ NORMAL` if probability < 0.5, `⚠ LOCAC DETECTED` if ≥ 0.5 |

**Expected probability range across break sizes:**

| Break Size | Expected Probability | Interpretation |
|-----------|---------------------|----------------|
| 0% | ~0.10 | Normal operation — very low risk |
| 2% | ~0.13 | Minor perturbation — still normal |
| 4% | ~0.40 | Transition zone — elevated concern |
| 5% | ~0.56 | Just above threshold — marginal LOCAC |
| 7% | ~0.74 | Significant break — clear LOCAC |
| 10% | ~0.91 | Severe break — high-confidence LOCAC |

---

## Quick Reference: File Index

| File | Type | Description |
|------|------|-------------|
| `models/best_model.pth` | Model | Best DeepONet checkpoint (lowest val loss) |
| `models/final_model.pth` | Model | Final-epoch DeepONet checkpoint |
| `models/locac_detector.pkl` | Model | LOCAC classifier + scaler |
| `models/training_history.json` | Data | Per-epoch train/val loss and learning rate |
| `plots/locac_detection_performance.png` | Plot | ROC curve, PR curve, confusion matrix |
| `plots/test_{0-4}_pressure_comparison.png` | Plot | Pressure field: CFD vs DeepONet vs error |
| `plots/test_{0-4}_velocity_magnitude_comparison.png` | Plot | Velocity field: CFD vs DeepONet vs error |
| `plots/test_{0-4}_turbulence_k_comparison.png` | Plot | Turbulence field: CFD vs DeepONet vs error |
| `plots/test_{0-4}_temperature_comparison.png` | Plot | Temperature field: CFD vs DeepONet vs error |
