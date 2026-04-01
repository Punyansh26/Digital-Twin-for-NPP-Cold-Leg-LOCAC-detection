# AI Architectures in the AP1000 Digital Twin for LOCA Detection

> **Project**: Digital Twin for Nuclear Power Plant (NPP) Cold-Leg LOCA (Loss of Coolant Accident) Detection  
> **Purpose**: Real-time, physics-informed prediction of reactor coolant system flow fields and automated LOCA event classification using modern neural operators

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Landscape](#2-architecture-landscape)
3. [DeepONet — Legacy Baseline](#3-deeponet--legacy-baseline)
4. [DeepONet with Fourier Feature Encoding (DeepONetFourier) — Default](#4-deeponet-with-fourier-feature-encoding-deeponetfourier--default)
5. [Transolver++ — Transformer Neural Operator](#5-transolver--transformer-neural-operator)
6. [Clifford Neural Operator — Geometric Algebra](#6-clifford-neural-operator--geometric-algebra)
7. [Mamba Temporal Operator — Selective State Space Model](#7-mamba-temporal-operator--selective-state-space-model)
8. [Liquid Neural Network / CfC — Sensor Fusion Model](#8-liquid-neural-network--cfc--sensor-fusion-model)
9. [Diffusion Turbulence Model — DDPM](#9-diffusion-turbulence-model--ddpm)
10. [Supporting Loss Functions & Physics Constraints](#10-supporting-loss-functions--physics-constraints)
11. [LOCA Classifier (Downstream Task)](#11-loca-classifier-downstream-task)
12. [Model Comparison & Recommendation](#12-model-comparison--recommendation)
13. [Quick Reference Table](#13-quick-reference-table)

---

## 1. Project Overview

The AP1000 is a Westinghouse pressurised water reactor (PWR). Its **cold leg** is the pipe that carries cooled primary coolant back to the reactor core. A rupture anywhere in this loop is classified as a **Cold-Leg LOCA** event and demands immediate detection because delayed response can lead to core damage.

Traditional detection relies on CFD (Computational Fluid Dynamics) solvers (e.g. ANSYS Fluent), which take **hours per simulation**. This project replaces those solvers with **neural operators** that learn the input-to-solution mapping from pre-computed CFD data and run in **< 20 ms** — a 1000× speedup — while still obeying the underlying physics.

The pipeline has three stages:

```
Physical Parameters ──► Neural Operator ──► Flow Field Prediction
(velocity, break size,    (DeepONet / Transolver    (pressure, velocity,
 temperature)              / Clifford)                TKE, temperature)
                                  │
                                  ▼
                      Feature Translator
                      (CFD fields → NPPAD signals)
                                  │
                                  ▼
                      LOCA Classifier (Gradient Boosting / MLP)
                      ──► LOCA Probability Score
```

---

## 2. Architecture Landscape

The project organises its models in a three-tier hierarchy:

| Tier | Model | Role | Status |
|------|-------|------|--------|
| **1** | DeepONet (Legacy) | Baseline neural operator | Functional, superseded |
| **1** | DeepONet Fourier | **Default** neural operator | ✅ Active |
| **2** | Transolver++ | Transformer neural operator | Selectable |
| **2** | Clifford Neural Operator | Geometric-algebra operator | Selectable |
| **3** | Mamba Temporal Operator | Transient sequence modelling | Optional extension |
| **3** | Liquid Neural Network (CfC) | Irregular sensor time-series | Optional extension |
| **3** | Diffusion Turbulence Model | Stochastic turbulence + UQ | Optional extension |

**Activation via `model_config.yaml`**:

```yaml
model_version: deeponet_fourier   # change to: deeponet | transolver | clifford
```

---

## 3. DeepONet — Legacy Baseline

**File**: `src/deeponet/model.py`  
**Classes**: `BranchNet`, `TrunkNet`, `DeepONet`, `DeepONetLoss`

### 3.1 Mathematical Concept

DeepONet (Deep Operator Network) is based on the **Universal Approximation Theorem for Operators** (Chen & Chen, 1995; Lu et al., 2021). It learns a mapping between two function spaces:

```
G : U → V
```

where `U` is the space of input parameters (physics conditions) and `V` is the space of output functions (flow field solutions). It approximates:

```
G(u)(y) ≈  Σₖ bₖ(u) · tₖ(y)  + b₀
```

- **bₖ(u)** — basis coefficients computed from the input parameters by the **Branch Network**
- **tₖ(y)** — basis functions evaluated at spatial coordinate `y` by the **Trunk Network**
- The operator output is the inner product (dot product) of the two networks' outputs

### 3.2 Architecture Diagram

```
Branch Input  [B, 3]                   Trunk Input  [N, 3]
(velocity, break_size, temperature)    (x, y, z coordinates)
        │                                      │
 ┌──────▼───────┐                     ┌────────▼──────────┐
 │  Linear(3→256)│                     │  Linear(3→256)    │
 │  ReLU + Drop  │                     │  ReLU + Dropout   │
 │  Linear(256→512)│                   │  Linear(256→512)  │
 │  ReLU + Drop  │                     │  ReLU + Dropout   │
 │  Linear(512→512)│                   │  Linear(512→256)  │
 │  ReLU + Drop  │                     └────────┬──────────┘
 │  Linear(512→256)│                            │
 └──────┬───────┘                     Trunk Out [N, 256]
        │
 Branch Out [B, 256]
        │
        └──────────── matmul ──────── [B, N] + bias
                                           │
                              ┌────────────▼──────────────┐
                              │  Stack 4 fields            │
                              │  [B, 4, N]                 │
                              └────────────────────────────┘
```

One independent Branch + Trunk pair exists per output field (4 pairs total).

### 3.3 Parameters

| Sub-Network | Layer Sizes | Activation |
|-------------|-------------|------------|
| BranchNet | 3 → 256 → 512 → 512 → 256 | ReLU + Dropout(0.1) |
| TrunkNet  | 3 → 256 → 512 → 256 | ReLU + Dropout(0.1) |
| Bias terms | 4 scalars | — |

**Total trainable parameters**: ~2–3 M  
**Dot-product basis dimension**: 256

### 3.4 Input / Output Specification

| Tensor | Shape | Description |
|--------|-------|-------------|
| `branch_input` | `[B, 3]` | Simulation parameters: (velocity m/s, break_size %, temperature °C) |
| `trunk_input` | `[N, 3]` | Mesh node coordinates: (x, y, z) in metres |
| Output | `[B, 4, N]` | Predicted fields: (pressure Pa, velocity magnitude m/s, TKE m²/s², temperature °C) |

### 3.5 Forward Pass (Code)

```python
for i in range(self.n_outputs):          # one pass per field
    branch_out = self.branch_nets[i](branch_input)   # [B, 256]
    trunk_out  = self.trunk_nets[i](trunk_input)     # [N, 256]
    output     = branch_out @ trunk_out.T + self.biases[i]  # [B, N]
```

### 3.6 Loss Function

Weighted per-field MSE:

```python
L = Σᵢ wᵢ · MSE(pred[:, i, :], target[:, i, :])
```

Weights default to `[1.0, 1.0, 1.0, 1.0]`.

### 3.7 Limitations

- **Spectral bias**: ReLU networks struggle to learn high-frequency spatial patterns (sharp gradients near walls, turbulence interfaces).
- **Fixed activations**: Cannot adapt activation steepness during training.
- **Larger parameter count** than the upgraded architecture (more layers for same effective expressivity).

---

## 4. DeepONet with Fourier Feature Encoding (DeepONetFourier) — Default

**File**: `src/deeponet/deeponet_fourier.py`  
**Supporting files**:  
- `src/deeponet/fourier_encoding.py` — `FourierFeatureEncoding`  
- `src/deeponet/adaptive_activation.py` — `AdaptiveActivationLayer`, `AdaptiveGELU`  
**Classes**: `BranchNetFourier`, `TrunkNetFourier`, `DeepONetFourier`

### 4.1 Key Innovations Over Legacy DeepONet

| Feature | Legacy DeepONet | DeepONetFourier |
|---------|----------------|-----------------|
| Coordinate encoding | Raw (x, y, z) | **Random Fourier Features** |
| Activation | ReLU | **Adaptive GELU** (learnable slope) |
| Branch depth | 5 layers | 3 layers (128, 256, 256) |
| Trunk depth | 4 layers | Fourier(512) + 2 AdaptGELU + Linear |
| Physics loss | Weighted MSE only | **Sobolev gradient loss + Divergence penalty** |

### 4.2 Component 1 — Fourier Feature Encoding

**File**: `src/deeponet/fourier_encoding.py`  
**Class**: `FourierFeatureEncoding`

**Problem it solves**: Standard MLPs suffer from *spectral bias* — they preferentially learn low-frequency components of the target function and are slow to capture sharp spatial gradients (important near walls and at turbulence boundaries).

**Formula**:

```
γ(x) = [sin(2π B x),  cos(2π B x)]   ∈ ℝ^{2m}

where:
  x ∈ ℝ³         — spatial coordinates (x, y, z)
  B ∈ ℝ^{3×m}    — random frequency matrix (sampled from N(0, σ²))
  m = 256         — mapping_size (output dimension = 2m = 512)
  σ = 10.0        — fourier_scale (controls frequency bandwidth)
```

**Code**:
```python
x_proj = 2.0 * π * (x @ B)          # [N, mapping_size]
output = [sin(x_proj), cos(x_proj)]  # [N, 2 * mapping_size = 512]
```

The matrix `B` is **fixed** (not trained) by default, making the encoding a stable random projection. Setting `trainable=True` allows the frequencies to be learned.

**Effect**: By injecting sinusoidal features of many frequencies simultaneously, the network can match both slowly-varying (bulk flow) and rapidly-varying (turbulence, boundary layers) spatial patterns from the first layer onwards.

### 4.3 Component 2 — Adaptive GELU Activation

**File**: `src/deeponet/adaptive_activation.py`  
**Classes**: `AdaptiveGELU`, `AdaptiveActivationLayer`

Each neuron in the trunk's hidden layers has a **learnable slope parameter** `a`:

```
f(x) = GELU(a · x)
```

- `a` is a **per-neuron** `nn.Parameter`, initialised to `1.0`
- Gradient descent adjusts `a` independently for each neuron
- Allows the effective curvature and saturation of the activation to adapt to the local function landscape

This is particularly beneficial in the trunk network where the mapping from spatial coordinates to basis functions varies widely across the geometry.

### 4.4 Architecture Diagram

```
Branch Input [B, 3]                   Trunk Input [N, 3]
       │                                     │
 ┌─────▼──────────────┐           ┌──────────▼──────────────────┐
 │ Linear(3→128)       │           │ FourierEncoding             │
 │ GELU + Dropout(5%) │           │   B ∈ R^{3×256}, σ=10      │
 │ Linear(128→256)    │           │   [N,3] → [N,512]            │
 │ GELU + Dropout(5%) │           └──────────┬──────────────────┘
 │ Linear(256→256)    │                      │
 └─────┬──────────────┘           ┌──────────▼──────────────────┐
       │                           │ AdaptiveGELU Layer          │
 Branch Out [B, 256]               │   Linear(512→256)           │
       │                           │   GELU(a·x), a per-neuron   │
       │                           │   Dropout(5%)               │
       │                           ├─────────────────────────────┤
       │                           │ AdaptiveGELU Layer          │
       │                           │   Linear(256→256)           │
       │                           │   GELU(a·x), a per-neuron   │
       │                           └──────────┬──────────────────┘
       │                                      │
       │                           ┌──────────▼──────────────────┐
       │                           │ Linear(256→256)             │
       │                           │ (no activation — raw basis) │
       │                           └──────────┬──────────────────┘
       │                                      │
       │                           Trunk Out [N, 256]
       │                                      │
       └──────────── matmul ──────── [B, N]  + bias[i]
                                           │
                              ┌────────────▼───────────┐
                              │  Stack 4 field outputs  │
                              │  → [B, 4, N]            │
                              └────────────────────────┘
```

### 4.5 Parameters

| Sub-Network | Sizes | Activation |
|-------------|-------|------------|
| BranchNet | 3 → 128 → 256 → 256 | GELU + Dropout(5%) |
| Fourier Encoding | 3 → 512 (fixed random proj.) | sin / cos |
| TrunkNet Adapt Layer 1 | 512 → 256 | Adaptive GELU |
| TrunkNet Adapt Layer 2 | 256 → 256 | Adaptive GELU |
| TrunkNet Output | 256 → 256 | None (linear) |

**Total trainable parameters (per field)**: ~500 K  
**Total with 4 fields**: ~1.5–2 M  
**Reduction vs. legacy**: ~25–40% fewer parameters

### 4.6 Hyperparameter Configuration

```yaml
fourier_deeponet:
  branch_net:
    input_dim:   3
    hidden_dims: [128, 256]
    output_dim:  256
  trunk_net:
    coord_dim:     3
    hidden_dims:   [256, 256]
    output_dim:    256
    mapping_size:  256      # m → output_dim = 2m = 512
    fourier_scale: 10.0     # σ of random frequency matrix B
  output_fields: ["pressure", "velocity_magnitude", "turbulence_k", "temperature"]
  n_outputs: 4
```

### 4.7 Training Configuration

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau (patience=20, factor=0.5) |
| Batch size | 16 |
| Max epochs | 2000 |
| Early stopping patience | 50 |
| Mixed precision | Enabled (CUDA, AMP) |
| Dropout | 5% (branch and trunk) |
| Weight initialisation | Kaiming Normal |

### 4.8 Loss Function (Total)

```
L_total = α · L_Sobolev + λ · L_Divergence

where:
  L_Sobolev  = MSE(pred, target) + β · MSE(∇pred, ∇target)
  L_Divergence = λ · mean( (∇ · u_vel)² )

Default: α=1.0, β=0.1, λ=0.01
```

See [Section 10](#10-supporting-loss-functions--physics-constraints) for full details.

---

## 5. Transolver++ — Transformer Neural Operator

**File**: `src/operators/transolver_operator.py`  
**Classes**: `MeshEmbedding`, `MultiHeadSelfAttention`, `TransformerTokenBlock`, `TransolverOperator`  
**Reference**: Wu et al., *"Transolver: A Fast Transformer Solver for PDEs on General Geometries"*, ICML 2024

### 5.1 Motivation

Standard Transformers applied to PDE meshes are expensive: O(N²) for N mesh nodes. Transolver reduces this by **compressing the N-node mesh into M ≪ N learned tokens**, running attention over M tokens (M = 64 by default), then scattering predictions back to the full mesh.

### 5.2 Architecture Diagram

```
Branch Input [B, 3]                    Mesh Coordinates [N, 3]
       │                                        │
 ┌─────▼────────────────────┐       ┌───────────▼───────────────────────┐
 │ Branch Encoder            │       │ MeshEmbedding (Tokenizer)          │
 │  Linear(3→256) + GELU    │       │                                    │
 │  Linear(256→256)          │       │  coord_proj: [N,3] → [N,256]      │
 └─────┬─────────────────────┘       │  token_query: [N,256] → [N, M=64] │
       │                             │  attn_w = softmax(logits) [N,M]   │
 Branch Feat [B, 256]                │  tokens  = attn_w.T @ node_feat   │
       │                             │           [M, 256]                 │
       │                             └───────────┬───────────────────────┘
       │                                         │
       │                              tokens [M, 256]  attn_w [N, M]
       │                                         │
       └──────────── add ──────────────► tokens + branch_feat  [M, 256]
                                                 │
                              ┌──────────────────▼──────────────────────┐
                              │  Transformer Block × 4                   │
                              │                                          │
                              │  ┌─────────────────────────────────────┐│
                              │  │ LayerNorm → MultiHeadAttn (8 heads) ││
                              │  │ Residual +                          ││
                              │  │ LayerNorm → MLP (256→1024→256)      ││
                              │  │ Residual                            ││
                              │  └─────────────────────────────────────┘│
                              └──────────────────┬──────────────────────┘
                                                 │
                              LayerNorm  [M, 256]
                                                 │
                              Scatter Decoder:
                              node_feat = attn_w @ tokens  [N, 256]
                                                 │
                              Per-field Linear(256→1) × 4 + bias
                                                 │
                              Output  [B, 4, N]
```

### 5.3 Component Details

#### MeshEmbedding (Tokenizer)

```python
node_feat    = Linear(coord_dim → embed_dim)(coords)   # [N, 256]
attn_logits  = Linear(embed_dim → n_tokens)(node_feat) # [N, M]
attn_w       = softmax(attn_logits, dim=-1)            # [N, M]
tokens       = attn_w.T @ node_feat                    # [M, 256]
```

Each mesh node contributes a soft-weighted vote to its nearest "token cluster". The M token query vectors are learned, so they discover physics-meaningful spatial regions (e.g. near-wall boundary layer, free stream, elbow region).

#### MultiHeadSelfAttention

Standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) · V

Q, K, V ∈ ℝ^{M × embed_dim}
d_k = embed_dim / n_heads = 256 / 8 = 32
```

Complexity is O(M²) = O(64²) = O(4096) — independent of the mesh size N.

#### TransformerTokenBlock

```
x ← x + Dropout(MHSA(LayerNorm(x)))     # residual attention
x ← x + Dropout(MLP(LayerNorm(x)))      # residual MLP
```

MLP ratio = 4, so hidden size = 1024.

#### Scatter Decoder

```python
node_feat = attn_w @ tokens   # [N, M] × [M, 256] = [N, 256]
```

The same soft assignment weights used in the tokenizer are used here in reverse to re-project token features back to every mesh node.

### 5.4 Parameters

| Component | Parameters |
|-----------|-----------|
| Branch encoder | 3 → 256 → 256 (~67K) |
| MeshEmbedding | coord_proj + token_query (~33K) |
| 4 × Transformer blocks | ~4 × 200K = 800K |
| Output projection (×4 fields) | 4 × 256 ≈ 1K |
| **Total** | **~900K – 1M** |

### 5.5 Hyperparameter Configuration

```yaml
transolver:
  coord_dim:  3
  branch_dim: 3
  embed_dim:  256
  n_tokens:   64      # M — mesh compression vocabulary
  n_layers:   4       # Transformer depth
  n_heads:    8       # Attention heads (embed_dim / n_heads = 32)
  n_outputs:  4
  dropout:    0.1
```

### 5.6 Input / Output Specification

| Tensor | Shape | Description |
|--------|-------|-------------|
| `branch_input` | `[B, 3]` | Physics parameters |
| `mesh_points` | `[N, 3]` | Mesh node coordinates |
| Output | `[B, 4, N]` | Predicted flow fields |

### 5.7 Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|-----------|
| Captures global spatial context via attention | Higher memory footprint than DeepONet |
| Handles arbitrary geometries naturally | Requires careful n_tokens tuning |
| Scales sub-quadratically in N | Slower training convergence |
| Good generalisation across mesh sizes | No built-in physical symmetry |

---

## 6. Clifford Neural Operator — Geometric Algebra

**File**: `src/operators/clifford_operator.py`  
**Classes**: `CliffordLinear`, `CliffordLayerNorm`, `PhysicsToMultivector`, `MultivectorToFields`, `CliffordNeuralOperator`  
**Reference**: Brandstetter et al., *"Clifford Neural Layers for PDE Modeling"*, ICLR 2023

### 6.1 Motivation

Fluid dynamics is governed by vector equations that are **rotationally equivariant**: if you rotate the geometry, the velocity field rotates accordingly. Standard MLPs do not respect this symmetry, forcing them to learn it empirically from data. Clifford networks **encode the symmetry algebraically**, producing physically consistent predictions even in rotated geometries.

### 6.2 Geometric Algebra Background — Cl(3,0)

The Clifford algebra Cl(3,0) is a **16-dimensional algebra** built over ℝ³. Every element is a **multivector** with 8 independent grades:

| Grade | Basis Elements | Physical Meaning | Index |
|-------|---------------|-----------------|-------|
| 0 (scalar) | 1 | Pressure, temperature | 0 |
| 1 (vectors) | e₁, e₂, e₃ | Velocity components | 1, 2, 3 |
| 2 (bivectors) | e₁₂, e₁₃, e₂₃ | Angular momentum, vorticity | 4, 5, 6 |
| 3 (trivector) | e₁₂₃ | Volume element, pseudoscalar | 7 |

**Multiplication rules** (metric signature +,+,+):
```
eᵢ² = +1
eᵢ · eⱼ = −eⱼ · eᵢ   (i ≠ j)
e₁₂ = e₁ · e₂,  etc.
```

**Geometric product formula** (full 8×8 bilinear table):
```python
# a, b: [..., 8] multivector tensors

c[0] = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7  # scalar
c[1] = a0*b1 + a1*b0 - a2*b4 + a3*b5 + a4*b2 - a5*b3 - a6*b7 - a7*b6  # e1
c[2] = a0*b2 + a1*b4 + a2*b0 - a3*b6 - a4*b1 + a5*b7 + a6*b3 - a7*b5  # e2
c[3] = a0*b3 - a1*b5 + a2*b6 + a3*b0 - a4*b7 - a5*b1 - a6*b2 + a7*b4  # e3
# ... (grades 2 and 3 follow similar rules)
```

This single operation simultaneously computes inner products (grade-lowering) and outer products (grade-raising), naturally coupling scalar and vector physics.

### 6.3 Architecture Diagram

```
Branch Params [B, 3]        Mesh Coords [B, N, 3]
       │                           │
       │                           │
       └────────────────┬──────────┘
                        │
           PhysicsToMultivector
           ┌─────────────────────────────────────────┐
           │  scalar_proj:  [3] → [C=32]  (grade-0)  │
           │  vector_proj:  [3] → [C×3]   (grade-1)  │
           │  zeros for grades 2, 3                   │
           │  Output: [B, N, C=32, 8]                 │
           └───────────────────┬─────────────────────┘
                               │
              CliffordLinear × 4 (with residual + LayerNorm)
           ┌─────────────────────────────────────────┐
           │  weight: [out_C, in_C, 8]               │
           │  out = Σᵢ weight_i ⊗ x_i  (geo. prod.) │
           │  + bias: [out_C, 8]                     │
           │  CliffordLayerNorm(n_channels=32)        │
           │  Residual: h = h + h_prev               │
           │  GELU on grade-0 (scalar gate)           │
           └───────────────────┬─────────────────────┘
                               │ [B, N, 32, 8]
                               │
              MultivectorToFields
           ┌─────────────────────────────────────────┐
           │  Flatten: [B, N, 32*8=256]              │
           │  Linear(256 → 4)                        │
           └───────────────────┬─────────────────────┘
                               │
                   Output [B, N, 4]
                         permute
                   Output [B, 4, N]
```

### 6.4 CliffordLinear Layer (Core Operation)

```python
# weight: [C_out, C_in, 8]
# x:      [..., C_in, 8]

# Vectorised geometric product:
x_exp = x.unsqueeze(-3)         # [..., 1,     C_in, 8]
w_exp = weight.unsqueeze(0)     # [1,   C_out, C_in, 8]
prod  = clifford_product(w_exp, x_exp)   # [..., C_out, C_in, 8]
out   = prod.sum(dim=-2) + bias          # [..., C_out, 8]
```

Each output channel is a sum of geometric products of all input channels with a learned multivector weight — this is the Clifford algebra generalisation of a standard linear layer.

### 6.5 Scalar-Gate Residual

```python
h = layer(h)          # Clifford transform
h = norm(h)           # LayerNorm over [C, 8]
h = h + h_res         # additive residual
h[..., 0] = F.gelu(h[..., 0])  # GELU only on grade-0 (scalar)
```

Only the scalar grade is activated — grade-1 vector components remain linear to preserve equivariance under rotation.

### 6.6 Parameters

| Component | Parameters |
|-----------|-----------|
| PhysicsToMultivector | ~3K |
| 4 × CliffordLinear (32→32) | 4 × (32×32×8 + 32×8) ≈ 4 × 8.5K ≈ 34K |
| 4 × CliffordLayerNorm | ~4 × 512 ≈ 2K |
| MultivectorToFields | 256 × 4 = 1K |
| **Total** | **~40–60K learnable + ~10K weight tensors** |

> **Note**: The Clifford operator is the most **parameter-efficient** model — ~10× fewer parameters than DeepONetFourier — due to the structured algebraic representation.

### 6.7 Hyperparameter Configuration

```yaml
clifford:
  coord_dim:  3
  branch_dim: 3
  n_channels: 32    # C — multivector channels per layer
  n_layers:   4     # Clifford layer depth
  n_outputs:  4
```

### 6.8 Rotational Equivariance Property

For a rotation matrix `R ∈ SO(3)`:

```
CliffordNet(R · x_coords, params) = R_applied · CliffordNet(x_coords, params)
```

Velocity vector outputs transform correctly under rotations, while scalar outputs (pressure, temperature) are invariant. This is automatically guaranteed by the geometric product structure, not enforced by data augmentation.

---

## 7. Mamba Temporal Operator — Selective State Space Model

**File**: `src/temporal/mamba_operator.py`  
**Classes**: `SelectiveSSM`, `MambaBlock`, `MambaTemporalOperator`  
**Reference**: Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*, arXiv 2312.00752 (2023)

### 7.1 Purpose

The previous three models predict a **steady-state** flow field for a given set of parameters. For a **LOCA transient**, the flow evolves over time — the Mamba operator models this **temporal evolution**, predicting how the flow state changes step-by-step from the moment of the rupture.

### 7.2 Why Not a Transformer for Sequences?

Standard Transformers compute self-attention in O(T²) time for a sequence of T timesteps. For long transients (T = 1000+ timesteps at sub-second resolution), this becomes prohibitively expensive. Mamba scales **O(T)** by using a recurrent state space.

### 7.3 Selective State Space Model (SSM) Theory

A continuous-time SSM is:

```
ẋ(t) = A·x(t) + B·u(t)    (state transition ODE)
y(t)  = C·x(t)             (readout)
```

Discretised (Euler) with timestep Δ:

```
xₖ = exp(Δ·A)·xₖ₋₁ + Δ·B·uₖ     (≈ dA·xₖ₋₁ + dB·uₖ)
yₖ = C·xₖ
```

**Selectivity** means Δ, B, and C are **input-dependent** (computed from the current token `uₖ`), allowing the model to decide how much history to retain vs. reset at each timestep — analogous to LSTM gates, but more expressive and parallelisable.

### 7.4 Architecture

```
Flow States [B, T, state_dim=16]
       │
 ┌─────▼───────────────────────────────┐
 │ Encoder: Linear(16→128) + GELU      │
 └─────┬───────────────────────────────┘
       │                    [B, T, 128]
 ┌─────▼─────────────────────────────────────────────┐
 │  MambaBlock × 4                                   │
 │  ┌─────────────────────────────────────────────┐  │
 │  │ LayerNorm → SelectiveSSM → Dropout          │  │
 │  │ + Residual                                  │  │
 │  └─────────────────────────────────────────────┘  │
 └─────┬─────────────────────────────────────────────┘
       │
 LayerNorm  [B, T, 128]
       │
 Decoder: Linear(128→16)
       │
 Next States [B, T, 16]
```

### 7.5 SelectiveSSM Internal Structure

```python
# Input expansion
xz    = Linear(d_model, 2*d_inner)(x)  # [B, T, 2*256]
x_d, z = split(xz)                    # gate z, signal x_d

# Causal depthwise conv1d (kernel=4, grouped)
x_d = SiLU(conv1d(x_d))               # local receptive field [B, T, d_inner]

# Input-dependent SSM parameters
dt_raw, B_mat, C_mat = Linear(d_inner, 2*d_state+1)(x_d).split(...)
delta = Softplus(dt_proj(dt_raw))     # [B, T, d_inner] — step size

# Discretise and scan
A     = -exp(A_log)                    # [d_inner, d_state] — diagonal
dA    = exp(delta[:,:,:,None] * A)    # [B, T, d_inner, d_state]
dB    = delta[:,:,:,None] * B_mat[:,:,None,:]  # [B, T, d_inner, d_state]

# Sequential scan (O(T) per sequence)
for t in range(T):
    h = dA[:,t] * h + dB[:,t] * u[:,t,None]   # state update
    y[:,t] = (h * C_mat[:,t,None]).sum(-1)     # readout

# Gate and project
y = (y + D * x_d) * SiLU(z)          # skip + gating
return out_proj(y)
```

### 7.6 Parameters

| Component | Parameters |
|-----------|-----------|
| Encoder | ~2K |
| 4 × MambaBlock (d_model=128, d_state=16) | ~4 × 130K ≈ 520K |
| LayerNorm + Decoder | ~17K |
| **Total** | **~540K** |

### 7.7 Hyperparameter Configuration

```yaml
mamba:
  state_dim: 16    # flow state vector size
  d_model:   128   # internal model width
  n_layers:  4     # Mamba block depth
  d_state:   16    # SSM latent state size
  dropout:   0.1
```

### 7.8 Autoregressive Prediction

```python
model.predict_sequence(
    initial_state,   # [B, 16] — state at rupture onset
    n_steps=200,     # predict 200 future timesteps
    context_len=64,  # sliding window of 64 recent states
)
```

---

## 8. Liquid Neural Network / CfC — Sensor Fusion Model

**File**: `src/temporal/liquid_nn_sensor_model.py`  
**Classes**: `CfCCell`, `LiquidNNSensorModel`  
**Reference**: Hasani et al., *"Closed-form Continuous-time Neural Networks"*, Nature Machine Intelligence 2022

### 8.1 Purpose

Nuclear plant sensors report at **irregular intervals** (e.g. fast-scan sensors every 10 ms, slow-scan every 1 s). Standard RNNs assume fixed timesteps; CfC neurons handle arbitrary Δt natively via an analytical ODE solution.

### 8.2 CfC Neuron — Mathematical Foundation

The underlying continuous-time dynamics:

```
dh/dt = −τ⁻¹(x,h) · (h − g(x,h))
```

This says the state `h` exponentially decays toward an attractor `g(x,h)` with input-dependent time constant `τ(x,h)`.

**Closed-form solution** (exact for any Δt):

```
h(t + Δt) = h(t) · exp(−Δt / τ(x,h))
           + (1 − exp(−Δt / τ(x,h))) · g(x,h)
```

Implemented as:

```python
decay  = exp(−Δt · τ)             # forgetting factor in [0,1]
h_next = decay * h + (1−decay) * f   # convex mix of old state + attractor
```

Where:
- `f = Tanh(W_x·x + W_h·h + b)` — target attractor (backbone network)
- `τ = Softplus(log_τ) + |gate(x,h)|` — effective time constant (positive)
- `gate = σ(gate_x·x + gate_h·h + gate_b)` — gating signal

### 8.3 Architecture

```
Sensor Series [B, T, n_sensors=8]     Timestamps [B, T]
       │                                      │
 LayerNorm(n_sensors)                         │
       │                                      │
 ┌─────▼──────────────────────────────────────▼───────┐
 │  CfC Layer 1 (input=8, hidden=64)                   │
 │   For each timestep t:                              │
 │     Δt   = timestamps[:,t] − timestamps[:,t-1]      │
 │     gate = σ(Wx·xₜ + Wh·h + b)                     │
 │     f    = Tanh(backbone([xₜ, h]))                  │
 │     τ    = Softplus(log_τ) + |gate|                 │
 │     h    = exp(−Δt·τ)·h + (1−exp(−Δt·τ))·f         │
 └────────────────────────────────┬───────────────────┘
                                  │ [B, hidden=64]
 ┌────────────────────────────────▼───────────────────┐
 │  CfC Layer 2 (input=64, hidden=64)  (same logic)   │
 └────────────────────────────────┬───────────────────┘
                                  │ Final hidden state
 ┌────────────────────────────────▼───────────────────┐
 │  Latent Projection: Linear(64→32) + Tanh           │
 └────────────────────────────────┬───────────────────┘
                                  │ [B, 32]
               ┌──────────────────┴──────────────────┐
               │   Risk Head                         │
               │   Linear(32→32) + GELU + Drop(0.1)  │
               │   Linear(32→1) + Sigmoid            │
               └──────────────────┬──────────────────┘
                                  │
          ┌───────────────────────┴───────────────────┐
          │               │                           │
 latent_state [B, 32]   risk_prob [B, 1]
```

### 8.4 NPPAD Sensor Features

The 7 reactor signals used as inputs:

| Signal | Description |
|--------|-------------|
| P | Reactor coolant system pressure |
| TAVG | Average coolant temperature |
| WRCA | Reactor coolant flow rate |
| PSGA | Pressuriser steam generator availability |
| SCMA | Safety injection signal |
| DNBR | Departure from nucleate boiling ratio |
| DT_HL_CL | Temperature differential (hot leg – cold leg) |

### 8.5 Parameters

| Component | Parameters |
|-----------|-----------|
| CfC Layer 1 (8 → 64) | ~9.5K |
| CfC Layer 2 (64 → 64) | ~17K |
| Latent + Risk Head | ~3K |
| **Total** | **~30K** |

### 8.6 Hyperparameter Configuration

```yaml
liquid_nn:
  n_sensors:   8     # input sensor channels
  hidden_size: 64    # CfC hidden state size per layer
  latent_dim:  32    # compressed latent state for risk head
  n_layers:    2     # stacked CfC layers
```

---

## 9. Diffusion Turbulence Model — DDPM

**File**: `src/generative/diffusion_turbulence_model.py`  
**Classes**: `ResBlock1D`, `TurbulenceDenoisingUNet`, `DDPMScheduler`, `DiffusionTurbulenceModel`  
**Reference**: Ho et al., *"Denoising Diffusion Probabilistic Models"*, NeurIPS 2020

### 9.1 Purpose

DeepONet predicts the **mean** flow field. Turbulence is inherently **stochastic** — the actual flow instantaneously fluctuates around the mean. This model generates multiple realistic turbulence **realisations**, providing:

1. **Uncertainty quantification** (UQ) — how spread out are plausible scenarios?
2. **Super-resolved turbulence** — finer-scale structures than the mean-field prediction
3. **Ensemble LOCA assessment** — downstream classifier sees worst-case realisations

### 9.2 Diffusion Process Theory

**Forward process** (training): Gradually add noise to clean turbulence fluctuations `x₀`:

```
q(xₜ | x₀) = N(√ᾱₜ · x₀,  (1 − ᾱₜ) I)

where:
  βₜ ∈ [1e-4, 0.02] — linear noise schedule (1000 steps)
  αₜ = 1 − βₜ
  ᾱₜ = Π_{s=1}^{t} αₛ   (cumulative product)
```

**Reverse process** (inference): The neural network denoises from pure noise `xₜ` back to clean fluctuations `x₀`:

```
p_θ(xₜ₋₁ | xₜ) = N(μ_θ(xₜ, t),  β̃ₜ I)

μ_θ = (1/√αₜ) · (xₜ − (βₜ / √(1−ᾱₜ)) · ε_θ(xₜ, t, cond))
```

The network `ε_θ` learns to **predict the noise** added at step t, conditioned on:
- `t` — diffusion timestep (sinusoidal embedding)
- `cond` — mean flow statistics from DeepONet

### 9.3 Architecture (1-D U-Net Denoiser)

```
Noisy Fields [B, 4, N]     Timestep t [B]     Conditioning [B, 8]
       │                        │                     │
       │                 SinusoidalEmbed(64)    ConcatEncode (n_fields+1 → 8)
       │                 Linear(64→128)+SiLU         │
       │                 Linear(128→64)              │
       │                 t_emb [B, 64]               │
       │                        │                     │
 ┌─────▼────────────────────────▼─────────────────────▼──────┐
 │ ResBlock1D: in_ch=4 → out_ch=C0=32                        │
 │   GroupNorm + Conv1D(3) + SiLU                            │
 │   + time_mlp inject: Linear(64→32)                        │
 │   + cond_mlp inject: Linear(8→32)                         │
 │   Skip: Conv1d(4→32,k=1)                                  │
 │   h0 [B, 32, N]                                           │
 ├───────────────────────────────────────────────────────────┤
 │ ResBlock1D: C0=32 → C1=64                                 │
 │   h1 [B, 64, N]                                           │
 ├───────────────────────────────────────────────────────────┤
 │ Bottleneck ResBlock1D: C1=64 → C2=64                      │
 │   h [B, 64, N]                                            │
 ├───────────────────────────────────────────────────────────┤
 │ Decoder ResBlock1D: cat([h, h1]) → C3=32                  │
 │   Input ch = C2+C1 = 128 (skip connection from encoder)   │
 │   h [B, 32, N]                                            │
 ├───────────────────────────────────────────────────────────┤
 │ Decoder ResBlock1D: cat([h, h0]) → n_fields=4             │
 │   Input ch = C3+C0 = 64 (skip connection from encoder)    │
 │   Predicted noise ε̂ [B, 4, N]                            │
 └───────────────────────────────────────────────────────────┘
```

### 9.4 Conditioning

```python
cond_input = [mean_fields.mean(dim=-1),    # [B, n_fields] — mean over nodes
              tke_field.mean(dim=-1)]      # [B, 1]
cond = Linear(n_fields+1, 64) + GELU + Linear(64, 8) + Tanh
```

The mean-flow statistics tell the denoiser what kind of flow regime it is operating in, and thus what kind of turbulence to generate.

### 9.5 Sampling (Inference)

```python
x = randn(n_samples, n_fields, N)        # pure Gaussian noise
for t in reversed(range(1000)):
    ε̂   = denoiser(x, t, cond)           # predict noise
    mean = (1/√αₜ) · (x − (βₜ/√(1−ᾱₜ)) · ε̂)
    x    = mean + √βₜ · randn_like(x)   # resample (skip at t=0)
x = x.clamp(-5, 5)                       # stability clip
```

Each call to `sample()` returns a **different realisation** from the distribution of turbulence fields consistent with the given mean flow.

### 9.6 Training Loss

Standard DDPM ε-prediction loss:

```
L_diffusion = E[||ε − ε̂_θ(√ᾱₜ · x₀ + √(1−ᾱₜ) · ε, t, cond)||²]
```

### 9.7 Parameters

| Component | Parameters |
|-----------|-----------|
| Cond encoder | ~4.5K |
| U-Net (2 enc + bottleneck + 2 dec blocks) | ~80K |
| **Total** | **~85K** |

### 9.8 Hyperparameter Configuration

```yaml
diffusion_turbulence:
  n_fields:     4       # flow field channels
  cond_dim:     8       # conditioning vector size
  n_diff_steps: 1000    # DDPM forward process steps
```

---

## 10. Supporting Loss Functions & Physics Constraints

### 10.1 Sobolev Gradient-Enhanced Loss

**File**: `src/deeponet/sobolev_loss.py`  
**Class**: `SobolevLoss`

```
L_Sobolev = α · ||u_pred − u_true||²   (value error)
           + β · ||∇u_pred − ∇u_true||² (gradient error)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `alpha` | 1.0 | Weight for value MSE term |
| `beta` | 0.1 | Weight for spatial gradient MSE term |

Gradients are computed by:
1. **Autograd** (primary): `torch.autograd.grad(u.sum(), coords)` when trunk coordinates have `requires_grad=True`
2. **Finite differences** (fallback): `(u[..., 2:] − u[..., :-2]) / 2.0` (central differences on the node ordering)

**Why it matters**: Without the gradient term, two predictions with the same point-wise values but different spatial derivatives (one smooth, one oscillatory) would have identical loss. The Sobolev term penalises non-physical oscillations.

### 10.2 Divergence Penalty (Physics Regularisation)

**File**: `src/physics/divergence_penalty.py`  
**Class**: `DivergencePenalty`

For incompressible flow, the **continuity equation** requires:

```
∇ · u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z = 0
```

The penalty enforces this:

```
L_div = λ · mean( ||∇ · u||² )
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `weight` (λ) | 0.01 | Penalty strength |

Since the model predicts velocity **magnitude** (not components), a **proxy divergence** based on the spatial variation of magnitude is used:

```python
div_proxy = central_diff(velocity_magnitude)   # ≈ ∂|u|/∂x along node ordering
L_div = λ · mean(div_proxy²)
```

For vector velocity components (full 3D), the exact divergence can be computed via `compute_full_divergence_penalty(vx, vy, vz)`.

### 10.3 Combined Total Loss

```
L_total = L_Sobolev + L_div
        = α · MSE(pred, true)
        + β · MSE(∇pred, ∇true)
        + λ · mean((∇·u)²)
```

Configuration switches:

```yaml
use_sobolev_loss:       true
sobolev_alpha:          1.0
sobolev_beta:           0.1
use_divergence_penalty: true
divergence_weight:      0.01
```

---

## 11. LOCA Classifier (Downstream Task)

**File**: `src/accident_model/train_locac_model.py`  
**Class**: `LOCACDetector`

This is not a neural operator but the final binary classifier that converts flow-field features into a LOCA probability score.

### 11.1 Feature Pipeline

```
Neural Operator output [B, 4, N]
       │
Feature Translator (src/feature_translation/translator.py)
       │
Extracted Features:
  • avg_pressure              • pressure_gradient
  • mass_flow_rate            • max_TKE, avg_TKE
  • avg_temperature           • velocity_std
  • pressure_std
       │
NPPAD Feature Mapping
  • P  (coolant pressure)
  • TAVG (average temperature)
  • WRCA (coolant flow rate)
  • PSGA (pressuriser availability)
  • SCMA (safety injection signal)
  • DNBR (departure from nucleate boiling ratio)
  • DT_HL_CL (hot-cold leg temp. differential)
       │
Classifier [7 features → LOCA probability]
```

### 11.2 Classifier Options

**Option A — Gradient Boosting (default)**:

```python
GradientBoostingClassifier(
    n_estimators  = 200,
    max_depth     = 4,
    learning_rate = 0.05,
    min_samples_leaf = 20
)
```

**Option B — MLP**:

```python
MLPClassifier(
    hidden_layer_sizes = (128, 64, 32),
    dropout            = 0.3,
    max_iter           = 1000
)
```

### 11.3 Data Augmentation — Transitional Samples

To handle the class-imbalance between Normal and LOCA and improve probability calibration, synthetic intermediate samples are generated:

```
19 severity levels from 5% to 95% break size
200 samples per level
→ Interpolation between Normal and LOCA distributions
```

### 11.4 Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy | > 90% |
| ROC-AUC | > 0.95 |
| Inference latency | < 5 ms |

---

## 12. Model Comparison & Recommendation

### 12.1 Head-to-Head Comparison

| Criterion | DeepONet (Legacy) | DeepONetFourier ⭐ | Transolver++ | Clifford Operator |
|-----------|:-----------------:|:------------------:|:-------------:|:-----------------:|
| **Parameters** | ~2.5M | ~1.8M | ~900K | ~50K |
| **Inference speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **High-frequency accuracy** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Global spatial context** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Physics symmetry** | ❌ | ❌ | ❌ | ✅ Rotational equivariance |
| **Training stability** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory footprint** | Medium | Medium | High (attention) | Very low |
| **Generalisation to new geometries** | Low | Medium | High | High |
| **Interpretability** | Medium | Medium | Low | Medium (grade analysis) |
| **Training data needed** | Many | Many | Many | Few |
| **Implementation complexity** | Simple | Medium | Medium | High |

### 12.2 When to Use Each Model

#### ✅ Use **DeepONetFourier** (Default) when:

- You have **pre-processed CFD data** from the exact AP1000 cold-leg geometry
- You need **maximum inference speed** (< 5 ms on CPU, < 1 ms on GPU)
- Flow fields contain **sharp gradients** (boundary layers, elbow wake regions)
- You are training on **≥ 500 CFD simulations**
- Hardware is limited (single GPU or even CPU deployment)

**Why it works best for this project**: The AP1000 cold-leg geometry is fixed. The model learns the exact geometry's operator. Fourier encoding handles the wall boundary layers and turbulent regions that ReLU networks struggle with. Sobolev + Divergence losses ensure the predictions obey conservation laws.

---

#### ✅ Use **Transolver++** when:

- You need to **generalise across different reactor geometries** (straight pipe → elbow → T-junction) without retraining
- The mesh topology **changes** between evaluations (adaptive mesh refinement)
- You have **large training datasets** (1000+ simulations) and sufficient GPU memory
- **Global flow patterns** (recirculation zones, separation bubbles) are critical to capture
- You are doing **real-time monitoring** and can afford the slightly higher latency (~10 ms)

**Condition**: If a variant of the AP1000 were retrofitted with additional piping configurations, Transolver's mesh-agnostic tokenisation would require far less retraining data.

---

#### ✅ Use **Clifford Neural Operator** when:

- **Data is scarce** (< 100 CFD simulations) — the structured algebraic representation acts as a strong inductive prior, requiring less data to generalise
- You need **rotationally equivariant predictions** — e.g. the geometry includes multiple elbow angles and orientations
- **Physical consistency** is paramount and must be guaranteed by the model architecture (not just trained for)
- You want **interpretable internal representations** — grade-0 channels encode scalars (pressure), grade-1 channels encode vectors (velocity)
- **Memory is severely constrained** — only ~50K parameters

**Condition**: In a multi-loop reactor analysis where the same neural operator must handle differently-oriented cold legs without geometry-specific retraining, Clifford's equivariance pays off.

---

#### ✅ Use **Mamba Temporal Operator** (Tier 3 addition) when:

- You need to model **transient dynamics** — how the flow evolves second-by-second after a rupture
- Sequences are **very long** (T > 200 timesteps) where transformers would be too slow
- You want to generate **timeline forecasts** of a LOCA event for operator dashboards
- Combined with a neural operator for spatial predictions and Mamba for temporal evolution

---

#### ✅ Use **Liquid Neural Network (CfC)** (Tier 3 addition) when:

- **Sensor data is irregularly sampled** (different scan frequencies, communication dropouts)
- You want a **direct sensor-to-risk** pathway without going through CFD fields
- **Online learning** or **edge deployment** is required (very small model, ~30K params)
- As a **cross-validation layer** — if both the neural operator + downstream classifier and the CfC sensor model agree on LOCA, confidence is much higher

---

#### ✅ Use **Diffusion Turbulence Model** (Tier 3 addition) when:

- You need **uncertainty quantification** — the operator provides a mean field, the diffusion model generates a distribution
- You are doing **probabilistic safety assessment** and need worst-case scenario generation
- **Turbulent fluctuations** matter for the DNBR calculation (departure from nucleate boiling is sensitive to turbulence intensity)
- Post-hoc analysis of already-detected LOCA events to understand the turbulence structure

---

### 12.3 Decision Tree

```
Is data limited (< 200 simulations)?
├── YES → Clifford Neural Operator (structured prior, few parameters)
└── NO
    ├── Is geometry fixed (always AP1000 cold-leg)?
    │   ├── YES → DeepONetFourier (fastest, best at known geometry)
    │   └── NO (multiple geometries or new configurations)
    │       └── Transolver++ (mesh-agnostic attention)
    │
    └── Add Tier-3 extensions as needed:
        ├── Temporal dynamics → Mamba
        ├── Sensor time-series → Liquid NN
        └── Uncertainty quantification → Diffusion model
```

### 12.4 Performance Benchmark Summary

| Model | Target R² | Target Inference | Parameters | Training Time |
|-------|-----------|-----------------|------------|---------------|
| DeepONet Legacy | > 0.85 | < 5 ms | ~2.5M | ~10 min |
| **DeepONetFourier** | **> 0.90** | **< 5 ms** | **~1.8M** | **~5 min** |
| Transolver++ | > 0.88 | < 15 ms | ~900K | ~15 min |
| Clifford Operator | > 0.85 | < 5 ms | ~50K | ~8 min |
| Mamba (temporal) | > 0.90 (seq) | < 10 ms/step | ~540K | ~10 min |
| Liquid NN | > 0.90 (risk) | < 1 ms | ~30K | ~3 min |
| Diffusion Model | (generative) | ~1 s/sample | ~85K | ~30 min |

> Training times are estimated on an NVIDIA RTX 4060 with 2000 CFD samples.

---

## 13. Quick Reference Table

| Model | File | Key Classes | Input | Output | Default? |
|-------|------|-------------|-------|--------|----------|
| DeepONet | `src/deeponet/model.py` | `DeepONet`, `BranchNet`, `TrunkNet` | `[B,3]`, `[N,3]` | `[B,4,N]` | No (legacy) |
| DeepONetFourier | `src/deeponet/deeponet_fourier.py` | `DeepONetFourier`, `BranchNetFourier`, `TrunkNetFourier` | `[B,3]`, `[N,3]` | `[B,4,N]` | **✅ Yes** |
| Transolver++ | `src/operators/transolver_operator.py` | `TransolverOperator`, `MeshEmbedding` | `[B,3]`, `[N,3]` | `[B,4,N]` | No |
| Clifford | `src/operators/clifford_operator.py` | `CliffordNeuralOperator`, `CliffordLinear` | `[B,3]`, `[N,3]` | `[B,4,N]` | No |
| Mamba | `src/temporal/mamba_operator.py` | `MambaTemporalOperator`, `SelectiveSSM` | `[B,T,16]` | `[B,T,16]` | No (Tier 3) |
| Liquid NN | `src/temporal/liquid_nn_sensor_model.py` | `LiquidNNSensorModel`, `CfCCell` | `[B,T,8]`, `[B,T]` | `[B,32]`, `[B,1]` | No (Tier 3) |
| Diffusion | `src/generative/diffusion_turbulence_model.py` | `DiffusionTurbulenceModel`, `TurbulenceDenoisingUNet` | `[B,4,N]`, `[B,N]` | `[n,4,N]` | No (Tier 3) |
| Sobolev Loss | `src/deeponet/sobolev_loss.py` | `SobolevLoss` | `pred`, `target` | scalar | With DeepONetFourier |
| Divergence | `src/physics/divergence_penalty.py` | `DivergencePenalty` | `[B,4,N]` | scalar | With DeepONetFourier |
| LOCA Classifier | `src/accident_model/train_locac_model.py` | `LOCACDetector` | `[B,7]` (NPPAD signals) | probability | Downstream |

---

*Generated from source code analysis of `src/` directory. All parameter counts are approximate and depend on exact configuration values in `configs/model_config.yaml`.*
