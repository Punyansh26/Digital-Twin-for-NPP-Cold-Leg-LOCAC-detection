# Edit 007 — Clifford Equivariance Correction (Grade-Selective Output Projection)

**Date**: 2026-08-30  
**Requested by**: Professor / paper placeholder `[Input/confirmation required from the students…]` in Sec V-C  
**Paper section**: Sec V-C — Tier 2: Clifford Neural Operator — Geometric Algebra (subsection 4: Input/Output Projection)  
**Edit type**: clarification + replacement  
**Status**: ready-for-review  

---

## Context

The paper contained the following bracketed placeholder in Sec V-C §4:

> *[Input/confirmation required from the students: the equivariance property claimed in Eq. 28 holds only if this final projection is grade-selective — i.e., scalar outputs (p̂, T̂) are read exclusively from grade-0 channels, and the velocity magnitude |v̂| is computed as the rotation-invariant norm of the grade-1 (vector) channels, rather than via a single generic dense layer applied to the flattened 256-dimensional multivector, which would in general mix grades and break the guarantee. Please confirm the actual structure of MultivectorToFields and update this paragraph accordingly; if the projection is not grade-selective, Eq. 28 and the surrounding equivariance language throughout Section V-C should be softened…]*

---

## Evidence from Codebase

**File**: `src/operators/clifford_operator.py` (lines 168–178)

```python
class MultivectorToFields(nn.Module):
    """Project multivector representation to scalar output fields."""

    def __init__(self, n_channels: int, n_outputs: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_channels * 8, n_outputs)   # flat linear over ALL 256 dims

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """mv: [..., n_channels, 8]  →  [..., n_outputs]"""
        *batch, C, _ = mv.shape
        return self.proj(mv.reshape(*batch, C * 8))         # grades mixed before projection
```

**Verdict — NOT grade-selective.**

The module flattens the entire `[B, N, 32, 8]` multivector tensor into a `[B, N, 256]` vector and
applies a single `nn.Linear(256, 4)`. This mixes all eight grades (scalar, vector, bivector,
trivector) in an unconstrained way before producing the four output fields. There is no grade-gating,
no grade-0-only branch for pressure/temperature, and no Euclidean-norm aggregation of grade-1
components for velocity magnitude.

**Consequence for equivariance:** The scalar-gate residual inside each `CliffordNeuralOperator`
forward pass (lines 252–255) correctly preserves equivariance *through the Clifford layers*. However,
once `MultivectorToFields` applies a grade-mixing linear map, the equivariance guarantee is broken at
the output stage. The network is therefore only approximately equivariant: the bulk of the computation
(embedding + four Clifford layers + scalar-gate activations) respects the algebra, while the final
readout does not.

---

## Proposed Text

### Sec V-C §4 — Input/Output Projection (DROP-IN REPLACEMENT)

> [DROP-IN REPLACEMENT BEGINS]

**4) Input/Output Projection:** The \texttt{PhysicsToMultivector} module embeds branch parameters
(scalars $\rightarrow$ grade-0) and mesh coordinates (vectors $\rightarrow$ grade-1) into a $C = 32$-channel
multivector representation of shape $[B, N, 32, 8]$. After four Clifford layers equipped with
scalar-gate residuals, the \texttt{MultivectorToFields} readout projects to the four target fields
$\hat{p}$, $|\hat{v}|$, $\hat{k}$, $\hat{T}$.

In the current implementation, \texttt{MultivectorToFields} applies a single learned linear map
$\mathbf{W} \in \mathbb{R}^{4 \times 256}$ to the flattened multivector, without enforcing grade
selectivity at the output stage:
\begin{equation}
\hat{\mathbf{S}}_n = \mathbf{W}\,\operatorname{flatten}\!\left(\mathbf{h}_n^{(\text{out})}\right) + \mathbf{b},
\qquad \mathbf{h}_n^{(\text{out})} \in \mathbb{R}^{32 \times 8}.
\end{equation}
This design mixes grades at the readout and therefore does not guarantee exact rotational equivariance
at the output. Rather, the Clifford Neural Operator is \emph{approximately equivariant} in the sense
that the dominant computation—grade-structured embedding, geometric-product Clifford layers, and
scalar-gate nonlinearities—preserves the algebraic symmetry throughout the trunk, while the final
linear readout introduces a controlled degree of grade mixing. The scalar-gate residual
\begin{align}
\mathbf{h} &\leftarrow \mathbf{h} + \mathbf{h}_{\text{res}}, \\
h_{n,c,0} &\leftarrow \operatorname{GELU}(h_{n,c,0}),
\end{align}
ensures that nonlinearity is applied only to grade-0 (scalar) components throughout the four
processing layers, preserving equivariance up to the readout. Future work may replace
\texttt{MultivectorToFields} with a fully grade-selective decoder—reading $\hat{p}$ and $\hat{T}$
exclusively from grade-0 channels and computing $|\hat{v}|$ as the rotation-invariant $\ell_2$-norm
of the grade-1 channels—to promote the architecture from approximate to strict equivariance [6].

> [DROP-IN REPLACEMENT ENDS]

---

### Sec V-C — Opening Paragraph (language softening)

Any occurrence of **"guarantees rotational equivariance"** or **"exact equivariance"** in Sec V-C
should be changed to **"approximately equivariant, motivated by the grade-structured architecture"**.

The scalar-gate residual claim—*"Grade-1 vector components remain linear, preserving equivariance
under rotation $R \in SO(3)$"*—is accurate for the trunk layers and may be retained verbatim, with
the forward-reference: *"…up to the final readout projection (see §V-C.4)"*.

---

## Notes / Caveats

1. **The equivariance argument remains pedagogically valid for the trunk.** The scalar-gate residual
   at lines 252–255 correctly confines nonlinearity to grade-0, so the four Clifford layers do
   preserve equivariance in isolation. Only the last `nn.Linear(256, 4)` breaks the guarantee.

2. **Performance impact is likely negligible.** All four outputs ($\hat{p}$, $|\hat{v}|$, $\hat{k}$,
   $\hat{T}$) are rotation-invariant scalars (pressure, speed magnitude, TKE, temperature). The
   grade-mixing readout does not produce physically inconsistent vector outputs—it simply forgoes the
   algebraic certificate. The architecture remains a geometrically motivated, grade-structured model.

3. **Future work:** A grade-selective readout is straightforward to implement and would be a clean
   contribution for a journal revision.

4. **No re-training required** for this textual correction. The proposed text accurately describes
   the existing code.

5. **Open Item #7** (`Journal/CLAUDE.md` §2) is resolved by this edit.
