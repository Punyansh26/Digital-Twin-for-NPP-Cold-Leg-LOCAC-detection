# Edit 004 — Divergence-Free Penalty & Output Layer Clarification (Sec III)

**Date**: 2026-08-27  
**Requested by**: Professor / "[Input/confirmation required]" placeholder in Sec III  
**Paper section**: Sec III — Problem Formulation  
**Edit type**: clarification + replacement  
**Status**: ready-for-review  

---

## Context

The Sec III draft contained the following placeholder (quoted verbatim):

> *"[Input/confirmation required from the students: confirm that the branch-trunk
> implementation does in fact predict (v̂_x, v̂_y, v̂_z) internally before reducing
> to |v̂|, and specify the corresponding output layer width/shape here and in
> Section V-A. If instead only the scalar magnitude is predicted, the
> divergence-free penalty as currently formulated is not computable and Eq. 15
> must be reformulated or removed.]"*

The question asks: does the model internally predict the full velocity vector
**(v̂_x, v̂_y, v̂_z)** and then derive the scalar magnitude, or does it directly
predict only **|v̂|**?

---

## Evidence from Codebase

### 1. `DeepONetFourier` output layer — `src/deeponet/deeponet_fourier.py`

```
self.n_outputs    = cfg.get("n_outputs", 4)
self.output_fields = cfg.get(
    "output_fields",
    ["pressure", "velocity_magnitude", "turbulence_k", "temperature"],
)
...
self.branch_nets = nn.ModuleList([
    BranchNetFourier(b_input, b_hidden, b_out)     # one head per output
    for _ in range(self.n_outputs)
])
```

`forward()` returns `[B, n_outputs=4, N]`.

**Finding**: The model directly predicts **4 scalar fields**:
`[pressure, velocity_magnitude, turbulence_k, temperature]`.
The velocity vector components (v_x, v_y, v_z) are **not** predicted as
intermediate outputs. The network learns scalar |v̂| directly as output head index 1.

### 2. `DivergencePenalty.forward()` — `src/physics/divergence_penalty.py` (L118–142)

```python
def forward(self, predictions, coords=None):
    # Velocity magnitude field (index 1)
    vel_mag = predictions[:, 1, :]           # [B, N]
    div_proxy = self._central_diff(vel_mag)  # [B, N]
    return self.weight * torch.mean(div_proxy ** 2)
```

**Finding**: The operative training path computes a **proxy divergence** —
the 1-D central finite-difference of the scalar velocity-magnitude field along
the sorted mesh-node index — not the true vector divergence
∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z.
A full-vector method (`compute_full_divergence_penalty`) and autograd path exist
but are **not called** in standard training.

### 3. `train.py`

No references to `divergence` found in `src/deeponet/train.py`. The flag
`use_divergence_penalty: true` routes through `DivergencePenalty.forward()`
(the proxy path only).

### Summary

| Claim in paper draft (Eq. 15)                          | Actual implementation        |
|--------------------------------------------------------|------------------------------|
| Model predicts (v̂_x, v̂_y, v̂_z) internally           | ✗ — only scalar |v̂| predicted |
| Divergence = ‖∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z‖²          | ✗ — proxy: ‖Δ_h|v̂|‖²        |
| Output layer width includes 3 velocity components      | ✗ — n_outputs = 4 scalars    |

---

## Proposed Text

### Section III — Problem Formulation
*(Replace the bracketed "[Input/confirmation required…]" paragraph with the following)*

> [DROP-IN REPLACEMENT BEGINS]

The branch-trunk network predicts **four scalar output fields** simultaneously:
pressure $\hat{p}$, velocity magnitude $|\hat{\mathbf{v}}|$, turbulent kinetic
energy $\hat{k}$, and normalised temperature $\hat{\tilde{T}}$, yielding the
output tensor $\hat{S} \in \mathbb{R}^{B \times 4 \times N}$.  Each of the four
output heads consists of one independent branch sub-network
($\mathbb{R}^{3} \to \mathbb{R}^{256}$) and one independent Fourier-enhanced
trunk sub-network ($\mathbb{R}^{3} \to \mathbb{R}^{256}$); a scalar bias is added
to each inner product, giving output layer widths of $\mathbb{R}^{256}$ per head.

The velocity magnitude $|\hat{\mathbf{v}}|$ is predicted \emph{directly} as a
scalar field; the network does not internally resolve the three Cartesian
velocity components $(\hat{v}_x, \hat{v}_y, \hat{v}_z)$.  Consequently, the
incompressibility regularisation applied during training (Section~V-A,
Eq.~\ref{eq:div}) is implemented as a \emph{proxy divergence} — the
root-mean-square spatial variation of the predicted velocity-magnitude field
computed via central finite differences along the sorted mesh-node index — rather
than the exact vector divergence $\nabla \cdot \hat{\mathbf{v}}$.  While this
proxy does not enforce strict point-wise incompressibility, it penalises
unphysical large-scale spatial fluctuations in the velocity field and is
consistent with the scalar-output formulation of the surrogate.  Extending the
output to predict explicit vector components and enforce true incompressibility
is identified as a direction for future work (Section~IX).

> [DROP-IN REPLACEMENT ENDS]

---

### Section V-A — Physics-Informed Regularisation
*(Update Eq. 15 and its surrounding description)*

> [REPLACEMENT BEGINS — replaces the current Eq. 15 block]

The divergence-free physics regularisation term is defined as:
$$
\mathcal{L}_{\mathrm{div}} = \lambda \left\| \Delta_h |\hat{\mathbf{v}}| \right\|^2_F
\tag{15}
$$
where $\Delta_h$ denotes the central finite-difference operator applied along the
sorted spatial node index and $\lambda = 0.01$ is the penalty weight
(see \texttt{configs/model\_config.yaml}).  Because the surrogate predicts the
scalar velocity magnitude rather than the full velocity vector, the exact
divergence operator $\nabla \cdot \hat{\mathbf{v}}$ is not directly computable.
Equation~\eqref{eq:div} therefore acts as a proxy incompressibility constraint:
it penalises large point-wise variation in $|\hat{\mathbf{v}}|$ over the mesh,
which regularises the velocity field against unphysical oscillations without
requiring explicit vector-component outputs.

> [REPLACEMENT ENDS]

---

## Notes / Caveats

1. **Eq. 15 must be updated** in the LaTeX source. The current form ‖∇·u‖² is
   physically misleading; the proposed replacement ‖Δ_h|v̂|‖² is accurate.

2. **Future-work upgrade path** (for Section IX, if desired): set
   `n_outputs = 6`, add (v_x, v_y, v_z) heads, derive magnitude as
   `|v̂| = sqrt(v_x² + v_y² + v_z²)`, and route
   `compute_full_divergence_penalty(vx, vy, vz)` from the training loop.

3. **[VERIFIED ✓]**: `grep -r "compute_full_divergence" src/ scripts/` (run
   2026-08-27) returns hits **only inside `src/physics/divergence_penalty.py`**
   (docstring + method definition). No training script or other module calls
   `compute_full_divergence_penalty`. The proxy-only path is confirmed.

4. **Open Item #3** from `Journal/CLAUDE.md` is now **closed** (answer: scalar
   only; proxy penalty). The `edit_NNN` table in `CLAUDE.md` may be updated
   accordingly.

5. Open Item #4 (Sobolev gradient reconstruction scheme) remains open and
   should be addressed in a subsequent edit.
