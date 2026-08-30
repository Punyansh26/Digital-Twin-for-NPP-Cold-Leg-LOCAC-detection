# Edit 006 — Sobolev Gradient Reconstruction

**Date**: 2026-08-30
**Requested by**: Professor/Reviewer (via "[Input/confirmation required]" placeholder)
**Paper section**: Sec III-A — Improvement 3: Sobolev Gradient-Enhanced Loss
**Edit type**: replacement
**Status**: ready-for-review

---

## Context

The paper contained a placeholder requesting confirmation of the gradient reconstruction scheme used for the Sobolev loss target gradients:
"[Input/confirmation required from the students: confirm which specific reconstruction scheme (Green–Gauss, least-squares, k-NN finite difference, or other) was used to compute target-field gradients, and state it explicitly here.]"

The original text claimed these gradients were precomputed using a mesh-based reconstruction and cached in the HDF5 dataset because central finite differences are not well-defined on unstructured meshes.

---

## Evidence from Codebase / Literature

A review of the codebase shows that the target gradients are **not** precomputed or cached in the HDF5 dataset (checked `src/preprocessing/prepare_deeponet_data.py` and `scripts/generate_mock_data.py`). 

Instead, `src/deeponet/sobolev_loss.py` dynamically computes the target gradient using a 1D central finite-difference along the flattened node array:

`src/deeponet/sobolev_loss.py` (Lines 58-66):
```python
    @staticmethod
    def _fd_gradient(u: torch.Tensor) -> torch.Tensor:
        """
        Central finite-difference gradient along the spatial (last) axis.

        u: [B, n_fields, N] → grad: [B, n_fields, N]
        """
        u_pad = F.pad(u, (1, 1), mode="replicate")
        return (u_pad[..., 2:] - u_pad[..., :-2]) / 2.0
```

Furthermore, the mock dataset generation in `scripts/generate_mock_data.py` spawns nodes completely at random without radial or connectivity sorting, meaning this 1D finite-difference is taking the difference between physically uncorrelated points in the pipe.

---

## Proposed Text

<Section: Sec III-A — Improvement 3: Sobolev Gradient-Enhanced Loss>

> [DROP-IN REPLACEMENT / ADDITION BEGINS]

with $\beta = 0.1$. The two gradient terms in Eq. (14) are computed differently: the predicted gradient $\nabla\hat{s}_{bn}$ is obtained via automatic differentiation of the trunk network with respect to its continuous coordinate input $y$ (well-defined regardless of mesh connectivity), while the target gradient $\nabla s_{bn}$ is computed dynamically during training using a simplified 1D central finite-difference approximation along the serialized node array [VERIFY: The codebase currently applies sequential finite differences directly over the unsorted node sequence lacking explicit connectivity-based physical reconstruction. The team must verify whether to justify this computational simplification or implement a geometric reconstruction (e.g., Green-Gauss) prior to final publication]. This directly penalizes smoothed-over pressure drops and velocity boundary layers—the features most critical for LOCA screening.

> [DROP-IN REPLACEMENT / ADDITION ENDS]

---

## Notes / Caveats

- The current implementation (`_fd_gradient`) applies a 1D sequence difference over the flattened node axis. Since the nodes are randomly sampled and unsorted (`generate_mock_data.py`), this operation lacks geometric/physical validity.
- The team must address this before final validation. They will either need to implement a true mesh-based gradient reconstruction (like Green-Gauss) in `prepare_deeponet_data.py` or sort the nodes radially to at least approximate near-wall gradients (as was attempted in `wall_shear_calculator.py`).
