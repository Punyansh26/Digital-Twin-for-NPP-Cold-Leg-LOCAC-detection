"""
Wall Shear Stress Calculator for AP1000 Digital Twin.

Computes wall shear stress (WSS) from predicted velocity fields:

    τ_w = μ · ∂u/∂y

Near-wall gradients are obtained from the predicted velocity field via
finite differences over the radially-sorted mesh nodes.

WSS maps are saved as numpy archives and can be used to detect
corrosion-prone regions in the AP1000 cold-leg piping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


class WallShearCalculator:
    """
    Compute and analyse wall shear stress from velocity predictions.

    Args:
        dynamic_viscosity:       μ in Pa·s.  Water near 300 °C ≈ 8.5 × 10⁻⁵.
        pipe_radius:             Pipe radius in metres (D = 0.7 m → R = 0.35 m).
        wall_threshold_fraction: Fraction of R that defines the "near-wall"
                                 region  (default 0.10 → outer 10 % of radius).
    """

    def __init__(
        self,
        dynamic_viscosity: float = 8.5e-5,
        pipe_radius: float = 0.35,
        wall_threshold_fraction: float = 0.10,
    ) -> None:
        self.mu            = dynamic_viscosity
        self.R             = pipe_radius
        self.wall_fraction = wall_threshold_fraction

    # ------------------------------------------------------------------
    # Node identification
    # ------------------------------------------------------------------

    def identify_wall_nodes(self, coords: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask selecting near-wall mesh nodes.

        Radial distance is measured in the y-z plane (pipe cross-section).

        Args:
            coords: [N, 3] spatial coordinates.

        Returns:
            wall_mask: [N] bool array.
        """
        r = np.sqrt(coords[:, 1] ** 2 + coords[:, 2] ** 2)
        r_max = np.max(r)
        if r_max < 1e-12:                  # 2-D or degenerate mesh
            r     = np.abs(coords[:, 1])
            r_max = np.max(r)
        return r >= r_max * (1.0 - self.wall_fraction)

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _compute_wall_gradient(
        self,
        velocity: np.ndarray,
        coords: np.ndarray,
        wall_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Finite-difference ∂u/∂r at each near-wall node.

        For each wall node, the nearest interior node in the radial
        direction is found and a one-sided finite difference is applied.

        Args:
            velocity:  [N] velocity magnitude.
            coords:    [N, 3] spatial coordinates.
            wall_mask: [N] bool selecting wall nodes.

        Returns:
            grads: [N_wall] gradient values.
        """
        wall_idx  = np.where(wall_mask)[0]
        inner_idx = np.where(~wall_mask)[0]

        if len(wall_idx) == 0 or len(inner_idx) == 0:
            return np.zeros(int(wall_mask.sum()))

        r_all   = np.sqrt(coords[:, 1] ** 2 + coords[:, 2] ** 2)
        grads   = np.empty(len(wall_idx))

        r_inner = r_all[inner_idx]
        x_inner = coords[inner_idx, 0]

        for k, wi in enumerate(wall_idx):
            r_w  = r_all[wi]
            u_w  = velocity[wi]
            x_w  = coords[wi, 0]

            # Combined radial + axial proximity metric
            dr   = r_inner - r_w                        # signed radial diff
            dx   = x_inner - x_w                        # axial diff
            dist = np.hypot(dr, dx)

            nearest = inner_idx[np.argmin(dist)]
            r_n     = r_all[nearest]
            u_n     = velocity[nearest]
            delta_r = abs(r_w - r_n) + 1e-14

            grads[k] = abs(u_n - u_w) / delta_r

        return grads

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_wss(
        self,
        velocity_field: np.ndarray,
        coords: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Compute wall shear stress map.

        Args:
            velocity_field: [N] velocity magnitude  *or*  [N, 3] vector.
            coords:         [N, 3] spatial coordinates.

        Returns:
            dict with keys:
                ``wss``          — [N_wall] WSS values (Pa)
                ``wall_coords``  — [N_wall, 3] coordinates of wall nodes
                ``wall_mask``    — [N] boolean mask
                ``mean_wss``     — scalar mean WSS (Pa)
                ``max_wss``      — scalar max WSS (Pa)
        """
        if velocity_field.ndim == 2:
            vel_mag = np.linalg.norm(velocity_field, axis=1)
        else:
            vel_mag = velocity_field.ravel()

        wall_mask = self.identify_wall_nodes(coords)

        if wall_mask.sum() == 0:
            return {
                "wss":         np.zeros(0),
                "wall_coords": np.zeros((0, 3)),
                "wall_mask":   wall_mask,
                "mean_wss":    0.0,
                "max_wss":     0.0,
            }

        du_dr = self._compute_wall_gradient(vel_mag, coords, wall_mask)
        wss   = self.mu * du_dr

        return {
            "wss":         wss,
            "wall_coords": coords[wall_mask],
            "wall_mask":   wall_mask,
            "mean_wss":    float(np.mean(wss)),
            "max_wss":     float(np.max(wss)) if len(wss) else 0.0,
        }

    # ------------------------------------------------------------------
    def assess_corrosion_risk(
        self,
        wss_results: Dict,
        low_threshold: float = 1.0,
        high_threshold: float = 10.0,
    ) -> Dict[str, Union[str, float]]:
        """
        Classify corrosion risk from the WSS map.

        Low  WSS → flow stagnation → deposit / corrosion.
        High WSS → erosion-corrosion.

        Args:
            wss_results:    Output of :meth:`compute_wss`.
            low_threshold:  Lower bound for acceptably active flow (Pa).
            high_threshold: Upper bound before erosion concern (Pa).

        Returns:
            dict with risk_level, mean_wss, max_wss, fraction metrics.
        """
        wss = wss_results["wss"]
        if len(wss) == 0:
            return {"risk_level": "UNKNOWN", "mean_wss": 0.0, "max_wss": 0.0}

        mean_wss = float(np.mean(wss))
        max_wss  = float(np.max(wss))
        f_low    = float(np.mean(wss < low_threshold))
        f_high   = float(np.mean(wss > high_threshold))

        if f_low > 0.30:
            risk_level = "HIGH (stagnation zones detected)"
        elif f_high > 0.20:
            risk_level = "HIGH (erosion-corrosion zones detected)"
        elif f_low > 0.10 or f_high > 0.10:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "risk_level":         risk_level,
            "mean_wss":           mean_wss,
            "max_wss":            max_wss,
            "low_wss_fraction":   f_low,
            "high_wss_fraction":  f_high,
            "locac_correlation":  min(1.0, f_low + f_high),
        }

    # ------------------------------------------------------------------
    def save_wss_map(
        self,
        wss_results: Dict,
        output_path: Union[str, Path],
    ) -> None:
        """Save WSS map to a compressed numpy archive (.npz)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            wss         = wss_results["wss"],
            wall_coords = wss_results["wall_coords"],
            mean_wss    = wss_results["mean_wss"],
            max_wss     = wss_results["max_wss"],
        )
        print(f"✓ WSS map saved: {output_path}")
