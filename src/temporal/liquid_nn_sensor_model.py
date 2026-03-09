"""
Liquid Neural Network Sensor Model for AP1000 Digital Twin.

Uses Closed-form Continuous-time (CfC) neurons to process reactor sensor
time series with irregular sampling intervals.

    Input:  sensor observations  [B, T, n_sensors]
            observation timestamps [B, T]   (in seconds, may be irregular)
    Output: latent reactor state  [B, latent_dim]
            LOCAC risk probability [B, 1]

The CfC neuron solves the following ODE analytically:

    h(t + Δt) = h(t) · exp(-Δt · τ(x,h))
              + (1 - exp(-Δt · τ(x,h))) · g(x, h)

References:
    Hasani et al., "Closed-form Continuous-time Neural Networks",
    Nature Machine Intelligence, 2022.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CfC neuron cell
# ---------------------------------------------------------------------------

class CfCCell(nn.Module):
    """
    Single-layer Closed-form Continuous-time (CfC) cell.

    Args:
        input_size:  Input dimension at each timestep.
        hidden_size: Hidden state dimension.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Backbone: joint nonlinear feature of input + state
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Gating network: σ(W_x·x + W_h·h + b)  → controls which info flows
        self.gate_x = nn.Linear(input_size,  hidden_size)
        self.gate_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_b = nn.Parameter(torch.zeros(hidden_size))

        # Time-constant: softplus ensures positivity
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))   # τ = softplus(log_tau)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x:        torch.Tensor,    # [B, input_size]
        h:        torch.Tensor,    # [B, hidden_size]
        delta_t:  torch.Tensor,    # [B] or scalar — time elapsed (seconds)
    ) -> torch.Tensor:
        """Returns updated hidden state [B, hidden_size]."""
        g  = torch.sigmoid(self.gate_x(x) + self.gate_h(h) + self.gate_b)
        f  = self.backbone(torch.cat([x, h], dim=-1))   # target state
        tau = F.softplus(self.log_tau) + torch.abs(g)   # effective time constant

        # Expand Δt to [B, hidden_size]
        if delta_t.dim() == 0:
            dt = delta_t.expand(x.shape[0], 1)
        elif delta_t.dim() == 1:
            dt = delta_t.unsqueeze(1)
        else:
            dt = delta_t
        dt = dt.expand(-1, self.hidden_size)

        decay = torch.exp(-dt * tau)                    # [B, H]
        return decay * h + (1.0 - decay) * f


# ---------------------------------------------------------------------------
# Multi-layer Liquid NN sensor model
# ---------------------------------------------------------------------------

class LiquidNNSensorModel(nn.Module):
    """
    Multi-layer Liquid Neural Network for reactor sensor processing.

    Processes irregularly-sampled reactor sensor time series into a latent
    state that condenses the reactor's current safety status.

    Args:
        n_sensors:    Number of sensor input channels (default 8).
        hidden_size:  CfC hidden state dimension per layer (default 64).
        latent_dim:   Output latent state dimension (default 32).
        n_layers:     Number of stacked CfC layers (default 2).
    """

    def __init__(
        self,
        n_sensors:   int = 8,
        hidden_size: int = 64,
        latent_dim:  int = 32,
        n_layers:    int = 2,
    ) -> None:
        super().__init__()
        self.n_sensors   = n_sensors
        self.hidden_size = hidden_size
        self.latent_dim  = latent_dim
        self.n_layers    = n_layers

        self.input_norm  = nn.LayerNorm(n_sensors)

        self.cfc_cells = nn.ModuleList()
        for i in range(n_layers):
            in_dim = n_sensors if i == 0 else hidden_size
            self.cfc_cells.append(CfCCell(in_dim, hidden_size))

        # Latent state projection
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.Tanh(),
        )

        # LOCAC risk head (binary output)
        self.risk_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "LiquidNNSensorModel":
        cfg = config.get("liquid_nn", {})
        return cls(
            n_sensors   = cfg.get("n_sensors",   8),
            hidden_size = cfg.get("hidden_size", 64),
            latent_dim  = cfg.get("latent_dim",  32),
            n_layers    = cfg.get("n_layers",    2),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        sensor_series: torch.Tensor,              # [B, T, n_sensors]
        timestamps:    Optional[torch.Tensor] = None,  # [B, T] seconds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sensor_series: [B, T, n_sensors]
            timestamps:    [B, T] in seconds.  If None, assumes Δt = 1 s.

        Returns:
            latent_state: [B, latent_dim]
            risk_prob:    [B, 1]   LOCAC risk probability ∈ [0, 1]
        """
        B, T, _ = sensor_series.shape

        # Time deltas
        if timestamps is not None:
            dt_series = timestamps.clone()
            dt_series[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
            dt_series[:, 0]  = 1.0
            dt_series        = dt_series.clamp(min=1e-6)
        else:
            dt_series = torch.ones(B, T, device=sensor_series.device)

        # Initial hidden states
        hiddens = [
            torch.zeros(B, self.hidden_size, device=sensor_series.device)
            for _ in range(self.n_layers)
        ]

        # Sequential processing
        for t in range(T):
            x_t  = self.input_norm(sensor_series[:, t, :])
            dt_t = dt_series[:, t]
            for layer_idx, cell in enumerate(self.cfc_cells):
                x_t          = cell(x_t, hiddens[layer_idx], dt_t)
                hiddens[layer_idx] = x_t

        final_h    = hiddens[-1]                        # [B, hidden_size]
        latent     = self.latent_proj(final_h)          # [B, latent_dim]
        risk       = self.risk_head(latent)             # [B, 1]
        return latent, risk

    # ------------------------------------------------------------------
    def predict_risk(
        self,
        sensor_series: torch.Tensor,
        timestamps:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience wrapper returning only risk probability [B, 1]."""
        _, risk = self.forward(sensor_series, timestamps)
        return risk

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
