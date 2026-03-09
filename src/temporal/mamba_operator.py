"""
Mamba-style Temporal Neural Operator for AP1000 Digital Twin.

Models long transient sequences of flow states using a Selective State Space
Model (SSM).  Scales linearly O(T) with sequence length, compared to O(T²)
for transformer-based approaches.

Usage:
    Input  — sequence of flow states:  [B, T, state_dim]
    Output — predicted next states:    [B, T, state_dim]

    For autoregressive forecasting use ``predict_sequence``.

References:
    Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces", arXiv 2312.00752 (2023).
    (This module implements a self-contained SSM without the Mamba CUDA
     kernel dependency for maximum portability.)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective SSM cell
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Simplified selective scan SSM.

    The selectivity comes from input-dependent (Δ, B, C) projections that
    gate which parts of the history to retain.

    Args:
        d_model: Model (inner) dimension.
        d_state: Latent SSM state dimension (default 16).
        d_conv:  Depthwise conv kernel size (default 4).
        expand:  Channel-expansion ratio (default 2).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * expand

        # Input expansion
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local depthwise convolution (causal)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM projections: dt (scalar), B (state), C (state)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        # Log-A matrix — diagonal absorbs per-channel time constants
        A = torch.arange(1, d_state + 1, dtype=torch.float)
        A = A.unsqueeze(0).expand(self.d_inner, -1)     # [d_inner, d_state]
        self.A_log = nn.Parameter(torch.log(A))

        # Skip-connection scale
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def _selective_scan(
        self,
        u:      torch.Tensor,    # [B, T, d_inner]
        delta:  torch.Tensor,    # [B, T, d_inner]
        A:      torch.Tensor,    # [d_inner, d_state]
        B_mat:  torch.Tensor,    # [B, T, d_state]
        C_mat:  torch.Tensor,    # [B, T, d_state]
    ) -> torch.Tensor:
        B, T, D = u.shape
        S       = self.d_state

        # Discretise: dA[b,t,d,s] = exp(delta[b,t,d] * A[d,s])
        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # [B, T, D, S]

        # Discretise B: dB[b,t,d,s] = delta[b,t,d] * B_mat[b,t,s]
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)   # [B, T, D, S]

        # Sequential scan
        h  = torch.zeros(B, D, S, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            h  = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            y  = (h * C_mat[:, t].unsqueeze(1)).sum(-1)   # [B, D]
            ys.append(y)

        return torch.stack(ys, dim=1)   # [B, T, D]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]  →  [B, T, d_model]"""
        B, T, _ = x.shape

        xz    = self.in_proj(x)                       # [B, T, 2*d_inner]
        x_d, z = xz.chunk(2, dim=-1)                 # [B, T, d_inner] × 2

        # Causal conv1d
        x_d = F.silu(
            self.conv1d(x_d.transpose(1, 2))[..., :T].transpose(1, 2)
        )

        # Input-dependent SSM parameters
        A      = -torch.exp(self.A_log)               # [d_inner, d_state]
        ssm_in = self.x_proj(x_d)                    # [B, T, 2S+1]
        dt_raw, B_mat, C_mat = ssm_in.split(
            [1, self.d_state, self.d_state], dim=-1
        )
        delta  = F.softplus(self.dt_proj(dt_raw))     # [B, T, d_inner]

        y = self._selective_scan(x_d, delta, A, B_mat, C_mat)
        y = y + self.D * x_d                          # skip
        y = y * F.silu(z)                             # gate
        return self.out_proj(y)                       # [B, T, d_model]


# ---------------------------------------------------------------------------
# Mamba block with residual + layer norm
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = SelectiveSSM(d_model, d_state, d_conv)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ssm(self.norm(x)))


# ---------------------------------------------------------------------------
# MambaTemporalOperator — public API
# ---------------------------------------------------------------------------

class MambaTemporalOperator(nn.Module):
    """
    Mamba-based temporal operator for LOCAC transient sequence modelling.

    Predicts the next flow state from a history of states.  For long
    sequences it scales linearly in time (vs. quadratic for transformers).

    Args:
        state_dim: Dimension of the flow state vector (default 16).
        d_model:   Internal model dimension (default 128).
        n_layers:  Number of Mamba blocks (default 4).
        d_state:   SSM latent state size (default 16).
        dropout:   Dropout probability (default 0.1).
    """

    def __init__(
        self,
        state_dim: int = 16,
        d_model:   int = 128,
        n_layers:  int = 4,
        d_state:   int = 16,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, state_dim)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "MambaTemporalOperator":
        cfg = config.get("mamba", {})
        return cls(
            state_dim = cfg.get("state_dim", 16),
            d_model   = cfg.get("d_model",   128),
            n_layers  = cfg.get("n_layers",  4),
            d_state   = cfg.get("d_state",   16),
            dropout   = cfg.get("dropout",   0.1),
        )

    # ------------------------------------------------------------------
    def forward(self, flow_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow_states: [B, T, state_dim] — sequence of flow states.

        Returns:
            next_states: [B, T, state_dim] — next-step predictions.
        """
        h = self.encoder(flow_states)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.decoder(h)

    # ------------------------------------------------------------------
    def predict_sequence(
        self,
        initial_state: torch.Tensor,
        n_steps:       int,
        context_len:   int = 64,
    ) -> torch.Tensor:
        """
        Autoregressive prediction from an initial state.

        Args:
            initial_state: [B, state_dim]
            n_steps:       Total steps to predict (including initial).
            context_len:   Maximum history window to feed to the model.

        Returns:
            trajectory: [B, n_steps, state_dim]
        """
        trajectory: List[torch.Tensor] = [initial_state]

        for _ in range(n_steps - 1):
            ctx = torch.stack(trajectory, dim=1)          # [B, t, state_dim]
            ctx = ctx[:, -context_len:]                   # trim context
            preds = self.forward(ctx)                     # [B, ctx, state_dim]
            trajectory.append(preds[:, -1, :])            # [B, state_dim]

        return torch.stack(trajectory, dim=1)             # [B, n_steps, state_dim]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
