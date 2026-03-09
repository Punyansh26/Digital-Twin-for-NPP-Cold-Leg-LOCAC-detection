"""
Diffusion-based Turbulence Super-Resolution Model.

Pipeline:
    1. DeepONet predicts mean flow field (pressure, velocity, TKE, temperature).
    2. This module generates stochastic turbulence fluctuations conditioned on
       the mean flow and turbulence kinetic energy (TKE) field.
    3. Multiple realisations provide uncertainty quantification.

Architecture: DDPM (Ho et al. 2020) with a 1-D U-Net denoiser operating
over spatial node profiles.  Conditioning is injected via the mean-flow
statistics vector.

Usage:
    model = DiffusionTurbulenceModel(n_nodes=1000, n_fields=4)
    cond  = model.encode_condition(mean_fields, tke_field)
    samples = model.sample(mean_fields, tke_field, n_samples=5)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers: sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_timestep_embedding(
    t: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Standard DDPM sinusoidal positional encoding for diffusion timesteps."""
    assert dim % 2 == 0
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# U-Net building block
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """
    1-D residual block with time and conditioning embeddings.

    Args:
        in_ch:     Input channel count.
        out_ch:    Output channel count.
        time_dim:  Timestep embedding dimension.
        cond_dim:  Conditioning vector dimension (0 = no conditioning).
        n_groups:  GroupNorm group count (clamped to min(n_groups, out_ch)).
    """

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        time_dim: int,
        cond_dim: int = 0,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        ng1 = min(n_groups, in_ch)
        ng2 = min(n_groups, out_ch)

        self.norm1  = nn.GroupNorm(ng1, in_ch)
        self.conv1  = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(ng2, out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.cond_mlp = (
            nn.Linear(cond_dim, out_ch) if cond_dim > 0 else None
        )
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(
        self,
        x:    torch.Tensor,                      # [B, in_ch, N]
        t:    torch.Tensor,                      # [B, time_dim]
        cond: Optional[torch.Tensor] = None,     # [B, cond_dim]
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t).unsqueeze(-1)
        if cond is not None and self.cond_mlp is not None:
            h = h + self.cond_mlp(cond).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Denoising U-Net
# ---------------------------------------------------------------------------

class TurbulenceDenoisingUNet(nn.Module):
    """
    Lightweight 1-D U-Net denoiser for turbulence field generation.

    Channel schedule (encoder / decoder) is kept small to remain
    compatible with CPU training during prototyping.
    """

    def __init__(
        self,
        n_fields:    int = 4,
        cond_dim:    int = 8,
        time_dim:    int = 64,
        ch_schedule: Tuple[int, ...] = (32, 64, 64, 32),
    ) -> None:
        super().__init__()
        self.time_dim = time_dim

        # Timestep embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        C0, C1, C2, C3 = ch_schedule

        # Encoder
        self.enc0 = ResBlock1D(n_fields, C0, time_dim, cond_dim)
        self.enc1 = ResBlock1D(C0,       C1, time_dim, cond_dim)

        # Bottleneck
        self.mid  = ResBlock1D(C1, C2, time_dim, cond_dim)

        # Decoder (skip connections from encoder)
        self.dec1 = ResBlock1D(C2 + C1, C3,      time_dim, cond_dim)
        self.dec0 = ResBlock1D(C3 + C0, n_fields, time_dim, cond_dim)

    def forward(
        self,
        x_noisy: torch.Tensor,              # [B, n_fields, N]
        t:       torch.Tensor,              # [B]  diffusion timestep index
        cond:    Optional[torch.Tensor] = None,  # [B, cond_dim]
    ) -> torch.Tensor:
        """Returns predicted noise [B, n_fields, N]."""
        t_raw = sinusoidal_timestep_embedding(t, self.time_dim)
        t_emb = self.time_embed(t_raw)              # [B, time_dim]

        # Encoder
        h0 = self.enc0(x_noisy, t_emb, cond)       # [B, C0, N]
        h1 = self.enc1(h0,      t_emb, cond)       # [B, C1, N]

        # Bottleneck
        h  = self.mid(h1, t_emb, cond)             # [B, C2, N]

        # Decoder
        h  = self.dec1(torch.cat([h, h1], dim=1), t_emb, cond)
        h  = self.dec0(torch.cat([h, h0], dim=1), t_emb, cond)
        return h


# ---------------------------------------------------------------------------
# DDPM scheduler (pure-Python, no CUDA dependency)
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """
    Linear-beta DDPM noise schedule.

    Pre-computes α̅_t, √α̅_t, √(1−α̅_t) tensors for forward/reverse steps.
    """

    def __init__(
        self,
        n_steps:    int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
    ) -> None:
        self.n_steps    = n_steps
        betas           = torch.linspace(beta_start, beta_end, n_steps)
        alphas          = 1.0 - betas
        alpha_bar       = torch.cumprod(alphas, dim=0)

        self.betas              = betas
        self.alphas             = alphas
        self.alpha_bar          = alpha_bar
        self.sqrt_ab            = alpha_bar.sqrt()
        self.sqrt_one_minus_ab  = (1.0 - alpha_bar).sqrt()

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,                       # [B] integer timestep
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward process: q(x_t | x_0) = N(√α̅_t · x_0, (1−α̅_t) I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab  = self.sqrt_ab[t].to(x0.device)
        s_1ab = self.sqrt_one_minus_ab[t].to(x0.device)
        for _ in range(x0.dim() - 1):
            s_ab  = s_ab.unsqueeze(-1)
            s_1ab = s_1ab.unsqueeze(-1)
        return s_ab * x0 + s_1ab * noise, noise


# ---------------------------------------------------------------------------
# DiffusionTurbulenceModel — public API
# ---------------------------------------------------------------------------

class DiffusionTurbulenceModel(nn.Module):
    """
    DDPM-based turbulence super-resolution model.

    Generates stochastic turbulence fluctuation fields conditioned on the
    mean-flow prediction from DeepONet, enabling uncertainty quantification.

    Args:
        n_nodes:      Number of spatial mesh nodes (default 1000).
        n_fields:     Number of flow field channels (default 4).
        cond_dim:     Conditioning vector dimension (default 8).
        n_diff_steps: DDPM forward-process timesteps (default 1000).
    """

    def __init__(
        self,
        n_nodes:      int = 1000,
        n_fields:     int = 4,
        cond_dim:     int = 8,
        n_diff_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.n_nodes      = n_nodes
        self.n_fields     = n_fields
        self.n_diff_steps = n_diff_steps

        # Conditioning encoder: mean-flow statistics → cond vector
        self.cond_encoder = nn.Sequential(
            nn.Linear(n_fields + 1, 64),   # n_fields means + 1 TKE mean
            nn.GELU(),
            nn.Linear(64, cond_dim),
            nn.Tanh(),
        )

        # Denoising network
        self.denoiser  = TurbulenceDenoisingUNet(n_fields=n_fields, cond_dim=cond_dim)
        self.scheduler = DDPMScheduler(n_steps=n_diff_steps)

    # ------------------------------------------------------------------
    def encode_condition(
        self,
        mean_fields: torch.Tensor,   # [B, n_fields, N]
        tke_field:   torch.Tensor,   # [B, N]
    ) -> torch.Tensor:
        """Returns conditioning vector [B, cond_dim]."""
        field_mean = mean_fields.mean(dim=-1)           # [B, n_fields]
        tke_mean   = tke_field.mean(dim=-1, keepdim=True)   # [B, 1]
        cond_input = torch.cat([field_mean, tke_mean], dim=-1)
        return self.cond_encoder(cond_input)

    # ------------------------------------------------------------------
    def training_loss(
        self,
        x0:   torch.Tensor,    # [B, n_fields, N]  clean turbulence fluctuations
        cond: torch.Tensor,    # [B, cond_dim]
    ) -> torch.Tensor:
        """DDPM ε-prediction training loss."""
        B   = x0.shape[0]
        t   = torch.randint(0, self.n_diff_steps, (B,), device=x0.device)
        x_t, eps = self.scheduler.q_sample(x0, t)
        eps_pred  = self.denoiser(x_t, t, cond)
        return F.mse_loss(eps_pred, eps)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        mean_fields: torch.Tensor,   # [B, n_fields, N]  (B=1 typical)
        tke_field:   torch.Tensor,   # [B, N]
        n_samples:   int = 5,
    ) -> torch.Tensor:
        """
        Ancestral DDPM sampling to produce turbulence realisations.

        Args:
            mean_fields: Mean-flow prediction from DeepONet.
            tke_field:   Turbulence kinetic energy field from DeepONet.
            n_samples:   Number of stochastic realisations to generate.

        Returns:
            samples: [n_samples, n_fields, N]
        """
        device = mean_fields.device
        N      = mean_fields.shape[-1]

        cond_base = self.encode_condition(mean_fields, tke_field)  # [B, cond_dim]
        cond      = cond_base.expand(n_samples, -1)                # [n_samples, cond_dim]

        x = torch.randn(n_samples, self.n_fields, N, device=device)

        sched = self.scheduler
        for step in reversed(range(self.n_diff_steps)):
            t_tensor = torch.full((n_samples,), step, device=device, dtype=torch.long)
            eps_pred = self.denoiser(x, t_tensor, cond)

            alpha     = sched.alphas[step].to(device)
            alpha_bar = sched.alpha_bar[step].to(device)
            beta      = sched.betas[step].to(device)

            # DDPM reverse mean: μ_θ(x_t, t)
            coef1 = 1.0 / alpha.sqrt()
            coef2 = beta / (1.0 - alpha_bar).sqrt()
            mean  = coef1 * (x - coef2 * eps_pred)

            if step > 0:
                sigma = beta.sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x.clamp(-5.0, 5.0)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
