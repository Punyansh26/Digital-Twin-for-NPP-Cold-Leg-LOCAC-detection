"""
Transolver++ Neural Operator for AP1000 Digital Twin.

Architecture:
  1. Mesh embedding  — project mesh coordinates into a token space.
  2. Token generator — soft-assignment of mesh nodes to M learned tokens.
  3. Transformer     — multi-layer self-attention over tokens.
  4. Scatter decoder — reconstruct predictions on the full mesh.

This operator is selectable via ``operator: transolver`` in model_config.yaml.

References:
    Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General
    Geometries", ICML 2024.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component: Mesh Embedding
# ---------------------------------------------------------------------------

class MeshEmbedding(nn.Module):
    """
    Soft-assign mesh nodes to M learned token positions.

    Returns (tokens, assignment_weights):
        tokens           [M, embed_dim]  — pooled token representations.
        assignment_weights [N, M]        — used to scatter tokens back to mesh.
    """

    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_tokens: int,
    ) -> None:
        super().__init__()
        self.n_tokens  = n_tokens
        self.coord_proj = nn.Linear(coord_dim, embed_dim)
        # Learnable query for each token slot
        self.token_query = nn.Linear(embed_dim, n_tokens, bias=True)

    def forward(
        self,
        coords: torch.Tensor,
        extra_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords:         [N, coord_dim]
            extra_features: [N, extra_dim] optional per-node features.

        Returns:
            tokens:     [M, embed_dim]
            attn_w:     [N, M]  soft assignment weights
        """
        node_feat = self.coord_proj(coords)           # [N, embed_dim]
        if extra_features is not None:
            node_feat = node_feat + extra_features

        attn_logits = self.token_query(node_feat)     # [N, M]
        attn_w      = F.softmax(attn_logits, dim=-1)  # [N, M]
        tokens      = attn_w.T @ node_feat            # [M, embed_dim]
        return tokens, attn_w


# ---------------------------------------------------------------------------
# Component: Multi-head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.qkv      = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, D]  →  [T, D]"""
        T, D = x.shape
        qkv = self.qkv(x)                                      # [T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                        # [T, D] each

        # Reshape to [H, T, head_dim]
        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(T, self.n_heads, self.head_dim).permute(1, 0, 2)

        q, k, v = _split(q), _split(k), _split(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale         # [H, T, T]
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).permute(1, 0, 2).reshape(T, D)       # [T, D]
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Component: Transformer Token Block
# ---------------------------------------------------------------------------

class TransformerTokenBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden     = embed_dim * mlp_ratio
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main module: Transolver++
# ---------------------------------------------------------------------------

class TransolverOperator(nn.Module):
    """
    Transolver++ neural operator for AP1000 flow field prediction.

    Args:
        coord_dim:  Spatial coordinate dimension (default 3).
        branch_dim: Branch (parameter) input dimension (default 3).
        embed_dim:  Token embedding dimension (default 256).
        n_tokens:   Number of mesh tokens M (default 64).
        n_layers:   Number of transformer blocks (default 4).
        n_heads:    Attention heads per block (default 8).
        n_outputs:  Number of output flow fields (default 4).
        dropout:    Dropout probability (default 0.1).
    """

    def __init__(
        self,
        coord_dim: int = 3,
        branch_dim: int = 3,
        embed_dim: int = 256,
        n_tokens: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        n_outputs: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.n_tokens   = n_tokens
        self.n_outputs  = n_outputs

        # Branch encoder: physics parameters → embed_dim
        self.branch_encoder = nn.Sequential(
            nn.Linear(branch_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Mesh tokenizer
        self.mesh_embed = MeshEmbedding(coord_dim, embed_dim, n_tokens)

        # Transformer over tokens
        self.transformer = nn.ModuleList([
            TransformerTokenBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Per-field projection from node features to scalar
        self.output_proj = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(n_outputs)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_outputs)
        ])

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "TransolverOperator":
        cfg = config.get("transolver", {})
        return cls(
            coord_dim  = cfg.get("coord_dim",  3),
            branch_dim = cfg.get("branch_dim", 3),
            embed_dim  = cfg.get("embed_dim",  256),
            n_tokens   = cfg.get("n_tokens",   64),
            n_layers   = cfg.get("n_layers",   4),
            n_heads    = cfg.get("n_heads",    8),
            n_outputs  = cfg.get("n_outputs",  4),
            dropout    = cfg.get("dropout",    0.1),
        )

    # ------------------------------------------------------------------
    def _forward_single(
        self,
        branch_feat: torch.Tensor,   # [embed_dim]
        mesh_tokens: torch.Tensor,   # [M, embed_dim]
        attn_w:      torch.Tensor,   # [N, M]
    ) -> torch.Tensor:               # [n_outputs, N]
        """Single-sample forward (unbatched transformer)."""
        # Condition tokens on branch feature
        tokens = mesh_tokens + branch_feat.unsqueeze(0)  # [M, embed_dim]

        for block in self.transformer:
            tokens = block(tokens)
        tokens = self.norm(tokens)  # [M, embed_dim]

        # Scatter back to mesh: [N, embed_dim]
        node_feat = attn_w @ tokens  # [N, embed_dim]

        fields = []
        for i in range(self.n_outputs):
            f = self.output_proj[i](node_feat).squeeze(-1) + self.biases[i]
            fields.append(f)
        return torch.stack(fields, dim=0)  # [n_outputs, N]

    # ------------------------------------------------------------------
    def forward(
        self,
        branch_input: torch.Tensor,
        mesh_points:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            branch_input: [B, branch_dim]
            mesh_points:  [N, coord_dim]

        Returns:
            [B, n_outputs, N]
        """
        branch_feat         = self.branch_encoder(branch_input)   # [B, embed_dim]
        mesh_tokens, attn_w = self.mesh_embed(mesh_points)        # [M, D], [N, M]

        outputs = []
        for b in range(branch_input.shape[0]):
            out = self._forward_single(branch_feat[b], mesh_tokens, attn_w)
            outputs.append(out)

        return torch.stack(outputs, dim=0)   # [B, n_outputs, N]

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
