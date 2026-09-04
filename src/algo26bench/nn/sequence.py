from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MaskSafeMultiheadAttention
from .mixing import SwiGLU


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = MaskSafeMultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_multiplier, dropout)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(-1).to(inputs.dtype)
        normalized = self.norm_attention(inputs)
        inputs = inputs + self.attention(
            normalized, normalized, valid, query_valid=valid
        )
        inputs = inputs * mask
        inputs = inputs + self.ffn(self.norm_ffn(inputs))
        return inputs * mask


class PointwiseSequenceEncoder(nn.Module):
    """HyFormer's attention-free decoder-style SwiGLU encoder."""

    def __init__(
        self, d_model: int, hidden_multiplier: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_multiplier, dropout)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        output = self.ffn(self.norm(inputs))
        return output * valid.unsqueeze(-1).to(output.dtype)
