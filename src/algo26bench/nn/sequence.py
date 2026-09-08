from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MaskSafeMultiheadAttention
from .mixing import SwiGLU


def recent_valid_index(
    valid: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the most recent ``num_tokens`` valid slots, in chronological order.

    Reads the mask instead of assuming which side the padding sits on, so a caller
    that switches between left and right padding cannot silently select the wrong
    behaviours.  Returns ``(index, index_valid)``, both ``[B, num_tokens]``; rows
    with fewer than ``num_tokens`` valid slots are marked invalid in the tail.
    """

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    batch, length = valid.shape
    if num_tokens > length:
        raise ValueError(
            f"cannot select {num_tokens} recent slots from a length-{length} sequence"
        )
    position = torch.arange(length, device=valid.device).expand(batch, -1)
    rank_from_end = valid.flip(1).cumsum(dim=1).flip(1)
    is_recent = valid & (rank_from_end <= num_tokens)
    order = torch.where(is_recent, position, position + length).argsort(dim=1)
    index = order[:, :num_tokens]
    return index, is_recent.gather(1, index)


class TransformerSequenceEncoder(nn.Module):
    """HyFormer Eq. 5: full self-attention encoding, output length equals input."""

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

    def forward(
        self, inputs: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = valid.unsqueeze(-1).to(inputs.dtype)
        normalized = self.norm_attention(inputs)
        inputs = inputs + self.attention(
            normalized, normalized, valid, query_valid=valid
        )
        inputs = inputs * mask
        inputs = inputs + self.ffn(self.norm_ffn(inputs))
        return inputs * mask, valid


class LongerSequenceEncoder(nn.Module):
    """HyFormer Eq. 6: ``H_l = CrossAttn(S_short, S, S)`` with ``L_H << L_S``.

    ``S_short`` is the most recent ``num_short_tokens`` behaviours of ``S``.  That
    choice is not free: LONGER (arXiv:2505.04421, Table 2) compares recent-k against
    uniform-k and learnable-k at k=100 and measures +1.57% / +1.45% / +1.17% AUC, so
    learnable query tokens are the worst of the three.

    Output length is ``num_short_tokens`` rather than the input length, which is the
    whole point (``O(L_S^2)`` becomes ``O(L_H * L_S)``).  Callers must therefore use
    the returned mask, not the one they passed in.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_short_tokens: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_short_tokens <= 0:
            raise ValueError("num_short_tokens must be positive")
        self.num_short_tokens = num_short_tokens
        self.norm_sequence = nn.LayerNorm(d_model)
        self.attention = MaskSafeMultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_multiplier, dropout)

    def forward(
        self, inputs: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index, short_valid = recent_valid_index(valid, self.num_short_tokens)
        gather_index = index.unsqueeze(-1).expand(-1, -1, inputs.shape[-1])
        mask = short_valid.unsqueeze(-1).to(inputs.dtype)

        normalized = self.norm_sequence(inputs)
        hidden = inputs.gather(1, gather_index) * mask
        hidden = hidden + self.attention(
            normalized.gather(1, gather_index),
            normalized,
            valid,
            query_valid=short_valid,
        )
        hidden = hidden * mask
        hidden = hidden + self.ffn(self.norm_ffn(hidden))
        return hidden * mask, short_valid


class PointwiseSequenceEncoder(nn.Module):
    """HyFormer Eq. 7: attention-free decoder-style SwiGLU encoding."""

    def __init__(
        self, d_model: int, hidden_multiplier: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_multiplier, dropout)

    def forward(
        self, inputs: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.ffn(self.norm(inputs))
        return output * valid.unsqueeze(-1).to(output.dtype), valid
