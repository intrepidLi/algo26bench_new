from __future__ import annotations

import math

import torch
import torch.nn as nn


def _safe_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    dropout: nn.Module,
) -> torch.Tensor:
    """Attention that returns exact zeros for rows with no legal key."""

    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    expanded = allowed.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
    masked = logits.masked_fill(~expanded, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    weights = torch.exp(masked - row_max) * expanded.to(logits.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.matmul(dropout(weights), value)


def _allowed_matrix(
    key_valid: torch.Tensor,
    query_length: int,
    *,
    query_valid: torch.Tensor | None,
    causal: bool,
    query_positions: torch.Tensor | None,
    key_positions: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, key_length = key_valid.shape
    allowed = key_valid[:, None, :].expand(batch_size, query_length, key_length)
    if causal:
        if query_positions is None:
            query_positions = torch.arange(
                query_length, device=key_valid.device
            ).expand(batch_size, -1)
        elif query_positions.ndim == 1:
            query_positions = query_positions[None, :].expand(batch_size, -1)
        if key_positions is None:
            key_positions = torch.arange(
                key_length, device=key_valid.device
            ).expand(batch_size, -1)
        elif key_positions.ndim == 1:
            key_positions = key_positions[None, :].expand(batch_size, -1)
        allowed = allowed & (
            key_positions[:, None, :] <= query_positions[:, :, None]
        )
    if query_valid is not None:
        allowed = allowed & query_valid[:, :, None]
    return allowed


class MaskSafeMultiheadAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.0, *, bias: bool = True
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, inputs: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch, length, _ = inputs.shape
        return (
            projection(inputs)
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_valid: torch.Tensor,
        *,
        query_valid: torch.Tensor | None = None,
        causal: bool = False,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        allowed = _allowed_matrix(
            key_valid,
            query.shape[1],
            query_valid=query_valid,
            causal=causal,
            query_positions=query_positions,
            key_positions=key_positions,
        )
        attended = _safe_attention(
            self._heads(query, self.q_proj),
            self._heads(key_value, self.k_proj),
            self._heads(key_value, self.v_proj),
            allowed,
            self.dropout,
        )
        batch, _, query_length, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(
            batch, query_length, self.d_model
        )
        output = self.out_proj(merged)
        return output * allowed.any(dim=-1, keepdim=True).to(output.dtype)


class MixedCausalAttention(nn.Module):
    """Shared S-token QKV and token-specific NS-token QKV."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_ns_tokens = num_ns_tokens
        self.q_s = nn.Linear(d_model, d_model, bias=False)
        self.k_s = nn.Linear(d_model, d_model, bias=False)
        self.v_s = nn.Linear(d_model, d_model, bias=False)
        self.q_ns = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_ns_tokens)
        )
        self.k_ns = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_ns_tokens)
        )
        self.v_ns = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_ns_tokens)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _mixed_projection(
        self,
        inputs: torch.Tensor,
        token_kinds: torch.Tensor,
        shared: nn.Linear,
        specific: nn.ModuleList,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for position, kind_tensor in enumerate(token_kinds):
            kind = int(kind_tensor)
            projection = shared if kind < 0 else specific[kind]
            pieces.append(projection(inputs[:, position]))
        return torch.stack(pieces, dim=1)

    def _heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        return inputs.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        token_kinds: torch.Tensor,
        valid: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        if token_kinds.ndim != 1 or token_kinds.shape[0] != inputs.shape[1]:
            raise ValueError("token_kinds must describe every token position")
        if bool(torch.any(token_kinds >= self.num_ns_tokens)):
            raise ValueError("invalid NS token kind")
        query_inputs = inputs.index_select(1, query_indices)
        query_kinds = token_kinds.index_select(0, query_indices)

        query = self._mixed_projection(query_inputs, query_kinds, self.q_s, self.q_ns)
        key = self._mixed_projection(inputs, token_kinds, self.k_s, self.k_ns)
        value = self._mixed_projection(inputs, token_kinds, self.v_s, self.v_ns)
        query_valid = valid.index_select(1, query_indices)
        allowed = _allowed_matrix(
            valid,
            query_indices.numel(),
            query_valid=query_valid,
            causal=True,
            query_positions=query_indices,
            key_positions=None,
        )
        attended = _safe_attention(
            self._heads(query),
            self._heads(key),
            self._heads(value),
            allowed,
            self.dropout,
        )
        batch, _, query_length, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(
            batch, query_length, self.d_model
        )
        output = self.out_proj(merged)
        return output * allowed.any(dim=-1, keepdim=True).to(output.dtype)
