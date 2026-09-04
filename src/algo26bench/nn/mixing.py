from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, hidden_multiplier: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_multiplier
        self.gate_up = nn.Linear(d_model, hidden_dim * 2)
        self.down = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(inputs).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_multiplier
        self.num_tokens = num_tokens
        self.weight_in = nn.Parameter(torch.empty(num_tokens, d_model, hidden_dim))
        self.bias_in = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.weight_out = nn.Parameter(torch.empty(num_tokens, hidden_dim, d_model))
        self.bias_out = nn.Parameter(torch.zeros(num_tokens, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight_in)
        nn.init.xavier_uniform_(self.weight_out)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] != self.num_tokens:
            raise ValueError(
                f"expected {self.num_tokens} tokens, got {inputs.shape[1]}"
            )
        hidden = torch.einsum("btd,tdh->bth", inputs, self.weight_in)
        hidden = F.silu(hidden + self.bias_in.unsqueeze(0))
        hidden = self.dropout(hidden)
        output = torch.einsum("bth,thd->btd", hidden, self.weight_out)
        return self.dropout(output + self.bias_out.unsqueeze(0))


class RankMixerRewire(nn.Module):
    """Parameter-free channel/token rewiring from HyFormer Eq. 11-13."""

    def __init__(self, num_tokens: int, d_model: int) -> None:
        super().__init__()
        if d_model % num_tokens:
            raise ValueError("d_model must be divisible by num_tokens")
        self.num_tokens = num_tokens
        self.sub_dimension = d_model // num_tokens

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, dimension = inputs.shape
        if tokens != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {tokens}")
        return (
            inputs.view(batch, tokens, self.num_tokens, self.sub_dimension)
            .transpose(1, 2)
            .contiguous()
            .view(batch, tokens, dimension)
        )


class QueryBoosting(nn.Module):
    """HyFormer rewire, independent per-token FFNs, and exactly one residual."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rewire = RankMixerRewire(num_tokens, d_model)
        self.ffn = PerTokenFFN(
            num_tokens, d_model, hidden_multiplier=hidden_multiplier, dropout=dropout
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.ffn(self.rewire(inputs))


class MixedFFN(nn.Module):
    """Shared S-token FFN and independent FFN parameters for each NS token."""

    def __init__(
        self,
        d_model: int,
        num_ns_tokens: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_multiplier
        self.num_ns_tokens = num_ns_tokens
        self.shared = SwiGLU(d_model, hidden_multiplier, dropout)
        self.weight_in = nn.Parameter(
            torch.empty(num_ns_tokens, d_model, hidden_dim * 2)
        )
        self.bias_in = nn.Parameter(torch.zeros(num_ns_tokens, hidden_dim * 2))
        self.weight_out = nn.Parameter(
            torch.empty(num_ns_tokens, hidden_dim, d_model)
        )
        self.bias_out = nn.Parameter(torch.zeros(num_ns_tokens, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight_in)
        nn.init.xavier_uniform_(self.weight_out)

    def forward(
        self, inputs: torch.Tensor, token_kinds: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        output = self.shared(inputs)
        for kind in range(self.num_ns_tokens):
            positions = torch.nonzero(token_kinds == kind, as_tuple=False).flatten()
            if positions.numel() != 1:
                raise ValueError(f"NS token {kind} must occur exactly once")
            position = int(positions[0])
            token = inputs[:, position]
            gate_value = torch.einsum(
                "bd,dh->bh", token, self.weight_in[kind]
            ) + self.bias_in[kind]
            gate, value = gate_value.chunk(2, dim=-1)
            hidden = self.dropout(F.silu(gate) * value)
            specific = torch.einsum(
                "bh,hd->bd", hidden, self.weight_out[kind]
            ) + self.bias_out[kind]
            output[:, position] = self.dropout(specific)
        return output * valid.unsqueeze(-1).to(output.dtype)
