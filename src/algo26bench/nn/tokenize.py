from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

class EmbeddingBank(nn.Module):
    def __init__(
        self, fields: Mapping[str, tuple[int, int]], embedding_dim: int
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.specs = dict(fields)
        self._keys = {name: f"f{index}" for index, name in enumerate(self.specs)}
        self.tables = nn.ModuleDict(
            {
                self._keys[name]: nn.Embedding(
                    vocab_size, embedding_dim, padding_idx=0
                )
                for name, (vocab_size, _) in self.specs.items()
            }
        )

    def embed_flat(self, name: str, values: torch.Tensor) -> torch.Tensor:
        _, width = self.specs[name]
        embedded = self.tables[self._keys[name]](values.long())
        if width == 1:
            return embedded
        if values.shape[-1] != width:
            raise ValueError(f"{name}: expected width={width}")
        return embedded.flatten(start_dim=-2)


def _field_width(
    field_names: Sequence[str],
    specs: Mapping[str, tuple[int, int]],
    embedding_dim: int,
) -> int:
    return sum(specs[name][1] * embedding_dim for name in field_names)


class SemanticGroupTokenizer(nn.Module):
    def __init__(
        self,
        bank: EmbeddingBank,
        groups: Sequence[Sequence[str]],
        d_model: int,
    ) -> None:
        super().__init__()
        self.bank = bank
        self.groups = tuple(tuple(group) for group in groups)
        flattened = [name for group in self.groups for name in group]
        if not self.groups or len(flattened) != len(set(flattened)):
            raise ValueError("semantic groups must be non-empty and disjoint")
        unknown = set(flattened) - set(bank.specs)
        if unknown:
            raise ValueError(f"unknown grouped fields: {sorted(unknown)}")
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(
                    _field_width(group, bank.specs, bank.embedding_dim), d_model
                ),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for group in self.groups
        )

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for group, projection in zip(self.groups, self.projections):
            vector = torch.cat(
                [self.bank.embed_flat(name, values[name]) for name in group], dim=-1
            )
            tokens.append(projection(vector))
        return torch.stack(tokens, dim=1)


class DenseTokenizer(nn.Module):
    def __init__(self, dense_dim: int, d_model: int) -> None:
        super().__init__()
        if dense_dim <= 0:
            raise ValueError("dense_dim must be positive")
        self.projection = nn.Sequential(
            nn.Linear(dense_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        return self.projection(dense).unsqueeze(1)


class AutoSplitTokenizer(nn.Module):
    def __init__(
        self,
        bank: EmbeddingBank,
        field_names: Sequence[str],
        dense_dim: int,
        num_tokens: int,
        d_model: int,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        self.bank = bank
        self.field_names = tuple(field_names)
        self.dense_dim = dense_dim
        self.num_tokens = num_tokens
        input_dim = _field_width(
            self.field_names, bank.specs, bank.embedding_dim
        ) + dense_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model * num_tokens),
            nn.SiLU(),
            nn.Linear(d_model * num_tokens, d_model * num_tokens),
        )
        self.d_model = d_model

    def forward(
        self, values: Mapping[str, torch.Tensor], dense: torch.Tensor
    ) -> torch.Tensor:
        vectors = [self.bank.embed_flat(name, values[name]) for name in self.field_names]
        if self.dense_dim:
            vectors.append(dense)
        merged = torch.cat(vectors, dim=-1)
        return self.projection(merged).view(-1, self.num_tokens, self.d_model)


class SequenceTokenizer(nn.Module):
    def __init__(
        self,
        bank: EmbeddingBank,
        field_names: Sequence[str],
        d_model: int,
    ) -> None:
        super().__init__()
        self.bank = bank
        self.field_names = tuple(field_names)
        input_dim = _field_width(
            self.field_names, bank.specs, bank.embedding_dim
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(
        self, fields: Mapping[str, torch.Tensor], valid: torch.Tensor
    ) -> torch.Tensor:
        vector = torch.cat(
            [self.bank.embed_flat(name, fields[name]) for name in self.field_names],
            dim=-1,
        )
        return self.projection(vector) * valid.unsqueeze(-1).to(vector.dtype)


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
