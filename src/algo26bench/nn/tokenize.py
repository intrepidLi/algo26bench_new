from __future__ import annotations

from collections.abc import Mapping, Sequence

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingBank(nn.Module):
    def __init__(
        self,
        fields: Mapping[str, tuple[int, int]],
        embedding_dim: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.specs = dict(fields)
        self.emb_skip_threshold = int(emb_skip_threshold)
        self._keys = {name: f"f{index}" for index, name in enumerate(self.specs)}
        self._skipped: set[str] = set()
        tables: dict[str, nn.Embedding] = {}
        for name, (vocab_size, _) in self.specs.items():
            if self.emb_skip_threshold > 0 and int(vocab_size) > self.emb_skip_threshold:
                # Skip the table entirely; embed_flat returns zeros for this
                # field. Matches algo26bench's --emb_skip_threshold semantics.
                self._skipped.add(name)
                continue
            table = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            # v1 algo26bench init_embedding: Xavier-normal on the weight, zero
            # the padding_idx row. Default nn.Embedding init is N(0, 1) which
            # blows sequence tokens up ~1.0 in std vs v1's ~0.04 — measurable
            # AUC gap over 1 epoch.
            nn.init.xavier_normal_(table.weight.data)
            table.weight.data[0].zero_()
            tables[self._keys[name]] = table
        self.tables = nn.ModuleDict(tables)

    def embed_flat(self, name: str, values: torch.Tensor) -> torch.Tensor:
        _, width = self.specs[name]
        if name in self._skipped:
            # Return a zero embedding shaped as if a real Embedding had been
            # called and flattened. values may be scalar-per-row (shape (B,))
            # or width>1 (shape (B, width)); mirror embed_flat's normal path.
            out_dim = self.embedding_dim * width
            batch_shape = values.shape if width == 1 else values.shape[:-1]
            return values.new_zeros(*batch_shape, out_dim, dtype=torch.float32)
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


class RankMixerNSTokenizer(nn.Module):
    """RankMixer-style NS tokenizer: concat all field embeddings, split into
    equal-size chunks, project each chunk to d_model. Semantic groups are
    ignored — fields are laid out in the given field_names order. Mirrors
    algo26bench (v1) get_embedding.py::RankMixerNSTokenizer.
    """

    def __init__(
        self,
        bank: EmbeddingBank,
        field_names: Sequence[str],
        num_ns_tokens: int,
        d_model: int,
    ) -> None:
        super().__init__()
        if num_ns_tokens <= 0:
            raise ValueError("num_ns_tokens must be positive")
        self.bank = bank
        self.field_names = tuple(field_names)
        unknown = set(self.field_names) - set(bank.specs)
        if unknown:
            raise ValueError(f"unknown rankmixer fields: {sorted(unknown)}")
        self.num_ns_tokens = num_ns_tokens
        total_dim = _field_width(self.field_names, bank.specs, bank.embedding_dim)
        self.chunk_dim = math.ceil(total_dim / num_ns_tokens)
        self.padded_total = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total - total_dim
        self.token_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        )

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(
            [self.bank.embed_flat(name, values[name]) for name in self.field_names],
            dim=-1,
        )
        if self._pad_size:
            merged = F.pad(merged, (0, self._pad_size))
        chunks = merged.split(self.chunk_dim, dim=-1)
        tokens = [projection(chunk) for chunk, projection in zip(chunks, self.token_projections)]
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
