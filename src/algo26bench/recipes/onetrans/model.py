from __future__ import annotations

import torch
import torch.nn as nn

from algo26bench.core.types import DataContext, ModelOutput
from algo26bench.nn import (
    AutoSplitTokenizer,
    EmbeddingBank,
    MixedCausalAttention,
    MixedFFN,
    RMSNorm,
    SequenceTokenizer,
)
from .batch import OneTransBatch
from .config import OneTransConfig


class OneTransLayer(nn.Module):
    def __init__(self, config: OneTransConfig) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(config.d_model)
        self.attention = MixedCausalAttention(
            config.d_model,
            config.num_heads,
            config.num_ns_tokens,
            config.dropout,
        )
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = MixedFFN(
            config.d_model,
            config.num_ns_tokens,
            config.hidden_multiplier,
            config.dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        token_kinds: torch.Tensor,
        valid: torch.Tensor,
        keep_sequence_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_indices = torch.nonzero(token_kinds < 0, as_tuple=False).flatten()
        ns_indices = torch.nonzero(token_kinds >= 0, as_tuple=False).flatten()
        keep = min(keep_sequence_tokens, sequence_indices.numel())
        kept_sequence = (
            sequence_indices[-keep:]
            if keep
            else sequence_indices.new_empty((0,), dtype=torch.long)
        )
        query_indices = torch.cat([kept_sequence, ns_indices])
        selected = inputs.index_select(1, query_indices)
        selected_valid = valid.index_select(1, query_indices)
        selected_kinds = token_kinds.index_select(0, query_indices)
        selected = selected + self.attention(
            self.norm_attention(inputs), token_kinds, valid, query_indices
        )
        selected = selected * selected_valid.unsqueeze(-1).to(selected.dtype)
        selected = selected + self.ffn(
            self.norm_ffn(selected), selected_kinds, selected_valid
        )
        selected = selected * selected_valid.unsqueeze(-1).to(selected.dtype)
        return selected, selected_kinds, selected_valid


class OneTransModel(nn.Module):
    def __init__(self, context: DataContext, config: OneTransConfig) -> None:
        super().__init__()
        self.context = context
        self.config = config
        spec = context.data_spec
        self.sequence_names = spec.sequence_names
        embedding_specs: dict[str, tuple[int, int]] = {}
        for field in spec.scalar_fields + spec.candidate_fields:
            embedding_specs[field.name] = (field.vocab_size, field.width)
        for domain in spec.sequence_domains:
            for field in domain.fields:
                value = (field.vocab_size, field.width)
                if field.name in embedding_specs and embedding_specs[field.name] != value:
                    raise ValueError(f"inconsistent repeated field spec: {field.name}")
                embedding_specs[field.name] = value
        self.embedding_bank = EmbeddingBank(embedding_specs, config.embedding_dim)
        non_sequence_fields = tuple(
            field.name for field in spec.scalar_fields + spec.candidate_fields
        )
        self.ns_tokenizer = AutoSplitTokenizer(
            self.embedding_bank,
            non_sequence_fields,
            spec.dense_dim,
            config.num_ns_tokens,
            config.d_model,
        )
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain.name: SequenceTokenizer(
                    self.embedding_bank,
                    tuple(field.name for field in domain.fields),
                    config.d_model,
                )
                for domain in spec.sequence_domains
            }
        )
        self.domain_embeddings = nn.Embedding(
            len(self.sequence_names), config.d_model
        )
        self.separator = nn.Parameter(torch.zeros(config.d_model))
        nn.init.normal_(self.separator, std=0.02)
        self.layers = nn.ModuleList(
            OneTransLayer(config) for _ in range(config.num_layers)
        )
        self.pyramid_lengths = config.resolved_pyramid_lengths()
        readout_dim = config.num_ns_tokens * config.d_model
        self.readout = nn.Sequential(
            RMSNorm(readout_dim),
            nn.Linear(readout_dim, config.d_model),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {task.name: nn.Linear(config.d_model, 1) for task in context.task_set.tasks}
        )

    def _sequence_tokens(self, batch: OneTransBatch) -> torch.Tensor:
        encoded = {
            name: self.sequence_tokenizers[name](
                batch.sequences[name].fields, batch.sequences[name].valid
            )
            for name in self.sequence_names
        }
        merged = batch.dense.new_zeros(
            (
                batch.batch_size,
                self.config.max_sequence_tokens,
                self.config.d_model,
            )
        )
        for source, name in enumerate(self.sequence_names):
            selected = batch.merge_source == source
            rows, targets = torch.nonzero(selected, as_tuple=True)
            if rows.numel():
                positions = batch.merge_position[rows, targets]
                merged[rows, targets] = (
                    encoded[name][rows, positions]
                    + self.domain_embeddings.weight[source]
                )
        separator_positions = batch.merge_source == -2
        if bool(torch.any(separator_positions)):
            merged[separator_positions] = self.separator
        return merged * batch.sequence_valid.unsqueeze(-1).to(merged.dtype)

    def forward(self, batch: OneTransBatch) -> ModelOutput:
        non_sequence_values = dict(batch.scalars)
        non_sequence_values.update(batch.candidates)
        ns_tokens = self.ns_tokenizer(non_sequence_values, batch.dense)
        sequence_tokens = self._sequence_tokens(batch)
        inputs = torch.cat([sequence_tokens, ns_tokens], dim=1)
        valid = torch.cat(
            [
                batch.sequence_valid,
                torch.ones(
                    (batch.batch_size, self.config.num_ns_tokens),
                    dtype=torch.bool,
                    device=batch.sequence_valid.device,
                ),
            ],
            dim=1,
        )
        token_kinds = torch.cat(
            [
                torch.full(
                    (self.config.max_sequence_tokens,),
                    -1,
                    dtype=torch.long,
                    device=inputs.device,
                ),
                torch.arange(self.config.num_ns_tokens, device=inputs.device),
            ]
        )

        observed_lengths = []
        for layer, keep in zip(self.layers, self.pyramid_lengths):
            inputs, token_kinds, valid = layer(inputs, token_kinds, valid, keep)
            observed_lengths.append(int(torch.sum(token_kinds < 0)))

        ns_indices = torch.nonzero(token_kinds >= 0, as_tuple=False).flatten()
        final_ns = inputs.index_select(1, ns_indices)
        representation = self.readout(final_ns.flatten(start_dim=1))
        return ModelOutput(
            logits={
                task.name: self.heads[task.name](representation).squeeze(-1)
                for task in self.context.task_set.tasks
            },
            representation=representation,
            diagnostics={
                "pyramid_sequence_lengths": tuple(observed_lengths),
                "merged_valid_lengths": batch.sequence_valid.sum(dim=1),
            },
        )
