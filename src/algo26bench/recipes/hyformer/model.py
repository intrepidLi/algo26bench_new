from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from algo26bench.core.types import DataContext, ModelOutput
from algo26bench.nn import (
    DenseTokenizer,
    EmbeddingBank,
    LongerSequenceEncoder,
    MaskSafeMultiheadAttention,
    PointwiseSequenceEncoder,
    QueryBoosting,
    SemanticGroupTokenizer,
    SequenceTokenizer,
    TransformerSequenceEncoder,
    masked_mean,
)
from .batch import HyFormerBatch
from .config import HyFormerConfig


def _build_sequence_encoder(config: HyFormerConfig, max_len: int) -> nn.Module:
    """One of the three encoding strategies of HyFormer Sec. 3.4.1."""

    if config.sequence_encoder == "transformer":
        return TransformerSequenceEncoder(
            config.d_model,
            config.num_heads,
            config.hidden_multiplier,
            config.dropout,
        )
    if config.sequence_encoder == "longer":
        # L_H << L_S is the premise of Eq. 6; a sequence domain shorter than the
        # configured L_H simply cannot be compressed further than its own length.
        return LongerSequenceEncoder(
            config.d_model,
            config.num_heads,
            min(config.num_short_tokens, max_len),
            config.hidden_multiplier,
            config.dropout,
        )
    return PointwiseSequenceEncoder(
        config.d_model,
        config.hidden_multiplier,
        config.dropout,
    )


class MultiSequenceQueryGenerator(nn.Module):
    def __init__(
        self,
        sequence_names: Sequence[str],
        num_ns_tokens: int,
        num_queries: int,
        d_model: int,
        hidden_multiplier: int,
    ) -> None:
        super().__init__()
        self.sequence_names = tuple(sequence_names)
        self.num_queries = num_queries
        input_dim = (num_ns_tokens + 1) * d_model
        hidden_dim = d_model * hidden_multiplier
        self.networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, num_queries * d_model),
                )
                for name in self.sequence_names
            }
        )
        self.d_model = d_model

    def forward(
        self,
        ns_tokens: torch.Tensor,
        sequence_tokens: Mapping[str, torch.Tensor],
        valid_masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        static = ns_tokens.flatten(start_dim=1)
        return {
            name: self.networks[name](
                torch.cat(
                    [static, masked_mean(sequence_tokens[name], valid_masks[name])],
                    dim=-1,
                )
            ).view(-1, self.num_queries, self.d_model)
            for name in self.sequence_names
        }


class HyFormerLayer(nn.Module):
    def __init__(
        self,
        sequence_names: Sequence[str],
        num_queries: int,
        num_ns_tokens: int,
        config: HyFormerConfig,
        sequence_max_lens: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.sequence_names = tuple(sequence_names)
        self.num_queries = num_queries
        self.sequence_encoders = nn.ModuleDict(
            {
                name: _build_sequence_encoder(config, sequence_max_lens[name])
                for name in self.sequence_names
            }
        )
        self.query_norms = nn.ModuleDict(
            {name: nn.LayerNorm(config.d_model) for name in self.sequence_names}
        )
        self.sequence_norms = nn.ModuleDict(
            {name: nn.LayerNorm(config.d_model) for name in self.sequence_names}
        )
        self.query_decoders = nn.ModuleDict(
            {
                name: MaskSafeMultiheadAttention(
                    config.d_model, config.num_heads, config.dropout
                )
                for name in self.sequence_names
            }
        )
        total_tokens = len(self.sequence_names) * num_queries + num_ns_tokens
        self.query_boosting = QueryBoosting(
            total_tokens,
            config.d_model,
            hidden_multiplier=config.hidden_multiplier,
            dropout=config.dropout,
        )

    def forward(
        self,
        queries: dict[str, torch.Tensor],
        ns_tokens: torch.Tensor,
        sequences: dict[str, torch.Tensor],
        valid_masks: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        kv_lengths: dict[str, torch.Tensor] = {}
        decoded: dict[str, torch.Tensor] = {}
        for name in self.sequence_names:
            # Eq. 5-7 may shorten the sequence, so the key/value mask comes back from
            # the encoder rather than being reused from its input.
            encoded, encoded_valid = self.sequence_encoders[name](
                sequences[name], valid_masks[name]
            )
            kv_lengths[name] = encoded_valid.sum(dim=1)
            query = queries[name]
            delta = self.query_decoders[name](
                self.query_norms[name](query),
                self.sequence_norms[name](encoded),
                encoded_valid,
            )
            decoded[name] = query + delta

        combined = torch.cat(
            [decoded[name] for name in self.sequence_names] + [ns_tokens], dim=1
        )
        boosted = self.query_boosting(combined)
        next_queries: dict[str, torch.Tensor] = {}
        offset = 0
        for name in self.sequence_names:
            next_queries[name] = boosted[:, offset : offset + self.num_queries]
            offset += self.num_queries
        return next_queries, boosted[:, offset:], kv_lengths


class HyFormerModel(nn.Module):
    def __init__(
        self,
        context: DataContext,
        config: HyFormerConfig,
        semantic_groups: Sequence[Sequence[str]],
    ) -> None:
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
        self.ns_tokenizer = SemanticGroupTokenizer(
            self.embedding_bank, semantic_groups, config.d_model
        )
        self.dense_tokenizer = (
            DenseTokenizer(spec.dense_dim, config.d_model)
            if spec.dense_dim
            else None
        )
        self.num_ns_tokens = len(semantic_groups) + int(self.dense_tokenizer is not None)
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
        self.query_generator = MultiSequenceQueryGenerator(
            self.sequence_names,
            self.num_ns_tokens,
            config.num_queries,
            config.d_model,
            config.hidden_multiplier,
        )
        total_tokens = (
            len(self.sequence_names) * config.num_queries + self.num_ns_tokens
        )
        if config.d_model % total_tokens:
            raise ValueError(
                "HyFormer RankMixer rewiring requires d_model divisible by "
                f"query+NS token count ({total_tokens})"
            )
        sequence_max_lens = {
            domain.name: domain.max_len for domain in spec.sequence_domains
        }
        self.layers = nn.ModuleList(
            HyFormerLayer(
                self.sequence_names,
                config.num_queries,
                self.num_ns_tokens,
                config,
                sequence_max_lens,
            )
            for _ in range(config.num_layers)
        )
        readout_dim = total_tokens * config.d_model
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, config.d_model),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {task.name: nn.Linear(config.d_model, 1) for task in context.task_set.tasks}
        )

    def forward(self, batch: HyFormerBatch) -> ModelOutput:
        non_sequence_values = dict(batch.scalars)
        non_sequence_values.update(batch.candidates)
        ns_tokens = self.ns_tokenizer(non_sequence_values)
        if self.dense_tokenizer is not None:
            ns_tokens = torch.cat(
                [ns_tokens, self.dense_tokenizer(batch.dense)], dim=1
            )

        sequences = {
            name: self.sequence_tokenizers[name](
                batch.sequences[name].fields, batch.sequences[name].valid
            )
            for name in self.sequence_names
        }
        valid_masks = {
            name: batch.sequences[name].valid for name in self.sequence_names
        }
        queries = self.query_generator(ns_tokens, sequences, valid_masks)
        # Eq. 5-7 all read S, not the previous layer's output: every layer re-encodes
        # the raw tokenized sequence with its own parameters, which is what makes the
        # key/value states of Eq. 8 "recomputed at each layer".
        kv_lengths: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            queries, ns_tokens, kv_lengths = layer(
                queries, ns_tokens, sequences, valid_masks
            )

        final_tokens = torch.cat(
            [queries[name] for name in self.sequence_names] + [ns_tokens], dim=1
        )
        representation = self.readout(final_tokens.flatten(start_dim=1))
        return ModelOutput(
            logits={
                task.name: self.heads[task.name](representation).squeeze(-1)
                for task in self.context.task_set.tasks
            },
            representation=representation,
            diagnostics={
                "boost_token_count": final_tokens.shape[1],
                "sequence_lengths": {
                    name: valid_masks[name].sum(dim=1) for name in self.sequence_names
                },
                "kv_lengths": kv_lengths,
            },
        )
