from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from algo26bench.core.types import DataContext, RawExample
from algo26bench.data.view import (
    PaddedSequence,
    collate_raw_examples,
    pad_ragged_sequence,
)

from .config import OneTransConfig


@dataclass
class OneTransBatch:
    scalars: Mapping[str, torch.Tensor]
    dense: torch.Tensor
    candidates: Mapping[str, torch.Tensor]
    sequences: Mapping[str, PaddedSequence]
    merge_source: torch.Tensor
    merge_position: torch.Tensor
    sequence_valid: torch.Tensor
    labels: Mapping[str, torch.Tensor]
    sample_ids: torch.Tensor
    group_ids: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return int(self.sample_ids.shape[0])

    def to(self, device: torch.device | str) -> "OneTransBatch":
        return OneTransBatch(
            scalars={name: value.to(device) for name, value in self.scalars.items()},
            dense=self.dense.to(device),
            candidates={
                name: value.to(device) for name, value in self.candidates.items()
            },
            sequences={
                name: value.to(device) for name, value in self.sequences.items()
            },
            merge_source=self.merge_source.to(device),
            merge_position=self.merge_position.to(device),
            sequence_valid=self.sequence_valid.to(device),
            labels={name: value.to(device) for name, value in self.labels.items()},
            sample_ids=self.sample_ids.to(device),
            group_ids=None if self.group_ids is None else self.group_ids.to(device),
        )


class OneTransCollator:
    """Builds a fixed-width, left-padded unified S-token merge plan."""

    def __init__(self, context: DataContext, config: OneTransConfig) -> None:
        self.context = context
        self.config = config
        names = context.data_spec.sequence_names
        self.domain_order = config.domain_order or names
        if len(self.domain_order) != len(set(self.domain_order)) or set(
            self.domain_order
        ) != set(names):
            raise ValueError("domain_order must contain every sequence domain once")
        self.domain_indices = {name: index for index, name in enumerate(names)}

    def __call__(self, examples: Sequence[RawExample]) -> OneTransBatch:
        raw = collate_raw_examples(
            examples, self.context.data_spec, self.context.task_set
        )
        padded = {
            domain.name: pad_ragged_sequence(
                raw.sequences[domain.name], domain.max_len, left_pad=False
            )
            for domain in self.context.data_spec.sequence_domains
        }
        batch_size = raw.batch_size
        max_tokens = self.config.max_sequence_tokens
        merge_source = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
        merge_position = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
        sequence_valid = torch.zeros((batch_size, max_tokens), dtype=torch.bool)

        for row in range(batch_size):
            plan: list[tuple[int, int]] = []
            if self.config.merge_mode == "timestamp":
                timestamped: list[tuple[int, int, int]] = []
                for name in self.domain_order:
                    source = self.domain_indices[name]
                    valid_positions = torch.nonzero(
                        padded[name].valid[row], as_tuple=False
                    ).flatten()
                    for position_tensor in valid_positions:
                        position = int(position_tensor)
                        timestamped.append(
                            (
                                int(padded[name].timestamps[row, position]),
                                source,
                                position,
                            )
                        )
                timestamped.sort(key=lambda event: (event[0], event[1], event[2]))
                plan = [(source, position) for _, source, position in timestamped]
            else:
                for order_index, name in enumerate(self.domain_order):
                    source = self.domain_indices[name]
                    valid_positions = torch.nonzero(
                        padded[name].valid[row], as_tuple=False
                    ).flatten()
                    plan.extend((source, int(position)) for position in valid_positions)
                    if order_index + 1 < len(self.domain_order):
                        plan.append((-2, -1))

            plan = plan[-max_tokens:]
            target_start = max_tokens - len(plan)
            for offset, (source, position) in enumerate(plan):
                target = target_start + offset
                merge_source[row, target] = source
                merge_position[row, target] = position
                sequence_valid[row, target] = True

        return OneTransBatch(
            scalars=raw.scalars,
            dense=raw.dense,
            candidates=raw.candidates,
            sequences=padded,
            merge_source=merge_source,
            merge_position=merge_position,
            sequence_valid=sequence_valid,
            labels=raw.labels,
            sample_ids=raw.sample_ids,
            group_ids=raw.group_ids,
        )
