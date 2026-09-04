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


@dataclass
class HyFormerBatch:
    scalars: Mapping[str, torch.Tensor]
    dense: torch.Tensor
    candidates: Mapping[str, torch.Tensor]
    sequences: Mapping[str, PaddedSequence]
    labels: Mapping[str, torch.Tensor]
    sample_ids: torch.Tensor
    group_ids: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return int(self.sample_ids.shape[0])

    def to(self, device: torch.device | str) -> "HyFormerBatch":
        return HyFormerBatch(
            scalars={name: value.to(device) for name, value in self.scalars.items()},
            dense=self.dense.to(device),
            candidates={
                name: value.to(device) for name, value in self.candidates.items()
            },
            sequences={
                name: value.to(device) for name, value in self.sequences.items()
            },
            labels={name: value.to(device) for name, value in self.labels.items()},
            sample_ids=self.sample_ids.to(device),
            group_ids=None if self.group_ids is None else self.group_ids.to(device),
        )


class HyFormerCollator:
    def __init__(self, context: DataContext) -> None:
        self.context = context

    def __call__(self, examples: Sequence[RawExample]) -> HyFormerBatch:
        raw = collate_raw_examples(
            examples, self.context.data_spec, self.context.task_set
        )
        sequences = {
            domain.name: pad_ragged_sequence(
                raw.sequences[domain.name], domain.max_len, left_pad=False
            )
            for domain in self.context.data_spec.sequence_domains
        }
        return HyFormerBatch(
            scalars=raw.scalars,
            dense=raw.dense,
            candidates=raw.candidates,
            sequences=sequences,
            labels=raw.labels,
            sample_ids=raw.sample_ids,
            group_ids=raw.group_ids,
        )
