"""Primitives that collators compose into recipe-specific batches.

`RawExample` is the only shape the data layer emits.  This module gives
collators two categories of helpers:

* stackers (``stack_scalars`` etc.) that just concatenate per-example tensors,
* domain-aware padders (``pad_domain``) that produce a ``PaddedSequence`` for
  one sequence domain.

No intermediate ``RawBatch`` type exists: recipes read what they need directly
from ``Sequence[RawExample]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from algo26bench.core.types import (
    DataSpec,
    RawExample,
    SequenceDomainSpec,
    TaskSet,
)


@dataclass
class PaddedSequence:
    fields: Mapping[str, torch.Tensor]
    timestamps: torch.Tensor
    valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.valid.shape[0])

    @property
    def length(self) -> int:
        return int(self.valid.shape[1])

    def to(self, device: torch.device | str) -> "PaddedSequence":
        return PaddedSequence(
            fields={name: value.to(device) for name, value in self.fields.items()},
            timestamps=self.timestamps.to(device),
            valid=self.valid.to(device),
        )


def validate_examples(
    examples: Sequence[RawExample],
    data_spec: DataSpec,
    task_set: TaskSet,
) -> None:
    """Check every example matches the declared spec.  Collators call this once."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    scalar_names = {field.name for field in data_spec.scalar_fields}
    candidate_names = {field.name for field in data_spec.candidate_fields}
    sequence_names = set(data_spec.sequence_names)
    label_keys = {task.label_key for task in task_set.tasks}

    for example in examples:
        if set(example.scalars) != scalar_names:
            raise ValueError("scalar fields do not match DataSpec")
        if set(example.candidates) != candidate_names:
            raise ValueError("candidate fields do not match DataSpec")
        if set(example.sequences) != sequence_names:
            raise ValueError("sequence domains do not match DataSpec")
        if set(example.labels) != label_keys:
            raise ValueError("labels do not match TaskSet")
        if example.dense.shape != (data_spec.dense_dim,):
            raise ValueError("dense feature width does not match DataSpec")
        for domain in data_spec.sequence_domains:
            example.sequences[domain.name].validate(domain)

    has_group = [example.group_id is not None for example in examples]
    if any(has_group) and not all(has_group):
        raise ValueError("group_id must be present for every example or none")


def stack_scalars(
    examples: Sequence[RawExample], field_names: Sequence[str]
) -> Mapping[str, torch.Tensor]:
    return {
        name: torch.stack([example.scalars[name] for example in examples])
        for name in field_names
    }


def stack_candidates(
    examples: Sequence[RawExample], field_names: Sequence[str]
) -> Mapping[str, torch.Tensor]:
    return {
        name: torch.stack([example.candidates[name] for example in examples])
        for name in field_names
    }


def stack_dense(examples: Sequence[RawExample]) -> torch.Tensor:
    return torch.stack([example.dense for example in examples])


def stack_labels(
    examples: Sequence[RawExample], task_set: TaskSet
) -> Mapping[str, torch.Tensor]:
    return {
        task.label_key: torch.stack(
            [example.labels[task.label_key] for example in examples]
        ).float()
        for task in task_set.tasks
    }


def stack_sample_ids(examples: Sequence[RawExample]) -> torch.Tensor:
    return torch.stack([example.sample_id for example in examples]).long()


def stack_group_ids(examples: Sequence[RawExample]) -> torch.Tensor | None:
    if examples[0].group_id is None:
        return None
    return torch.stack([example.group_id for example in examples]).long()  # type: ignore[arg-type]


def pad_domain(
    examples: Sequence[RawExample],
    spec: SequenceDomainSpec,
    *,
    max_len: int | None = None,
    left_pad: bool = False,
    keep_recent: bool = True,
) -> PaddedSequence:
    """Pad one sequence domain across a batch.

    ``max_len`` defaults to ``spec.max_len``; recipes override when they truncate
    more aggressively than the spec.
    """

    width = spec.max_len if max_len is None else max_len
    if width <= 0:
        raise ValueError("max_len must be positive")

    batch_size = len(examples)
    first_sequence = examples[0].sequences[spec.name]
    fields: dict[str, torch.Tensor] = {}
    for name, values in first_sequence.fields.items():
        fields[name] = values.new_zeros((batch_size, width, *values.shape[1:]))
    timestamps = first_sequence.timestamps.new_zeros((batch_size, width))
    valid = torch.zeros((batch_size, width), dtype=torch.bool)

    for row, example in enumerate(examples):
        sequence = example.sequences[spec.name]
        source_length = sequence.length
        length = min(source_length, width)
        if length == 0:
            continue
        if keep_recent:
            source_start = source_length - length
        else:
            source_start = 0
        source_end = source_start + length
        target_start = width - length if left_pad else 0
        target_end = target_start + length
        for name, values in sequence.fields.items():
            fields[name][row, target_start:target_end] = values[source_start:source_end]
        timestamps[row, target_start:target_end] = sequence.timestamps[
            source_start:source_end
        ]
        valid[row, target_start:target_end] = True

    return PaddedSequence(fields=fields, timestamps=timestamps, valid=valid)


__all__ = [
    "PaddedSequence",
    "pad_domain",
    "stack_candidates",
    "stack_dense",
    "stack_group_ids",
    "stack_labels",
    "stack_sample_ids",
    "stack_scalars",
    "validate_examples",
]
