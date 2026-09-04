from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from algo26bench.core.types import (
    DataSpec,
    RaggedSequence,
    RawBatch,
    RawExample,
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


def collate_raw_examples(
    examples: Sequence[RawExample], data_spec: DataSpec, task_set: TaskSet
) -> RawBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")

    for example in examples:
        if set(example.scalars) != {field.name for field in data_spec.scalar_fields}:
            raise ValueError("scalar fields do not match DataSpec")
        if set(example.candidates) != {
            field.name for field in data_spec.candidate_fields
        }:
            raise ValueError("candidate fields do not match DataSpec")
        if set(example.sequences) != set(data_spec.sequence_names):
            raise ValueError("sequence domains do not match DataSpec")
        if set(example.labels) != {task.label_key for task in task_set.tasks}:
            raise ValueError("labels do not match TaskSet")
        if example.dense.shape != (data_spec.dense_dim,):
            raise ValueError("dense feature width does not match DataSpec")
        for domain in data_spec.sequence_domains:
            example.sequences[domain.name].validate(domain)

    sequences: dict[str, RaggedSequence] = {}
    for domain in data_spec.sequence_domains:
        offsets = [0]
        fields: dict[str, list[torch.Tensor]] = {
            field.name: [] for field in domain.fields
        }
        timestamps: list[torch.Tensor] = []
        for example in examples:
            sequence = example.sequences[domain.name]
            offsets.append(offsets[-1] + sequence.length)
            timestamps.append(sequence.timestamps)
            for name, values in sequence.fields.items():
                fields[name].append(values)

        ragged = RaggedSequence(
            fields={name: torch.cat(parts, dim=0) for name, parts in fields.items()},
            offsets=torch.tensor(offsets, dtype=torch.long),
            timestamps=torch.cat(timestamps, dim=0),
        )
        ragged.validate()
        sequences[domain.name] = ragged

    has_group = [example.group_id is not None for example in examples]
    if any(has_group) and not all(has_group):
        raise ValueError("group_id must be present for every example or none")

    return RawBatch(
        scalars={
            field.name: torch.stack(
                [example.scalars[field.name] for example in examples]
            )
            for field in data_spec.scalar_fields
        },
        dense=torch.stack([example.dense for example in examples]),
        candidates={
            field.name: torch.stack(
                [example.candidates[field.name] for example in examples]
            )
            for field in data_spec.candidate_fields
        },
        sequences=sequences,
        labels={
            task.label_key: torch.stack(
                [example.labels[task.label_key] for example in examples]
            ).float()
            for task in task_set.tasks
        },
        sample_ids=torch.stack([example.sample_id for example in examples]).long(),
        group_ids=(
            torch.stack([example.group_id for example in examples]).long()  # type: ignore[arg-type]
            if all(has_group)
            else None
        ),
    )


def pad_ragged_sequence(
    sequence: RaggedSequence,
    max_len: int,
    *,
    left_pad: bool = False,
    keep_recent: bool = True,
) -> PaddedSequence:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    sequence.validate()
    batch_size = sequence.batch_size
    fields = {
        name: values.new_zeros((batch_size, max_len, *values.shape[1:]))
        for name, values in sequence.fields.items()
    }
    timestamps = sequence.timestamps.new_zeros((batch_size, max_len))
    valid = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for row in range(batch_size):
        source_start = int(sequence.offsets[row])
        source_end = int(sequence.offsets[row + 1])
        length = min(source_end - source_start, max_len)
        if length == 0:
            continue
        if keep_recent:
            source_start = source_end - length
        else:
            source_end = source_start + length
        target_start = max_len - length if left_pad else 0
        target_end = target_start + length
        for name, values in sequence.fields.items():
            fields[name][row, target_start:target_end] = values[
                source_start:source_end
            ]
        timestamps[row, target_start:target_end] = sequence.timestamps[
            source_start:source_end
        ]
        valid[row, target_start:target_end] = True

    return PaddedSequence(fields=fields, timestamps=timestamps, valid=valid)
