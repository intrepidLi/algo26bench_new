from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping


Tensor = Any


@dataclass(frozen=True)
class CategoricalField:
    name: str
    vocab_size: int
    width: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name cannot be empty")
        if self.vocab_size <= 0:
            raise ValueError(f"{self.name}: vocab_size must be positive")
        if self.width <= 0:
            raise ValueError(f"{self.name}: width must be positive")


@dataclass(frozen=True)
class SequenceDomainSpec:
    name: str
    fields: tuple[CategoricalField, ...]
    max_len: int

    def __post_init__(self) -> None:
        if not self.name or not self.fields:
            raise ValueError("a sequence domain needs a name and at least one field")
        if self.max_len <= 0:
            raise ValueError(f"{self.name}: max_len must be positive")


class TaskKind(Enum):
    BINARY = auto()
    MULTI_ACTION_SOFTMAX = auto()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_key: str
    kind: TaskKind = TaskKind.BINARY
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.kind == TaskKind.BINARY and self.num_classes != 2:
            raise ValueError(f"{self.name}: BINARY task must have num_classes == 2")
        if self.kind == TaskKind.MULTI_ACTION_SOFTMAX and self.num_classes <= 2:
            raise ValueError(
                f"{self.name}: MULTI_ACTION_SOFTMAX task must have num_classes > 2"
            )


@dataclass(frozen=True)
class TaskSet:
    tasks: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        names = [task.name for task in self.tasks]
        if not names or len(names) != len(set(names)):
            raise ValueError("task names must be non-empty and unique")


@dataclass(frozen=True)
class DataSpec:
    scalar_fields: tuple[CategoricalField, ...]
    candidate_fields: tuple[CategoricalField, ...]
    user_dense_dim: int
    item_dense_dim: int
    sequence_domains: tuple[SequenceDomainSpec, ...]

    def __post_init__(self) -> None:
        if self.user_dense_dim < 0 or self.item_dense_dim < 0:
            raise ValueError("dense dims cannot be negative")
        all_names = [field.name for field in self.scalar_fields + self.candidate_fields]
        if len(all_names) != len(set(all_names)):
            raise ValueError("scalar and candidate field names must be unique")
        sequence_names = [domain.name for domain in self.sequence_domains]
        if not sequence_names or len(sequence_names) != len(set(sequence_names)):
            raise ValueError("sequence domain names must be non-empty and unique")

    @property
    def dense_dim(self) -> int:
        return self.user_dense_dim + self.item_dense_dim

    @property
    def sequence_names(self) -> tuple[str, ...]:
        return tuple(domain.name for domain in self.sequence_domains)

    def sequence(self, name: str) -> SequenceDomainSpec:
        for domain in self.sequence_domains:
            if domain.name == name:
                return domain
        raise KeyError(name)


@dataclass(frozen=True)
class DataContext:
    data_spec: DataSpec
    task_set: TaskSet


@dataclass
class RawSequence:
    fields: Mapping[str, Tensor]
    timestamps: Tensor

    @property
    def length(self) -> int:
        return int(self.timestamps.shape[0])

    def validate(self, spec: SequenceDomainSpec) -> None:
        if self.timestamps.ndim != 1:
            raise ValueError(f"{spec.name}.timestamps must be one-dimensional")
        expected = {field.name for field in spec.fields}
        if set(self.fields) != expected:
            raise ValueError(
                f"{spec.name}: expected fields {sorted(expected)}, got {sorted(self.fields)}"
            )
        for name, values in self.fields.items():
            if values.shape[0] != self.length:
                raise ValueError(f"{spec.name}.{name}: length mismatch")


@dataclass
class RawExample:
    scalars: Mapping[str, Tensor]
    dense: Tensor
    candidates: Mapping[str, Tensor]
    sequences: Mapping[str, RawSequence]
    labels: Mapping[str, Tensor]
    sample_id: Tensor
    group_id: Tensor | None = None


@dataclass
class ModelOutput:
    logits: Mapping[str, Tensor]
    representation: Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossPacket:
    loss_sum: Tensor
    weight: Tensor

    @property
    def mean(self) -> Tensor:
        return self.loss_sum / self.weight.clamp_min(1.0)


@dataclass
class MetricPacket:
    task: str
    scores: Tensor
    targets: Tensor
    weights: Tensor
    sample_ids: Tensor
    group_ids: Tensor | None = None
