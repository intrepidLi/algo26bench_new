from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from .fidelity import Capability, FidelityCard
from .types import DataContext, LossPacket, MetricPacket, ModelOutput, RawExample


class DeviceMovable(Protocol):
    @property
    def batch_size(self) -> int: ...

    def to(self, device: Any) -> "DeviceMovable": ...


TBatch = TypeVar("TBatch", bound=DeviceMovable)


class Objective(Protocol[TBatch]):
    def loss(self, output: ModelOutput, batch: TBatch) -> LossPacket: ...

    def metrics(self, output: ModelOutput, batch: TBatch) -> Sequence[MetricPacket]: ...


@dataclass
class PreparedRecipe(Generic[TBatch]):
    name: str
    module: Any
    collator: Callable[[Sequence[RawExample]], TBatch]
    objective: Objective[TBatch]
    fidelity: FidelityCard
    capability: Capability


class Recipe(Protocol[TBatch]):
    name: str

    def prepare(self, context: DataContext) -> PreparedRecipe[TBatch]: ...


RecipeFactory = Callable[[Mapping[str, object]], Recipe[DeviceMovable]]
