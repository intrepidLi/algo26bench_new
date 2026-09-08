from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import torch

from .fidelity import FidelityCard
from .types import DataContext, LossPacket, MetricPacket, ModelOutput, RawExample


class DeviceMovable(Protocol):
    @property
    def batch_size(self) -> int: ...

    def to(self, device: Any) -> "DeviceMovable": ...


TBatch = TypeVar("TBatch", bound=DeviceMovable)


class Objective(Protocol[TBatch]):
    def loss(self, output: ModelOutput, batch: TBatch) -> LossPacket: ...

    def metrics(self, output: ModelOutput, batch: TBatch) -> Sequence[MetricPacket]: ...


@dataclass(frozen=True)
class DistributedInfo:
    """Snapshot of the process's place in a DDP world.

    A single-process run reports ``world_size=1``; ``enabled`` is the boolean
    consumers should branch on.  ``local_rank`` picks the CUDA device.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class TrainState:
    """Mutable state exposed to Callback methods."""

    step: int
    epoch: int
    module: torch.nn.Module
    optimizers: list[torch.optim.Optimizer]
    distributed: DistributedInfo = field(default_factory=DistributedInfo)


@runtime_checkable
class Callback(Protocol):
    """Optional hooks a recipe can attach to the training loop.

    Recipes only need to implement the hooks they care about; the default
    no-op is provided so a partial implementer still satisfies the protocol
    at runtime.
    """

    def on_epoch_start(self, state: TrainState) -> None: ...

    def on_step_end(self, state: TrainState) -> None: ...


@dataclass
class ParamGroup:
    """A slice of a module's parameters with its optimizer kind."""

    params: list[torch.nn.Parameter]
    kind: Literal["dense", "sparse"]


ParamGroupFn = Callable[[torch.nn.Module], list[ParamGroup]]


@dataclass
class PreparedRecipe(Generic[TBatch]):
    name: str
    module: torch.nn.Module
    collator: Callable[[Sequence[RawExample]], TBatch]
    objective: Objective[TBatch]
    fidelity: FidelityCard
    callbacks: tuple[Callback, ...] = ()
    param_groups: ParamGroupFn | None = None


class Recipe(Protocol[TBatch]):
    name: str

    def prepare(self, context: DataContext) -> PreparedRecipe[TBatch]: ...


RecipeFactory = Callable[[Mapping[str, object]], Recipe[DeviceMovable]]
