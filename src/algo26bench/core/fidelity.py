from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FidelityStatus(str, Enum):
    FAITHFUL = "faithful"
    ADAPTED = "adapted"


@dataclass(frozen=True)
class Mechanism:
    name: str
    status: FidelityStatus
    paper_ref: str
    test_id: str | None = None
    reason: str | None = None

    def validate(self) -> None:
        if self.status == FidelityStatus.FAITHFUL and not self.test_id:
            raise ValueError(f"faithful mechanism {self.name!r} requires a test_id")
        if self.status != FidelityStatus.FAITHFUL and not self.reason:
            raise ValueError(f"non-faithful mechanism {self.name!r} requires a reason")


@dataclass(frozen=True)
class FidelityCard:
    recipe: str
    paper: str
    mechanisms: tuple[Mechanism, ...]

    def validate(self) -> None:
        if not self.mechanisms:
            raise ValueError("fidelity card cannot be empty")
        for mechanism in self.mechanisms:
            mechanism.validate()

    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in FidelityStatus}
        for mechanism in self.mechanisms:
            counts[mechanism.status.value] += 1
        return counts
