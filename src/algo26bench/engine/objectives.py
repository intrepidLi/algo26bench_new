"""Generic objectives for experiments assembled directly by the bench layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
import torch.nn.functional as F

from algo26bench.core.types import (
    LossPacket,
    MetricPacket,
    ModelOutput,
    TaskKind,
    TaskSet,
)


class LabeledBatch(Protocol):
    labels: Mapping[str, torch.Tensor]
    sample_ids: torch.Tensor
    group_ids: torch.Tensor | None


class BinaryObjective:
    def __init__(self, task_set: TaskSet) -> None:
        for task in task_set.tasks:
            if task.kind != TaskKind.BINARY:
                raise ValueError(
                    f"BinaryObjective only handles TaskKind.BINARY tasks, "
                    f"got {task.kind} for task {task.name!r}"
                )
        self.task_set = task_set

    def loss(self, output: ModelOutput, batch: LabeledBatch) -> LossPacket:
        loss_sum = torch.zeros((), device=batch.sample_ids.device)
        weight = torch.zeros((), device=batch.sample_ids.device)
        for task in self.task_set.tasks:
            logits = output.logits[task.name].reshape(-1)
            targets = batch.labels[task.label_key].float().reshape(-1)
            loss_sum = loss_sum + F.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            )
            weight = weight + targets.new_tensor(targets.numel())
        return LossPacket(loss_sum=loss_sum, weight=weight)

    def metrics(
        self, output: ModelOutput, batch: LabeledBatch
    ) -> Sequence[MetricPacket]:
        packets = []
        for task in self.task_set.tasks:
            targets = batch.labels[task.label_key].float().reshape(-1)
            packets.append(
                MetricPacket(
                    task=task.name,
                    scores=torch.sigmoid(output.logits[task.name].reshape(-1)),
                    targets=targets,
                    weights=torch.ones_like(targets),
                    sample_ids=batch.sample_ids.reshape(-1),
                    group_ids=(
                        None
                        if batch.group_ids is None
                        else batch.group_ids.reshape(-1)
                    ),
                )
            )
        return packets


def choose_objective(task_set: TaskSet) -> BinaryObjective:
    """Pick the right objective based on the data-provided task kinds.

    Softmax objectives (for TokenFormer's multi-action head) will land in a
    follow-up branch; today only BINARY is wired.
    """

    kinds = {task.kind for task in task_set.tasks}
    if kinds == {TaskKind.BINARY}:
        return BinaryObjective(task_set)
    raise ValueError(f"unsupported task-kind mix: {sorted(kind.name for kind in kinds)}")


__all__ = ["BinaryObjective", "LabeledBatch", "choose_objective"]
