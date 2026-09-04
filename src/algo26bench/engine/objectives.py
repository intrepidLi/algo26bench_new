"""Generic objectives for experiments assembled directly by the bench layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
import torch.nn.functional as F

from algo26bench.core.types import LossPacket, MetricPacket, ModelOutput, TaskSet


class LabeledBatch(Protocol):
    labels: Mapping[str, torch.Tensor]
    sample_ids: torch.Tensor
    group_ids: torch.Tensor | None


class BinaryObjective:
    def __init__(self, task_set: TaskSet) -> None:
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


__all__ = ["BinaryObjective", "LabeledBatch"]
