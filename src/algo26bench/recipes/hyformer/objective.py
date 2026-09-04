from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from algo26bench.core.types import LossPacket, MetricPacket, ModelOutput, TaskSet

from .batch import HyFormerBatch


class HyFormerObjective:
    def __init__(self, task_set: TaskSet) -> None:
        self.task_set = task_set

    def loss(self, output: ModelOutput, batch: HyFormerBatch) -> LossPacket:
        loss_sum = torch.zeros((), device=batch.sample_ids.device)
        weight = torch.zeros((), device=batch.sample_ids.device)
        for task in self.task_set.tasks:
            logits = output.logits[task.name].reshape(-1)
            targets = batch.labels[task.label_key].float().reshape(-1)
            loss_sum = loss_sum + F.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            )
            weight = weight + targets.new_tensor(targets.numel())
        return LossPacket(loss_sum, weight)

    def metrics(
        self, output: ModelOutput, batch: HyFormerBatch
    ) -> Sequence[MetricPacket]:
        return tuple(
            MetricPacket(
                task=task.name,
                scores=torch.sigmoid(output.logits[task.name].reshape(-1)),
                targets=batch.labels[task.label_key].float().reshape(-1),
                weights=torch.ones_like(batch.labels[task.label_key]).reshape(-1),
                sample_ids=batch.sample_ids.reshape(-1),
                group_ids=(
                    None if batch.group_ids is None else batch.group_ids.reshape(-1)
                ),
            )
            for task in self.task_set.tasks
        )
