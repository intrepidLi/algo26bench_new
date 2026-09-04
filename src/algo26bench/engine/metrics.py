from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from algo26bench.core.types import MetricPacket


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[targets == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def binary_logloss(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(
        np.mean(-(targets * np.log(scores) + (1 - targets) * np.log(1 - scores)))
    )


def aggregate_metric_packets(
    packets: Sequence[MetricPacket],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[MetricPacket]] = defaultdict(list)
    for packet in packets:
        grouped[packet.task].append(packet)

    result: dict[str, dict[str, float]] = {}
    for task, task_packets in grouped.items():
        scores = np.concatenate(
            [packet.scores.detach().cpu().numpy() for packet in task_packets]
        )
        targets = np.concatenate(
            [packet.targets.detach().cpu().numpy() for packet in task_packets]
        )
        metrics = {
            "auc": binary_auc(targets, scores),
            "logloss": binary_logloss(targets, scores),
        }
        if all(packet.group_ids is not None for packet in task_packets):
            group_ids = np.concatenate(
                [
                    packet.group_ids.detach().cpu().numpy()  # type: ignore[union-attr]
                    for packet in task_packets
                ]
            )
            group_aucs = []
            for group in np.unique(group_ids):
                selected = group_ids == group
                auc = binary_auc(targets[selected], scores[selected])
                if np.isfinite(auc):
                    group_aucs.append(auc)
            if group_aucs:
                metrics["group_auc"] = float(np.mean(group_aucs))
        result[task] = metrics
    return result
