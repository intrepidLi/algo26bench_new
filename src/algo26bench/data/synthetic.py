from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from algo26bench.core.types import (
    CategoricalField,
    DataContext,
    DataSpec,
    RawExample,
    RawSequence,
    SequenceDomainSpec,
    TaskSet,
    TaskSpec,
)


def make_synthetic_context() -> DataContext:
    item = CategoricalField("item_id", 257)
    category = CategoricalField("category_id", 33)
    return DataContext(
        data_spec=DataSpec(
            scalar_fields=(
                CategoricalField("user_id", 129),
                CategoricalField("context_id", 17),
            ),
            candidate_fields=(item, category),
            user_dense_dim=3,
            item_dense_dim=0,
            sequence_domains=(
                SequenceDomainSpec("watch", (item, category), max_len=8),
                SequenceDomainSpec("search", (item, category), max_len=5),
            ),
        ),
        task_set=TaskSet((TaskSpec("pcvr", "pcvr"),)),
    )


class SyntheticRankingDataset(Dataset[RawExample]):
    """Small deterministic dataset for contract tests and smoke runs only."""

    def __init__(
        self,
        context: DataContext,
        size: int = 128,
        seed: int = 2026,
        *,
        force_lengths: Sequence[int] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.context = context
        self.size = size
        self.seed = seed
        self.force_lengths = tuple(force_lengths) if force_lengths is not None else None

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> RawExample:
        if not 0 <= index < self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index * 104729)
        spec = self.context.data_spec

        user_id = torch.randint(1, 129, (), generator=generator)
        context_id = torch.randint(1, 17, (), generator=generator)
        candidate_item = torch.randint(1, 257, (), generator=generator)
        candidate_category = torch.randint(1, 33, (), generator=generator)
        dense = torch.randn(spec.dense_dim, generator=generator)

        sequences: dict[str, RawSequence] = {}
        any_match = False
        for domain_index, domain in enumerate(spec.sequence_domains):
            if self.force_lengths is None:
                length = int(
                    torch.randint(0, domain.max_len + 1, (), generator=generator)
                )
            else:
                length = min(self.force_lengths[domain_index], domain.max_len)
            items = torch.randint(1, 257, (length,), generator=generator)
            categories = torch.randint(1, 33, (length,), generator=generator)
            if length and index % 4 == domain_index:
                items[-1] = candidate_item
            any_match = any_match or bool(torch.any(items == candidate_item))
            gaps = torch.randint(1, 8, (length,), generator=generator)
            timestamps = torch.cumsum(gaps, dim=0) + index * 100
            sequences[domain.name] = RawSequence(
                fields={"item_id": items, "category_id": categories},
                timestamps=timestamps.long(),
            )

        signal = (
            1.2 * float(any_match)
            + 0.45 * float(int(user_id) % 3 == int(candidate_category) % 3)
            + 0.35 * float(dense[0])
            - 0.75
        )
        probability = torch.sigmoid(torch.tensor(signal))
        label = torch.bernoulli(probability, generator=generator)
        return RawExample(
            scalars={"user_id": user_id, "context_id": context_id},
            dense=dense,
            candidates={
                "item_id": candidate_item,
                "category_id": candidate_category,
            },
            sequences=sequences,
            labels={"pcvr": label},
            sample_id=torch.tensor(index, dtype=torch.long),
            group_id=user_id.long(),
        )
