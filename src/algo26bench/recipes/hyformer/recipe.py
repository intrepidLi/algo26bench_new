from __future__ import annotations

from algo26bench.core.fidelity import (
    Capability,
    FidelityCard,
    FidelityStatus,
    Mechanism,
)
from algo26bench.core.protocols import PreparedRecipe
from algo26bench.core.registry import register_recipe
from algo26bench.core.types import DataContext

from .batch import HyFormerBatch, HyFormerCollator
from .config import HyFormerConfig
from .model import HyFormerModel
from .objective import HyFormerObjective


@register_recipe("hyformer")
class HyFormerRecipe:
    name = "hyformer"

    def __init__(self, config: HyFormerConfig) -> None:
        self.config = config

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "HyFormerRecipe":
        return cls(HyFormerConfig.from_dict(raw))

    def prepare(self, context: DataContext) -> PreparedRecipe[HyFormerBatch]:
        groups = self.config.semantic_groups
        if not groups:
            groups = tuple(
                group
                for group in (
                    tuple(field.name for field in context.data_spec.scalar_fields),
                    tuple(field.name for field in context.data_spec.candidate_fields),
                )
                if group
            )
        covered = [name for group in groups for name in group]
        expected = {
            field.name
            for field in (
                context.data_spec.scalar_fields + context.data_spec.candidate_fields
            )
        }
        if len(covered) != len(set(covered)) or set(covered) != expected:
            raise ValueError(
                "HyFormer semantic_groups must cover every scalar/candidate field once"
            )

        fidelity = FidelityCard(
            recipe=self.name,
            paper="HyFormer (arXiv:2601.12681v2)",
            mechanisms=(
                Mechanism(
                    "semantic group tokenization",
                    FidelityStatus.FAITHFUL,
                    "Sec. 3.3.1",
                    "test_hyformer_semantic_token_count",
                ),
                Mechanism(
                    "sequence-specific query generation with pooled history",
                    FidelityStatus.FAITHFUL,
                    "Eq. 3-4 and Sec. 3.7",
                    "test_hyformer_multi_sequence_forward",
                ),
                Mechanism(
                    "layer-wise sequence encoding and query decoding",
                    FidelityStatus.FAITHFUL,
                    "Eq. 5/7-9 and Eq. 16",
                    "test_hyformer_padding_and_empty_sequence_invariance",
                ),
                Mechanism(
                    "RankMixer rewiring and independent per-token FFN",
                    FidelityStatus.FAITHFUL,
                    "Eq. 11-15",
                    "test_rankmixer_rewire_exact",
                ),
                Mechanism(
                    "GPU pooling and asynchronous AllReduce",
                    FidelityStatus.OUT_OF_SCOPE,
                    "Sec. 3.8",
                    reason="systems optimization is outside the academic MVP",
                ),
            ),
        )
        fidelity.validate()
        return PreparedRecipe(
            name=self.name,
            module=HyFormerModel(context, self.config, groups),
            collator=HyFormerCollator(context),
            objective=HyFormerObjective(context.task_set),
            fidelity=fidelity,
            capability=Capability(
                needs_multi_sequence=True,
                notes=("empty sequences are supported with exact mask semantics",),
            ),
        )
