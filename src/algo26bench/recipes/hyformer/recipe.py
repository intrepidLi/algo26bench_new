from __future__ import annotations

from algo26bench.core.fidelity import FidelityCard, FidelityStatus, Mechanism
from algo26bench.core.protocols import PreparedRecipe
from algo26bench.core.registry import register_recipe
from algo26bench.core.types import DataContext
from algo26bench.engine.objectives import choose_objective
from algo26bench.recipes.common import split_embedding_params

from .batch import HyFormerBatch, HyFormerCollator
from .config import ENCODER_VARIANTS, HyFormerConfig
from .model import HyFormerModel


@register_recipe("hyformer")
class HyFormerRecipe:
    name = "hyformer"

    def __init__(self, config: HyFormerConfig) -> None:
        self.config = config

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "HyFormerRecipe":
        return cls(HyFormerConfig.from_dict(raw))

    def prepare(self, context: DataContext) -> PreparedRecipe[HyFormerBatch]:
        if self.config.ns_tokenizer_type == "rankmixer":
            # rankmixer flattens all fields, so semantic_groups is unused. Pass
            # an empty groups tuple downstream and skip the coverage check.
            groups: tuple[tuple[str, ...], ...] = ()
        else:
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

        encoder_desc, encoder_ref = ENCODER_VARIANTS[self.config.sequence_encoder]
        encoder_mechanisms = (
            Mechanism(
                f"sequence encoding strategy in use: {encoder_desc}",
                FidelityStatus.FAITHFUL,
                f"Sec. 3.4.1 {encoder_ref}",
                "test_hyformer_encoder_variant_selection",
            ),
        )
        if self.config.sequence_encoder == "longer":
            encoder_mechanisms += (
                Mechanism(
                    "S_short is the most recent L_H behaviours of the sequence",
                    FidelityStatus.FAITHFUL,
                    "Eq. 6; recent-k beats uniform-k and learnable-k in "
                    "LONGER (arXiv:2505.04421) Table 2",
                    "test_hyformer_longer_encoder_selects_recent_tokens",
                ),
                Mechanism(
                    "LONGER's reverse-causal cross-attention mask",
                    FidelityStatus.ADAPTED,
                    "LONGER (arXiv:2505.04421) Eq. 10",
                    reason=(
                        "HyFormer Eq. 6 states no mask, and this framework stores "
                        "behaviours earliest-to-latest rather than LONGER's reversed "
                        "layout, so the compression attends over every valid behaviour"
                    ),
                ),
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
                    "key/value states recomputed per layer from the raw sequence",
                    FidelityStatus.FAITHFUL,
                    "Eq. 5-8",
                    "test_hyformer_layerwise_kv_reads_raw_sequence",
                ),
                Mechanism(
                    "query decoding by cross-attention over the layer's key/values",
                    FidelityStatus.FAITHFUL,
                    "Eq. 9 and Eq. 16",
                    "test_hyformer_padding_and_empty_sequence_invariance",
                ),
                *encoder_mechanisms,
                Mechanism(
                    "RankMixer rewiring and independent per-token FFN",
                    FidelityStatus.FAITHFUL,
                    "Eq. 11-15",
                    "test_rankmixer_rewire_exact",
                ),
                Mechanism(
                    "GPU pooling and asynchronous AllReduce",
                    FidelityStatus.ADAPTED,
                    "Sec. 3.8",
                    reason=(
                        "systems throughput optimization; Sec. 3.8.2 reports the "
                        "resulting one-step staleness as lossless, so omitting it "
                        "does not change model quality"
                    ),
                ),
            ),
        )
        fidelity.validate()
        module = HyFormerModel(context, self.config, groups)
        return PreparedRecipe(
            name=self.name,
            module=module,
            collator=HyFormerCollator(context),
            objective=choose_objective(context.task_set),
            fidelity=fidelity,
            param_groups=split_embedding_params,
        )
