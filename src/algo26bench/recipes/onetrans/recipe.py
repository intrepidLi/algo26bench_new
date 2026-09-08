from __future__ import annotations

from algo26bench.core.fidelity import FidelityCard, FidelityStatus, Mechanism
from algo26bench.core.protocols import PreparedRecipe
from algo26bench.core.registry import register_recipe
from algo26bench.core.types import DataContext
from algo26bench.engine.objectives import choose_objective
from algo26bench.recipes.common import split_embedding_params

from .batch import OneTransBatch, OneTransCollator
from .config import OneTransConfig
from .model import OneTransModel


@register_recipe("onetrans")
class OneTransRecipe:
    name = "onetrans"

    def __init__(self, config: OneTransConfig) -> None:
        self.config = config

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "OneTransRecipe":
        return cls(OneTransConfig.from_dict(raw))

    def prepare(self, context: DataContext) -> PreparedRecipe[OneTransBatch]:
        fidelity = FidelityCard(
            recipe=self.name,
            paper="OneTrans (arXiv:2510.26104v2)",
            mechanisms=(
                Mechanism(
                    "unified S-token then NS-token sequence",
                    FidelityStatus.FAITHFUL,
                    "Eq. 3 and Sec. 3.2",
                    "test_onetrans_timestamp_merge_and_token_order",
                ),
                Mechanism(
                    "RMSNorm causal pre-norm block",
                    FidelityStatus.FAITHFUL,
                    "Eq. 4-5 and Sec. 3.3",
                    "test_onetrans_causal_future_isolation",
                ),
                Mechanism(
                    "shared S and token-specific NS QKV/FFN",
                    FidelityStatus.FAITHFUL,
                    "Eq. 11-13",
                    "test_onetrans_mixed_parameterization",
                ),
                Mechanism(
                    "tail-query pyramid with full current K/V",
                    FidelityStatus.FAITHFUL,
                    "Sec. 3.4",
                    "test_onetrans_pyramid_schedule",
                ),
                Mechanism(
                    "cross-request KV cache and fused kernels",
                    FidelityStatus.ADAPTED,
                    "Sec. 3.5",
                    reason="serving infrastructure is outside the academic MVP",
                ),
                Mechanism(
                    "paper optimizer pair",
                    FidelityStatus.ADAPTED,
                    "Sec. 4.1.4",
                    reason="the shared runner uses Adagrad plus AdamW by default",
                ),
            ),
        )
        fidelity.validate()
        return PreparedRecipe(
            name=self.name,
            module=OneTransModel(context, self.config),
            collator=OneTransCollator(context, self.config),
            objective=choose_objective(context.task_set),
            fidelity=fidelity,
            param_groups=split_embedding_params,
        )
