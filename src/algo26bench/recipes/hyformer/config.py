from __future__ import annotations

from dataclasses import dataclass

from algo26bench.core.registry import strict_dataclass


# The three sequence representation encoding strategies of HyFormer Sec. 3.4.1,
# mapped to the paper equation each one implements.
ENCODER_VARIANTS: dict[str, tuple[str, str]] = {
    "transformer": ("full Transformer encoding", "Eq. 5"),
    "longer": (
        "LONGER-style cross-attention compression onto the most recent L_H behaviours",
        "Eq. 6",
    ),
    "swiglu": ("decoder-style attention-free SwiGLU encoding", "Eq. 7"),
}


@dataclass(frozen=True)
class HyFormerConfig:
    embedding_dim: int = 8
    d_model: int = 40
    num_heads: int = 4
    num_layers: int = 2
    num_queries: int = 1
    hidden_multiplier: int = 2
    dropout: float = 0.0
    sequence_encoder: str = "transformer"
    num_short_tokens: int = 4
    semantic_groups: tuple[tuple[str, ...], ...] = ()
    emb_skip_threshold: int = 0
    ns_tokenizer_type: str = "semantic_group"  # "semantic_group" | "rankmixer"
    user_ns_tokens: int = 0
    item_ns_tokens: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "HyFormerConfig":
        normalized = dict(raw)
        if "semantic_groups" in normalized:
            normalized["semantic_groups"] = tuple(
                tuple(str(name) for name in group)  # type: ignore[union-attr]
                for group in normalized["semantic_groups"]  # type: ignore[union-attr]
            )
        config = strict_dataclass(cls, normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if min(
            self.embedding_dim,
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.num_queries,
            self.hidden_multiplier,
        ) <= 0:
            raise ValueError("HyFormer dimensions and layer counts must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.sequence_encoder not in ENCODER_VARIANTS:
            raise ValueError(
                f"sequence_encoder must be one of {sorted(ENCODER_VARIANTS)}"
            )
        if self.num_short_tokens <= 0:
            raise ValueError("num_short_tokens must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.ns_tokenizer_type not in ("semantic_group", "rankmixer"):
            raise ValueError(
                "ns_tokenizer_type must be 'semantic_group' or 'rankmixer'"
            )
        if self.ns_tokenizer_type == "rankmixer":
            if self.user_ns_tokens <= 0 or self.item_ns_tokens <= 0:
                raise ValueError(
                    "rankmixer ns_tokenizer requires user_ns_tokens > 0 and item_ns_tokens > 0"
                )
            if self.semantic_groups:
                raise ValueError(
                    "rankmixer ns_tokenizer does not use semantic_groups; leave it empty"
                )
