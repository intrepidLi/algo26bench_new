from __future__ import annotations

from dataclasses import dataclass

from algo26bench.core.registry import strict_dataclass


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
    semantic_groups: tuple[tuple[str, ...], ...] = ()

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
        if self.sequence_encoder not in {"transformer", "swiglu"}:
            raise ValueError("sequence_encoder must be transformer or swiglu")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
