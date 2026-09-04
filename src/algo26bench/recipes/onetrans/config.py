from __future__ import annotations

from dataclasses import dataclass

from algo26bench.core.registry import strict_dataclass


@dataclass(frozen=True)
class OneTransConfig:
    embedding_dim: int = 8
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 3
    num_ns_tokens: int = 4
    hidden_multiplier: int = 2
    dropout: float = 0.0
    merge_mode: str = "timestamp"
    max_sequence_tokens: int = 12
    pyramid_lengths: tuple[int, ...] = ()
    domain_order: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "OneTransConfig":
        normalized = dict(raw)
        for key in ("pyramid_lengths", "domain_order"):
            if key in normalized:
                normalized[key] = tuple(normalized[key])  # type: ignore[arg-type]
        config = strict_dataclass(cls, normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if min(
            self.embedding_dim,
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.num_ns_tokens,
            self.hidden_multiplier,
            self.max_sequence_tokens,
        ) <= 0:
            raise ValueError("OneTrans dimensions and layer counts must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.merge_mode not in {"timestamp", "concat"}:
            raise ValueError("merge_mode must be timestamp or concat")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pyramid_lengths:
            if len(self.pyramid_lengths) != self.num_layers:
                raise ValueError("pyramid_lengths must contain one value per layer")
            if any(
                length < 0 or length > self.max_sequence_tokens
                for length in self.pyramid_lengths
            ):
                raise ValueError("pyramid length is outside the sequence token range")
            if any(
                later > earlier
                for earlier, later in zip(
                    self.pyramid_lengths, self.pyramid_lengths[1:]
                )
            ):
                raise ValueError("pyramid_lengths must be non-increasing")

    def resolved_pyramid_lengths(self) -> tuple[int, ...]:
        if self.pyramid_lengths:
            return self.pyramid_lengths
        return tuple(
            round(self.max_sequence_tokens * (self.num_layers - layer - 1) / self.num_layers)
            for layer in range(self.num_layers)
        )
