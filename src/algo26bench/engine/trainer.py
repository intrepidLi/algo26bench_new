from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from algo26bench.core.protocols import ParamGroupFn, PreparedRecipe, TrainState
from algo26bench.core.registry import strict_dataclass
from algo26bench.core.types import MetricPacket, RawExample

from .metrics import aggregate_metric_packets


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 1
    max_steps: int | None = None
    dense_learning_rate: float = 1e-3
    embedding_learning_rate: float = 1e-2
    weight_decay: float = 0.0
    gradient_clip_norm: float = 5.0
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 0
    output_dir: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TrainingConfig":
        config = strict_dataclass(cls, raw)
        if config.batch_size <= 0 or config.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if config.max_steps is not None and config.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        return config


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _loader(
        self,
        dataset: Dataset[RawExample],
        prepared: PreparedRecipe[Any],
        *,
        shuffle: bool,
    ) -> DataLoader[Any]:
        common = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "collate_fn": prepared.collator,
        }
        if isinstance(dataset, IterableDataset):
            return DataLoader(dataset, **common)
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            shuffle=shuffle,
            generator=generator,
            **common,
        )

    def _optimizers(
        self,
        module: torch.nn.Module,
        param_groups_fn: ParamGroupFn | None,
    ) -> list[torch.optim.Optimizer]:
        if param_groups_fn is None:
            return [
                torch.optim.AdamW(
                    module.parameters(),
                    lr=self.config.dense_learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            ]
        groups = param_groups_fn(module)
        sparse_params = [p for g in groups if g.kind == "sparse" for p in g.params]
        dense_params = [p for g in groups if g.kind == "dense" for p in g.params]
        optimizers: list[torch.optim.Optimizer] = []
        if sparse_params:
            optimizers.append(
                torch.optim.Adagrad(
                    sparse_params, lr=self.config.embedding_learning_rate
                )
            )
        if dense_params:
            optimizers.append(
                torch.optim.AdamW(
                    dense_params,
                    lr=self.config.dense_learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            )
        return optimizers

    def fit(
        self,
        prepared: PreparedRecipe[Any],
        train_dataset: Dataset[RawExample],
        validation_dataset: Dataset[RawExample],
    ) -> dict[str, Any]:
        self._set_seed()
        module = prepared.module.to(self.device)
        optimizers = self._optimizers(module, prepared.param_groups)
        state = TrainState(step=0, epoch=0, module=module, optimizers=optimizers)
        train_loader = self._loader(train_dataset, prepared, shuffle=True)
        mean_losses: list[float] = []

        module.train()
        should_stop = False
        for epoch in range(self.config.epochs):
            state.epoch = epoch
            for callback in prepared.callbacks:
                callback.on_epoch_start(state)
            for batch in train_loader:
                if (
                    self.config.max_steps is not None
                    and state.step >= self.config.max_steps
                ):
                    should_stop = True
                    break
                loss = self._step(state, batch, prepared)
                mean_losses.append(loss)
                for callback in prepared.callbacks:
                    callback.on_step_end(state)
            if should_stop:
                break

        metrics = self.evaluate(prepared, validation_dataset)
        summary: dict[str, Any] = {
            "recipe": prepared.name,
            "device": str(self.device),
            "steps": state.step,
            "train_loss": float(np.mean(mean_losses)) if mean_losses else float("nan"),
            "validation": metrics,
            "parameters": sum(parameter.numel() for parameter in module.parameters()),
            "fidelity": prepared.fidelity.counts(),
        }
        if self.config.output_dir:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(module.state_dict(), output_dir / "model.pt")
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output_dir / "training_config.json").write_text(
                json.dumps(asdict(self.config), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return summary

    def _step(
        self,
        state: TrainState,
        batch: Any,
        prepared: PreparedRecipe[Any],
    ) -> float:
        batch = batch.to(self.device)
        for optimizer in state.optimizers:
            optimizer.zero_grad(set_to_none=True)
        output = state.module(batch)
        loss = prepared.objective.loss(output, batch).mean
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            state.module.parameters(), self.config.gradient_clip_norm
        )
        for optimizer in state.optimizers:
            optimizer.step()
        state.step += 1
        return float(loss.detach().cpu())

    @torch.no_grad()
    def evaluate(
        self,
        prepared: PreparedRecipe[Any],
        dataset: Dataset[RawExample],
    ) -> dict[str, dict[str, float]]:
        prepared.module.eval()
        packets: list[MetricPacket] = []
        for batch in self._loader(dataset, prepared, shuffle=False):
            batch = batch.to(self.device)
            output = prepared.module(batch)
            packets.extend(prepared.objective.metrics(output, batch))
        return aggregate_metric_packets(packets)
