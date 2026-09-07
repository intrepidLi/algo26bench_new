from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

import algo26bench.recipes  # noqa: F401 -- registration belongs to bench
from algo26bench.core.registry import build_recipe
from algo26bench.core.types import DataContext
from algo26bench.data.synthetic import SyntheticRankingDataset, make_synthetic_context
from algo26bench.engine.trainer import Trainer, TrainingConfig

_SYNTHETIC_KEYS = {"kind", "train_size", "validation_size", "seed"}


def _load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run configuration must be a JSON object")
    unknown = set(raw) - {"recipe", "model", "data", "training"}
    if unknown:
        raise ValueError(f"unknown run config keys: {sorted(unknown)}")
    return raw


def run(config_path: Path) -> dict[str, Any]:
    raw = _load_config(config_path)
    recipe_name = str(raw["recipe"])
    model_config = dict(raw.get("model", {}))
    data_config = dict(raw.get("data", {}))
    training = TrainingConfig.from_dict(dict(raw.get("training", {})))
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)

    context, train_dataset, validation_dataset = _build_data(data_config, training.seed)
    recipe = build_recipe(recipe_name, model_config)
    prepared = recipe.prepare(context)
    return Trainer(training).fit(prepared, train_dataset, validation_dataset)


def _build_data(
    data_config: dict[str, Any], seed: int
) -> tuple[DataContext, Any, Any]:
    kind = str(data_config.get("kind", "synthetic"))
    if kind == "synthetic":
        unknown = set(data_config) - _SYNTHETIC_KEYS
        if unknown:
            raise ValueError(f"unknown synthetic data keys: {sorted(unknown)}")
        context = make_synthetic_context()
        data_seed = int(data_config.get("seed", seed))
        return (
            context,
            SyntheticRankingDataset(
                context, size=int(data_config.get("train_size", 128)), seed=data_seed
            ),
            SyntheticRankingDataset(
                context,
                size=int(data_config.get("validation_size", 64)),
                seed=data_seed + 1_000_003,
            ),
        )
    if kind == "pcvr":
        from algo26bench.data.pcvr import PCVRDataConfig, build_pcvr_datasets

        allowed = {"kind"} | set(PCVRDataConfig.__dataclass_fields__)
        unknown = set(data_config) - allowed
        if unknown:
            raise ValueError(f"unknown pcvr data keys: {sorted(unknown)}")
        payload = dict(data_config)
        payload.pop("kind", None)
        payload.setdefault("seed", seed)
        schema, train_dataset, valid_dataset = build_pcvr_datasets(
            PCVRDataConfig.from_dict(payload)
        )
        return schema.context, train_dataset, valid_dataset
    raise ValueError(f"unsupported data.kind={kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an algo26bench recipe")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
