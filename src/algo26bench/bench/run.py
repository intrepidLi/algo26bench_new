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
from algo26bench.data.synthetic import SyntheticRankingDataset, make_synthetic_context
from algo26bench.engine.trainer import Trainer, TrainingConfig


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
    unknown_data = set(data_config) - {"kind", "train_size", "validation_size", "seed"}
    if unknown_data:
        raise ValueError(f"unknown data config keys: {sorted(unknown_data)}")
    if data_config.get("kind", "synthetic") != "synthetic":
        raise ValueError(
            "this MVP CLI ships only a synthetic adapter; real datasets should "
            "construct RawExample objects against the same DataContext contract"
        )
    training = TrainingConfig.from_dict(dict(raw.get("training", {})))
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)

    context = make_synthetic_context()
    recipe = build_recipe(recipe_name, model_config)
    prepared = recipe.prepare(context)
    data_seed = int(data_config.get("seed", training.seed))
    train_dataset = SyntheticRankingDataset(
        context, size=int(data_config.get("train_size", 128)), seed=data_seed
    )
    validation_dataset = SyntheticRankingDataset(
        context,
        size=int(data_config.get("validation_size", 64)),
        seed=data_seed + 1_000_003,
    )
    return Trainer(training).fit(prepared, train_dataset, validation_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an algo26bench recipe")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
