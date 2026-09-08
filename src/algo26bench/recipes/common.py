"""Helpers shared by every recipe registered against the benchmark.

Only functions that at least two recipes need land here; single-recipe
utilities stay in that recipe's directory (v3.md §2 rule 2).
"""

from __future__ import annotations

import torch.nn as nn

from algo26bench.core.protocols import ParamGroup


def split_embedding_params(module: nn.Module) -> list[ParamGroup]:
    """Partition parameters into sparse (embedding tables) and dense.

    Every recipe in this repo composes an ``EmbeddingBank`` whose
    ``tables`` submodule is an ``nn.ModuleDict`` of ``nn.Embedding`` layers.
    Their parameters go into an Adagrad optimizer, everything else into
    AdamW. Trainer consumes this via ``PreparedRecipe.param_groups``.
    """

    sparse_ids: set[int] = set()
    sparse_params: list[nn.Parameter] = []
    for submodule in module.modules():
        if isinstance(submodule, nn.Embedding):
            for parameter in submodule.parameters(recurse=False):
                if id(parameter) not in sparse_ids and parameter.requires_grad:
                    sparse_ids.add(id(parameter))
                    sparse_params.append(parameter)

    dense_params: list[nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in sparse_ids:
            continue
        dense_params.append(parameter)

    groups: list[ParamGroup] = []
    if sparse_params:
        groups.append(ParamGroup(params=sparse_params, kind="sparse"))
    if dense_params:
        groups.append(ParamGroup(params=dense_params, kind="dense"))
    return groups


__all__ = ["split_embedding_params"]
