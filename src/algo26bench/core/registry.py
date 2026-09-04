from __future__ import annotations

from typing import Any, Mapping, TypeVar


T = TypeVar("T")
_RECIPES: dict[str, type[Any]] = {}


def register_recipe(name: str):
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("recipe name cannot be empty")

    def decorator(recipe_type: type[T]) -> type[T]:
        previous = _RECIPES.get(normalized)
        if previous is not None and previous is not recipe_type:
            raise ValueError(f"duplicate recipe registration: {normalized}")
        _RECIPES[normalized] = recipe_type
        return recipe_type

    return decorator


def available_recipes() -> tuple[str, ...]:
    return tuple(sorted(_RECIPES))


def build_recipe(name: str, config: Mapping[str, object] | None = None):
    normalized = name.strip().lower()
    if normalized not in _RECIPES:
        raise KeyError(f"unknown recipe {name!r}; available={available_recipes()}")
    recipe_type = _RECIPES[normalized]
    return recipe_type.from_dict(dict(config or {}))


def strict_dataclass(cls: type[T], raw: Mapping[str, object]) -> T:
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)
