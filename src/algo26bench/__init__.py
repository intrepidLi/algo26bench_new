"""Fidelity-aware academic ranking benchmark."""

from .core.registry import available_recipes, build_recipe

__all__ = ["available_recipes", "build_recipe"]
__version__ = "0.1.0"
