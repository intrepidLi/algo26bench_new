"""Built-in recipe registration."""

from .hyformer.recipe import HyFormerRecipe
from .onetrans.recipe import OneTransRecipe

__all__ = ["HyFormerRecipe", "OneTransRecipe"]
