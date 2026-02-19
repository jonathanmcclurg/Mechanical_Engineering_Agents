"""Load and validate analysis recipes from YAML files."""

import os
from pathlib import Path
from typing import Optional
import yaml

from src.schemas.analysis_recipe import AnalysisRecipe, MethodConfig, DataRequirement, StatisticalMethod


VALID_RECIPE_MODES = {"strict", "advisory", "off"}


def normalize_recipe_mode(recipe_mode: Optional[str] = None) -> str:
    """Normalize recipe mode string with safe defaults."""
    if recipe_mode is None:
        from config.settings import get_settings
        recipe_mode = get_settings().recipe_mode
    mode = str(recipe_mode).strip().lower()
    if mode not in VALID_RECIPE_MODES:
        return "advisory"
    return mode


def load_recipe(file_path: str | Path) -> AnalysisRecipe:
    """Load a single analysis recipe from a YAML file."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Convert method configs
    primary_methods = []
    for m in data.get('primary_methods', []):
        m['method'] = StatisticalMethod(m['method'])
        primary_methods.append(MethodConfig(**m))
    
    secondary_methods = []
    for m in data.get('secondary_methods', []):
        m['method'] = StatisticalMethod(m['method'])
        secondary_methods.append(MethodConfig(**m))
    
    # Convert data requirements
    data_requirements = [
        DataRequirement(**d) for d in data.get('data_requirements', [])
    ]
    
    return AnalysisRecipe(
        recipe_id=data['recipe_id'],
        failure_type=data['failure_type'],
        version=data.get('version', '1.0.0'),
        name=data['name'],
        description=data['description'],
        required_case_fields=data['required_case_fields'],
        data_requirements=data_requirements,
        primary_methods=primary_methods,
        secondary_methods=secondary_methods,
        common_hypotheses=data.get('common_hypotheses', []),
        relevant_guide_sections=data.get('relevant_guide_sections', []),
        must_include_fields=data.get('must_include_fields'),
        min_effect_size=data.get('min_effect_size', 0.5),
        min_confidence=data.get('min_confidence', 0.7),
    )


def load_all_recipes(recipes_dir: str | Path) -> dict[str, AnalysisRecipe]:
    """Load all recipes from a directory.
    
    Returns:
        Dict mapping failure_type to AnalysisRecipe
    """
    recipes_dir = Path(recipes_dir)
    recipes = {}
    
    for file_path in recipes_dir.glob('*.yaml'):
        if file_path.name.startswith('_'):
            continue  # Skip template files
        try:
            recipe = load_recipe(file_path)
            recipes[recipe.failure_type] = recipe
        except Exception as e:
            print(f"Warning: Failed to load recipe {file_path}: {e}")
    
    return recipes


def get_recipe_for_failure(
    failure_type: str, 
    recipes_dir: Optional[str] = None,
    recipe_mode: Optional[str] = None,
) -> Optional[AnalysisRecipe]:
    """Get the analysis recipe for a specific failure type."""
    mode = normalize_recipe_mode(recipe_mode)
    if mode == "off" or not failure_type:
        return None

    if recipes_dir is None:
        from config.settings import get_settings
        recipes_dir = get_settings().recipes_dir
    
    recipes = load_all_recipes(recipes_dir)
    return recipes.get(failure_type)
