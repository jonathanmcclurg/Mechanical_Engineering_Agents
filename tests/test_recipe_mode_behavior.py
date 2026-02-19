"""Tests for recipe mode behavior in autonomy-first workflows."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import settings as settings_module
from config.analysis_recipes.loader import get_recipe_for_failure
from src.agents.intake_agent import IntakeTriageAgent


def _reset_settings() -> None:
    settings_module._settings = None


class RecipeModeBehaviorTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_settings()

    def test_recipe_loader_off_mode_returns_none(self):
        with patch.dict(os.environ, {"RECIPE_MODE": "off"}, clear=False):
            _reset_settings()
            recipe = get_recipe_for_failure("leak_test_fail")

        self.assertIsNone(recipe)

    def test_recipe_loader_strict_mode_loads_recipe(self):
        with patch.dict(os.environ, {"RECIPE_MODE": "strict"}, clear=False):
            _reset_settings()
            recipe = get_recipe_for_failure("leak_test_fail")

        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.name, "Helium Leak Test Failure Analysis")

    def test_intake_advisory_marks_recipe_fields_as_recommended(self):
        with patch.dict(os.environ, {"RECIPE_MODE": "advisory"}, clear=False):
            _reset_settings()
            agent = IntakeTriageAgent(llm=None, verbose=False)
            output = agent.execute(
                {
                    "raw_case": {
                        "case_id": "CASE-TEST-001",
                        "failure_type": "leak_test_fail",
                        "failure_description": "Leak test failed during final validation with visible drift.",
                        "part_number": "HYD-VALVE-200",
                    }
                }
            )

        self.assertTrue(output.success)
        self.assertTrue(output.data.get("recipe_applied"))
        self.assertEqual(output.data.get("recipe_mode"), "advisory")

        missing_fields = output.data.get("missing_fields", [])
        lot_field = next((m for m in missing_fields if m.get("field") == "lot_number"), None)
        self.assertIsNotNone(lot_field)
        self.assertEqual(lot_field.get("importance"), "medium")
        self.assertIn("Recommended by analysis recipe", lot_field.get("reason", ""))

    def test_intake_off_mode_uses_baseline_only(self):
        with patch.dict(os.environ, {"RECIPE_MODE": "off"}, clear=False):
            _reset_settings()
            agent = IntakeTriageAgent(llm=None, verbose=False)
            output = agent.execute(
                {
                    "raw_case": {
                        "case_id": "CASE-TEST-002",
                        "failure_type": "leak_test_fail",
                        "failure_description": "Leak test failed during final validation with visible drift.",
                        "part_number": "HYD-VALVE-200",
                    }
                }
            )

        self.assertTrue(output.success)
        self.assertFalse(output.data.get("recipe_applied"))
        self.assertEqual(output.data.get("recipe_mode"), "off")
        self.assertFalse(
            any("analysis recipe" in m.get("reason", "") for m in output.data.get("missing_fields", []))
        )


if __name__ == "__main__":
    unittest.main()
