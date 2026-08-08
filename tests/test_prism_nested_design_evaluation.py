"""Tests for participant-nested PRISM-Gait design evaluation."""

from __future__ import annotations

import unittest

import pandas as pd

from ablation.prism_nested_design_evaluation import select_best


class NestedDesignEvaluationTests(unittest.TestCase):
    def test_duration_tie_prefers_shorter_window(self) -> None:
        values = pd.Series({1.0: 0.8, 2.0: 0.9, 3.0: 0.9})
        self.assertEqual(select_best(values), 2.0)

    def test_architecture_tie_prefers_lower_dimension(self) -> None:
        values = pd.Series({"full": 0.9, "compact": 0.9, "core": 0.8})
        dimensions = {"full": 13_049, "compact": 8_441, "core": 7_812}
        self.assertEqual(select_best(values, dimensions), "compact")

    def test_unique_maximum_is_selected(self) -> None:
        values = pd.Series({"full": 0.92, "compact": 0.90})
        dimensions = {"full": 13_049, "compact": 8_441}
        self.assertEqual(select_best(values, dimensions), "full")


if __name__ == "__main__":
    unittest.main()
