import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.solution_eval import (
    evaluate_solution,
    format_solution_report,
    parse_constraints,
)


class TestParseConstraints(unittest.TestCase):
    def test_parse_constraints_reads_expected_values(self):
        content = "\n".join(
            [
                "capacities:100,200,300",
                "conflicts:1,2;3,4",
                "bonus_thresholds:3,4,5",
                "bonus_value:3",
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "constraints.txt"
            path.write_text(content, encoding="utf-8")

            capacities, conflict_pairs, thresholds, bonus_value = parse_constraints(
                str(path)
            )

        np.testing.assert_array_equal(
            capacities, np.array([100, 200, 300], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            conflict_pairs,
            np.array([[1, 2], [3, 4]], dtype=np.int32),
        )
        np.testing.assert_array_equal(thresholds, np.array([3, 4, 5], dtype=np.int32))
        self.assertEqual(bonus_value, 3)


class TestEvaluateSolution(unittest.TestCase):
    def test_evaluate_solution_bonus_is_applied_for_same_group(self):
        values = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        weights = np.ones((5, 3), dtype=np.int32)
        capacities = np.array([10, 10, 10], dtype=np.int32)
        item_groups = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        conflict_pairs = np.zeros((0, 2), dtype=np.int32)

        # 同一グループで5個選択: 3,4,5個目の到達で +3 +3 +3 = +9
        solution = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        result = evaluate_solution(
            solution=solution,
            values=values,
            weights=weights,
            capacities=capacities,
            item_groups=item_groups,
            conflict_pairs=conflict_pairs,
            group_max=10,
            bonus_val=3,
            bonus_thresholds=np.array([3, 4, 5], dtype=np.int32),
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["base_score"], 150)
        self.assertEqual(result["bonus_score"], 9)
        self.assertEqual(result["total_score"], 159)

    def test_evaluate_solution_detects_conflict_violation(self):
        values = np.array([10, 20], dtype=np.int32)
        weights = np.ones((2, 3), dtype=np.int32)
        capacities = np.array([10, 10, 10], dtype=np.int32)
        item_groups = np.array([1, 2], dtype=np.int32)
        conflict_pairs = np.array([[1, 2]], dtype=np.int32)

        solution = np.array([1, 1], dtype=np.int8)
        result = evaluate_solution(
            solution=solution,
            values=values,
            weights=weights,
            capacities=capacities,
            item_groups=item_groups,
            conflict_pairs=conflict_pairs,
            group_max=10,
            bonus_val=3,
            bonus_thresholds=np.array([3, 4, 5], dtype=np.int32),
        )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["conflict_violation_count"], 1)
        self.assertIn("Conflict violations", " ".join(result["errors"]))


class TestFormatSolutionReport(unittest.TestCase):
    def test_format_solution_report_preview_labels(self):
        evaluation = {
            "is_valid": True,
            "errors": [],
            "selected_count": 3,
            "selected_indices": np.array([0, 1, 2], dtype=np.int32),
            "selected_groups": np.array([0], dtype=np.int32),
            "group_counts": np.array([3], dtype=np.int32),
            "total_weights": np.array([3, 3, 3], dtype=np.int64),
            "capacities": np.array([10, 10, 10], dtype=np.int64),
            "base_score": 60,
            "bonus_score": 3,
            "total_score": 63,
            "conflict_violation_count": 0,
            "conflict_violations": [],
        }

        text = format_solution_report(
            solver_name="numba_single_sa",
            elapsed_sec=1.23,
            status="SATISFIED",
            evaluation=evaluation,
            objective_value=63,
            full_output=False,
        )

        self.assertIn("Selected Items (preview)", text)
        self.assertIn("Selected Groups (preview)", text)
        self.assertIn("Validation: VALID", text)


if __name__ == "__main__":
    unittest.main()
