import subprocess
import sys
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python_script(args):
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        raise AssertionError(
            "Command failed: "
            + " ".join([sys.executable, *args])
            + f"\nReturn code: {proc.returncode}\nOutput:\n{output}"
        )
    return output


def _extract_solver_block(output, solver_name):
    for block in output.split("-" * 50):
        if f"Solver: {solver_name}" in block:
            return block
    raise AssertionError(f"Solver block not found: {solver_name}")


def _extract_int(block, field):
    m = re.search(rf"{re.escape(field)}:\s*(\d+)", block)
    if not m:
        raise AssertionError(f"Missing integer field: {field}\nBlock:\n{block}")
    return int(m.group(1))


def _extract_float(block, field):
    m = re.search(rf"{re.escape(field)}:\s*([\d\.]+)", block)
    if not m:
        raise AssertionError(f"Missing float field: {field}\nBlock:\n{block}")
    return float(m.group(1))


def _assert_basic_boundaries(testcase, block):
    testcase.assertIn("Validation: VALID", block)
    testcase.assertGreater(_extract_int(block, "Selected Item Count"), 0)
    testcase.assertEqual(_extract_int(block, "Conflict Violations"), 0)
    testcase.assertGreater(_extract_int(block, "Recalculated Score"), 0)

    # CIマシン差を吸収するため、上限はやや広めに取る
    elapsed = _extract_float(block, "Execution Time")
    testcase.assertGreater(elapsed, 0.0)
    testcase.assertLess(elapsed, 20.0)


class TestSolversSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 依存ファイルを用意し、Cython拡張を事前ビルドする
        _run_python_script(["scripts/generate_and_save_problem.py"])
        _run_python_script(["src/solver_cython/setup.py", "build_ext", "--inplace"])

    def test_cython_solver_outputs_expected_fields(self):
        output = _run_python_script(
            ["scripts/solve_with_cython.py", "--timeout", "5", "--no-full-output"]
        )
        block_sa = _extract_solver_block(output, "cython_single_sa")
        block_evo = _extract_solver_block(output, "cython_sa_parallel_evolution")
        _assert_basic_boundaries(self, block_sa)
        _assert_basic_boundaries(self, block_evo)

    def test_numba_solver_outputs_expected_fields(self):
        output = _run_python_script(
            ["scripts/solve_with_numba.py", "--timeout", "5", "--no-full-output"]
        )
        block_sa = _extract_solver_block(output, "numba_single_sa")
        block_evo = _extract_solver_block(output, "numba_hybrid_evolution")
        _assert_basic_boundaries(self, block_sa)
        _assert_basic_boundaries(self, block_evo)

    def test_minizinc_solver_outputs_expected_fields(self):
        output = _run_python_script(
            [
                "scripts/solve_with_minizinc_solvers.py",
                "--solvers",
                "gecode",
                "--timeout",
                "5",
                "--no-full-output",
            ]
        )
        block = _extract_solver_block(output, "gecode")
        self.assertIn("Status:", block)
        _assert_basic_boundaries(self, block)


if __name__ == "__main__":
    unittest.main()
