import subprocess
import sys
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
        self.assertIn("Solver: cython_single_sa", output)
        self.assertIn("Solver: cython_sa_parallel_evolution", output)
        self.assertIn("Validation: VALID", output)

    def test_numba_solver_outputs_expected_fields(self):
        output = _run_python_script(
            ["scripts/solve_with_numba.py", "--timeout", "5", "--no-full-output"]
        )
        self.assertIn("Solver: numba_single_sa", output)
        self.assertIn("Solver: numba_hybrid_evolution", output)
        self.assertIn("Validation: VALID", output)

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
        self.assertIn("Solver: gecode", output)
        self.assertIn("Status:", output)
        self.assertIn("Validation:", output)


if __name__ == "__main__":
    unittest.main()
