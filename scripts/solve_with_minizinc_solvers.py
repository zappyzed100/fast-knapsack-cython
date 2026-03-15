import os
import datetime
import time
import argparse
import numpy as np
import pandas as pd
from minizinc import Instance, Model, Solver


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in os.sys.path:
    os.sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from utils.solution_eval import (
    evaluate_solution,
    format_solution_report,
    parse_constraints,
)


def run_single_benchmark(solver_id, timeout_sec=100, full_output=True):
    model_path = os.path.join(PROJECT_ROOT, "src", "solver_minizinc", "problem.mzn")
    data_path = os.path.join(PROJECT_ROOT, "data", "data.dzn")
    result_dir = os.path.join(PROJECT_ROOT, "results", "runs")

    clean_solver_id = solver_id.replace("-", "_").replace(".", "_")
    output_path = os.path.join(result_dir, f"{clean_solver_id}_results.txt")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(f"\n>>> Running Benchmark: {solver_id} (Timeout: {timeout_sec}s)")

    try:
        solver = Solver.lookup(solver_id)
        model = Model(model_path)
        model.add_file(data_path)
        instance = Instance(solver, model)

        start_time = time.perf_counter()
        # 決定変数 x の出力を確実にするため、solve() を実行
        result = instance.solve(timeout=datetime.timedelta(seconds=timeout_sec))
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        status = str(result.status)

        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "problem_data.csv"))
        capacities, conflicts, bonus_thresholds, bonus_value = parse_constraints(
            os.path.join(PROJECT_ROOT, "data", "constraints.txt")
        )
        values = df["value"].values.astype(np.int32)
        weights = df[["weight0", "weight1", "weight2"]].values.astype(np.int32)
        groups = df["group_id"].values.astype(np.int32)

        if result.solution is not None and hasattr(result.solution, "x"):
            sol = np.array(result.solution.x, dtype=np.int8)
            evaluation = evaluate_solution(
                sol,
                values,
                weights,
                capacities,
                groups,
                conflicts,
                group_max=10,
                bonus_val=bonus_value,
                bonus_thresholds=bonus_thresholds,
            )
        else:
            evaluation = {
                "is_valid": False,
                "errors": ["No solution produced"],
                "selected_count": 0,
                "selected_indices": np.zeros(0, dtype=np.int32),
                "selected_groups": np.zeros(0, dtype=np.int32),
                "group_counts": np.zeros(0, dtype=np.int32),
                "total_weights": np.zeros(3, dtype=np.int64),
                "capacities": capacities.astype(np.int64),
                "base_score": 0,
                "bonus_score": 0,
                "total_score": 0,
                "conflict_violation_count": 0,
                "conflict_violations": [],
            }

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_text = format_solution_report(
            solver_name=solver_id,
            elapsed_sec=elapsed_time,
            status=status,
            evaluation=evaluation,
            objective_value=result.objective,
            timestamp=timestamp,
            full_output=full_output,
        )
        print(result_text)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result_text)

        return True

    except Exception as e:
        print(f"Failed to run solver {solver_id}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument(
        "--solvers",
        type=str,
        default="cp-sat,gecode,cbc",
        help="Comma-separated solvers (e.g. gecode,cbc)",
    )
    parser.add_argument(
        "--full-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Output full selected item/group lists (default: true). Use --no-full-output for preview mode.",
    )
    args = parser.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    if not solvers:
        raise SystemExit("No solvers specified. Use --solvers.")

    any_failed = False
    for solver_name in solvers:
        ok = run_single_benchmark(
            solver_name,
            timeout_sec=args.timeout,
            full_output=args.full_output,
        )
        if not ok:
            any_failed = True

    if any_failed:
        raise SystemExit(1)
