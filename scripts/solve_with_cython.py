import pandas as pd
import numpy as np
import time
import os
import argparse
import sys
import datetime

# プロジェクトルートのパス設定
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# コンパイル済みのCythonモジュールをインポート
from solver_cython.core import (
    solve_knapsack_sa_parallel,
    solve_knapsack_sa_single,
    solve_knapsack_sa_timed,
)
from utils.solution_eval import (
    evaluate_solution,
    format_solution_report,
    parse_constraints,
)


class CythonBenchmarker:
    def __init__(self):
        self.csv_path = os.path.join(PROJECT_ROOT, "data", "problem_data.csv")
        self.constraints_path = os.path.join(PROJECT_ROOT, "data", "constraints.txt")
        self.group_max = 10

    def load_all_data(self):
        """データの読み込みと整形"""
        if not os.path.exists(self.csv_path):
            print(f"\n[!] Error: {self.csv_path} not found.")
            sys.exit(1)
        self.df = pd.read_csv(self.csv_path)
        self.capacities, self.conflicts, self.bonus_thresholds, self.bonus_value = (
            parse_constraints(self.constraints_path)
        )
        self.val_arr = self.df["value"].values.astype(np.int32)
        self.weight_arr = self.df[["weight0", "weight1", "weight2"]].values.astype(
            np.int32
        )
        self.group_arr = self.df["group_id"].values.astype(np.int32)
        self.n_items = len(self.df)
        self.n_groups = int(self.df["group_id"].max() + 1)

    def save_result(
        self,
        solver_name,
        elapsed,
        status,
        evaluation,
        objective_value=None,
        full_output=True,
    ):
        """results/runs/cython_results.txt に結果を追記保存する"""
        result_dir = os.path.join(PROJECT_ROOT, "results", "runs")
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, "cython_results.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_text = format_solution_report(
            solver_name=solver_name,
            elapsed_sec=elapsed,
            status=status,
            evaluation=evaluation,
            objective_value=objective_value,
            timestamp=timestamp,
            full_output=full_output,
        )

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result_text)
        print(result_text)

    def run(
        self,
        iterations=10000000,
        patience=10,
        full_output=True,
        timeout_sec=None,
        single_sa_only=False,
    ):
        self.load_all_data()

        # ---------------------------------------------------------
        # 1. SAソルバー (単体実行)
        # ---------------------------------------------------------
        if timeout_sec is not None:
            print(f"--- 1. Starting Cython Single SA (timeout={timeout_sec}s) ---")
            st1 = time.perf_counter()
            score1, sol1 = solve_knapsack_sa_timed(
                self.val_arr,
                self.weight_arr,
                self.capacities,
                self.group_arr,
                self.conflicts,
                self.n_items,
                self.n_groups,
                self.group_max,
                int(self.bonus_thresholds[0]),
                int(self.bonus_thresholds[1]),
                int(self.bonus_thresholds[2]),
                float(self.bonus_value),
                float(timeout_sec),
                verbose=False,
            )
        else:
            print(f"--- 1. Starting Cython Single SA (iters={iterations}) ---")
            st1 = time.perf_counter()
            score1, sol1 = solve_knapsack_sa_single(
                self.val_arr,
                self.weight_arr,
                self.capacities,
                self.group_arr,
                self.conflicts,
                self.n_items,
                self.n_groups,
                self.group_max,
                int(self.bonus_thresholds[0]),
                int(self.bonus_thresholds[1]),
                int(self.bonus_thresholds[2]),
                float(self.bonus_value),
                iterations,
            )
        el1 = time.perf_counter() - st1
        eval1 = evaluate_solution(
            sol1,
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.group_max,
            bonus_val=self.bonus_value,
            bonus_thresholds=self.bonus_thresholds,
        )
        status1 = "SATISFIED" if eval1["is_valid"] else "INFEASIBLE"
        self.save_result(
            "cython_single_sa",
            el1,
            status1,
            eval1,
            objective_value=int(score1),
            full_output=full_output,
        )

        if single_sa_only:
            print("--- single-sa-only enabled: skipping Hybrid GA-SA ---")
            return

        # ---------------------------------------------------------
        # 2. 並列進化計算 (Hybrid GA-SA)
        # ---------------------------------------------------------
        if timeout_sec is not None:
            print(
                f"--- 2. Starting Hybrid GA-SA (Parallel, timeout={timeout_sec}s) ---"
            )
        else:
            print(f"--- 2. Starting Hybrid GA-SA (Parallel, Patience={patience}) ---")
        st2 = time.perf_counter()

        score2, sol2 = solve_knapsack_sa_parallel(
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.n_items,
            self.n_groups,
            self.group_max,
            int(self.bonus_thresholds[0]),
            int(self.bonus_thresholds[1]),
            int(self.bonus_thresholds[2]),
            float(self.bonus_value),
            pop_size=20,
            rand_add_size=20,
            crossover_size=50,
            max_generations=2_000_000_000 if timeout_sec is not None else 1000,
            iter_per_ind=1000000,
            patience=999 if timeout_sec is not None else patience,
            timeout_sec=float(timeout_sec) if timeout_sec is not None else 0.0,
            verbose=False if timeout_sec is not None else True,
        )
        el2 = time.perf_counter() - st2
        eval2 = evaluate_solution(
            sol2,
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.group_max,
            bonus_val=self.bonus_value,
            bonus_thresholds=self.bonus_thresholds,
        )
        status2 = "SATISFIED" if eval2["is_valid"] else "INFEASIBLE"
        self.save_result(
            "cython_sa_parallel_evolution",
            el2,
            status2,
            eval2,
            objective_value=int(score2),
            full_output=full_output,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="SAのイテレーション数（--timeout と同時指定時は --timeout が優先）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="実行時間の目安（秒）。指定するとイテレーション数を自動推定する",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--full-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Output full selected item/group lists (default: true). Use --no-full-output for preview mode.",
    )
    parser.add_argument(
        "--single-sa-only",
        action="store_true",
        help="Run only single SA and skip hybrid evolution.",
    )
    args = parser.parse_args()

    # --timeout 未指定 かつ --iter 未指定の場合はデフォルト 10000000
    iterations = args.iter if args.iter is not None else 10000000

    bench = CythonBenchmarker()
    bench.run(
        iterations=iterations,
        patience=args.patience,
        full_output=args.full_output,
        timeout_sec=args.timeout,
        single_sa_only=args.single_sa_only,
    )
