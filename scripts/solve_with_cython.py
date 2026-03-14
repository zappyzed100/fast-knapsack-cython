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
    solve_knapsack_sa,
)


def validate_solution(
    solution,
    values,
    weights,
    capacities,
    item_groups,
    conflict_pairs,
    group_max,
    bonus_val=50,
):
    """最終解が全制約を遵守しているか検証する"""
    is_valid = True
    selected_indices = np.where(solution == 1)[0]
    if len(selected_indices) == 0:
        return False, 0

    # 1. 重量制約
    total_weights = np.sum(weights[selected_indices], axis=0)
    if any(total_weights > capacities):
        is_valid = False

    # 2. グループ最大数制約
    unique_groups, counts = np.unique(item_groups[selected_indices], return_counts=True)
    if any(counts > group_max):
        is_valid = False

    # 3. 排他制約
    selected_groups_set = set(unique_groups)
    for i in range(conflict_pairs.shape[0]):
        g1, g2 = conflict_pairs[i]
        if g1 in selected_groups_set and g2 in selected_groups_set:
            is_valid = False

    # 4. スコア計算
    base_score = np.sum(values[selected_indices])
    bonus_score = np.sum([bonus_val for c in counts if 3 <= c <= 5])

    return is_valid, int(base_score + bonus_score)


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
        with open(self.constraints_path, "r") as f:
            lines = {
                line.split(":")[0]: line.split(":")[1].strip()
                for line in f
                if ":" in line
            }
        self.capacities = np.array(
            list(map(int, lines["capacities"].split(","))), dtype=np.int32
        )
        conf_raw = lines.get("conflicts", "")
        self.conflicts = (
            np.array(
                [tuple(map(int, c.split(","))) for c in conf_raw.split(";") if c],
                dtype=np.int32,
            )
            if conf_raw
            else np.zeros((0, 2), dtype=np.int32)
        )
        self.val_arr = self.df["value"].values.astype(np.int32)
        self.weight_arr = self.df[["weight0", "weight1", "weight2"]].values.astype(
            np.int32
        )
        self.group_arr = self.df["group_id"].values.astype(np.int32)
        self.n_items = len(self.df)
        self.n_groups = int(self.df["group_id"].max() + 1)

    def save_result(self, solver_name, score, elapsed, is_valid):
        """results/runs/cython_results.txt に結果を追記保存する"""
        result_dir = os.path.join(PROJECT_ROOT, "results", "runs")
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, "cython_results.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_text = (
            f"[{timestamp}]\n"
            f"Solver: {solver_name}\n"
            f"Status: {'SATISFIED' if is_valid else 'INFEASIBLE'}\n"
            f"Validation: {'VALID' if is_valid else 'INVALID'}\n"
            f"Objective Value (Score): {score}\n"
            f"Execution Time: {elapsed:.4f} seconds\n" + "-" * 50 + "\n"
        )

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result_text)
        print(result_text)

    def run(self, iterations=10000000, patience=10):
        self.load_all_data()

        # ---------------------------------------------------------
        # 1. 修正版 SAソルバー (単体実行)
        # ---------------------------------------------------------
        print(f"--- 1. Starting Cython Single SA (iters={iterations}) ---")
        # core.pyx の solve_knapsack_sa は乱数配列を引数に取る [cite: 48]
        rand_add = np.random.randint(0, self.n_items, size=iterations).astype(np.int32)
        rand_rem = np.random.randint(0, self.n_items, size=iterations).astype(np.int32)
        rand_flt = np.random.random(size=iterations).astype(np.float64)

        st1 = time.perf_counter()
        score1, sol1 = solve_knapsack_sa(
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.n_items,
            self.n_groups,
            self.group_max,
            iterations,
            rand_add,
            rand_rem,
            rand_flt,
        )
        el1 = time.perf_counter() - st1
        valid1, v_score1 = validate_solution(
            sol1,
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.group_max,
        )
        self.save_result("cython_single_sa", v_score1, el1, valid1)

        # ---------------------------------------------------------
        # 2. 並列進化計算 (Hybrid GA-SA)
        # ---------------------------------------------------------
        print(f"--- 2. Starting Hybrid GA-SA (Parallel, Patience={patience}) ---")
        st2 = time.perf_counter()

        # core.pyx の solve_knapsack_sa_parallel を利用 [cite: 41]
        score2, sol2 = solve_knapsack_sa_parallel(
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.n_items,
            self.n_groups,
            self.group_max,
            pop_size=20,
            rand_add_size=20,
            crossover_size=50,
            max_generations=1000,
            iter_per_ind=1000000,  # 各個体のSAイテレーション
            patience=patience,
        )
        el2 = time.perf_counter() - st2
        valid2, v_score2 = validate_solution(
            sol2,
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.group_max,
        )
        self.save_result("cython_sa_parallel_evolution", v_score2, el2, valid2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=10000000)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    bench = CythonBenchmarker()
    bench.run(iterations=args.iter, patience=args.patience)
