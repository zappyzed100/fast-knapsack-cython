import pandas as pd
import numpy as np
import time
import os
import argparse
import sys
import datetime

# コンパイル済みのCythonモジュールをインポート
from solver_cython.core import solve_knapsack_sa


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
    """
    最終解が全制約を遵守しているか、元のデータから独立して検証する
    """
    is_valid = True
    selected_indices = np.where(solution == 1)[0]

    # 1. 重量制約のチェック
    total_weights = np.zeros(weights.shape[1], dtype=np.int64)
    for idx in selected_indices:
        total_weights += weights[idx]

    for tw, cap in zip(total_weights, capacities):
        if tw > cap:
            is_valid = False

    # 2. グループ最大数制約のチェック
    unique_groups, counts = np.unique(item_groups[selected_indices], return_counts=True)
    for count in counts:
        if count > group_max:
            is_valid = False

    # 3. 排他制約 (Conflicts) のチェック
    selected_groups_set = set(unique_groups)
    for i in range(conflict_pairs.shape[0]):
        g1, g2 = conflict_pairs[i]
        if g1 in selected_groups_set and g2 in selected_groups_set:
            is_valid = False

    # 4. スコアの再計算 (ベース価値 + ボーナス)
    base_score = np.sum(values[selected_indices])
    bonus_score = 0
    for count in counts:
        if 3 <= count <= 5:
            bonus_score += bonus_val

    return is_valid, int(base_score + bonus_score)


class CythonBenchmarker:
    def __init__(self, csv_path="problem_data.csv", constraints_path="constraints.txt"):
        self.csv_path = csv_path
        self.constraints_path = constraints_path
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
        if conf_raw:
            self.conflicts = np.array(
                [tuple(map(int, c.split(","))) for c in conf_raw.split(";") if c],
                dtype=np.int32,
            )
        else:
            self.conflicts = np.zeros((0, 2), dtype=np.int32)

        self.val_arr = self.df["value"].values.astype(np.int32)
        self.weight_arr = self.df[["weight0", "weight1", "weight2"]].values.astype(
            np.int32
        )
        self.group_arr = self.df["group_id"].values.astype(np.int32)
        self.n_items = len(self.df)
        self.n_groups = int(self.df["group_id"].max() + 1)

    def run(self, iterations=10000000):
        self.load_all_data()

        start_time = time.perf_counter()

        # Cython関数の呼び出し
        score, solution = solve_knapsack_sa(
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.n_items,
            self.n_groups,
            self.group_max,
            iterations,
        )

        elapsed = time.perf_counter() - start_time

        # バリデーション実行
        is_valid, v_score = validate_solution(
            solution,
            self.val_arr,
            self.weight_arr,
            self.capacities,
            self.group_arr,
            self.conflicts,
            self.group_max,
        )

        # 結果の保存処理
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        output_path = os.path.join(result_dir, "cython_results.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_text = f"""[{timestamp}]
Solver: cython
Status: SATISFIED
Validation: {'VALID' if is_valid else 'INVALID'}
Objective Value (Score): {v_score}
Execution Time: {elapsed:.4f} seconds
--------------------------------------------------
"""
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result_text)

        print(result_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=10000000)
    args = parser.parse_args()

    bench = CythonBenchmarker()
    bench.run(iterations=args.iter)
