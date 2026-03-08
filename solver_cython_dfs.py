import pandas as pd
import numpy as np
import time
import os
from solver_cython.core import solve_knapsack_exact


def load_constraints(constraints_file):
    """制約条件ファイルを読み込む"""
    if not os.path.exists(constraints_file):
        raise FileNotFoundError(f"制約ファイルが見つかりません: {constraints_file}")

    with open(constraints_file, "r") as f:
        lines = f.readlines()
        capacities = list(map(int, lines[0].split(":")[1].split(",")))
        group_max = int(lines[1].split(":")[1])
        conf_line = [l for l in lines if l.startswith("conflicts:")][0]
        conf_data = conf_line.split(":")[1].strip().split(";")
        conflicts = [tuple(map(int, c.split(","))) for c in conf_data if c]

    return capacities, group_max, conflicts


def main():
    # 1. データの読み込み
    try:
        df = pd.read_csv("problem_data.csv")
        capacities, group_max, conflicts = load_constraints("constraints.txt")
    except Exception as e:
        print(f"エラー: データの読み込みに失敗しました。{e}")
        return

    # 2. Cython関数用のデータ整形
    # Windows環境のBuffer dtype mismatch回避のため、適切に型を指定
    val_arr = df["value"].values.astype(np.int32)
    weight_arr = df[["weight0", "weight1", "weight2"]].values.astype(np.int32)
    group_arr = df["group_id"].values.astype(np.int32)
    conf_arr = np.array(conflicts, dtype=np.int32)
    cap_arr = np.array(capacities, dtype=np.int32)

    n_items = len(df)
    n_groups = df["group_id"].max() + 1

    # 3. 厳密解ソルバーの実行
    print("==================================================")
    print("Starting Cython Exact Solver (DFS with Pruning)")
    print(f"Items: {n_items}, Groups: {n_groups}")
    print("Time Limit: 360 seconds (6 minutes)")
    print("==================================================")
    print("Searching for the optimal solution...")

    start_time = time.time()

    # 探索の実行
    # 注: core.pyx側で remain_max の定義を long long[:] に修正しておく必要があります
    score, best_sol, timeout = solve_knapsack_exact(
        val_arr, weight_arr, cap_arr, group_arr, conf_arr, n_items, n_groups, group_max
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # 4. 結果の表示
    status = (
        "TIMEOUT (Partial Solution)"
        if timeout
        else "OPTIMAL FOUND (Full Search Completed)"
    )

    print("\n" + "=" * 50)
    print(f"Status: {status}")
    print(f"Best Score Found: {score}")
    print(f"Total Time: {elapsed:.4f} seconds")
    print("=" * 50)

    # 5. 比較用ログの保存
    result_text = (
        f"--- Exact Solver Report ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
        f"Status: {status}\n"
        f"Score: {score}\n"
        f"Elapsed: {elapsed:.4f}s\n"
        "--------------------------------------------------\n"
    )

    with open("cython_dfs_results.txt", "a", encoding="utf-8") as f:
        f.write(result_text)

    print(f"Results appended to 'cython_dfs_results.txt'")


if __name__ == "__main__":
    main()
