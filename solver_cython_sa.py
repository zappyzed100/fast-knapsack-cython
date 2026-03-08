import pandas as pd
import numpy as np
import time
import os
from solver_cython.core import solve_knapsack_cython


def load_constraints(constraints_file):
    """制約条件ファイルを読み込む"""
    if not os.path.exists(constraints_file):
        raise FileNotFoundError(f"制約ファイルが見つかりません: {constraints_file}")

    with open(constraints_file, "r") as f:
        lines = f.readlines()
        # 容量の取得
        capacities = list(map(int, lines[0].split(":")[1].split(",")))
        # グループ最大個数の取得
        group_max = int(lines[1].split(":")[1])
        # 排他制約の取得
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
    val_arr = df["value"].values.astype(np.int32)
    weight_arr = df[["weight0", "weight1", "weight2"]].values.astype(np.int32)
    group_arr = df["group_id"].values.astype(np.int32)
    conf_arr = np.array(conflicts, dtype=np.int32)
    cap_arr = np.array(capacities, dtype=np.int32)

    n_items = len(df)
    n_groups = df["group_id"].max() + 1

    # 3. 近似解（焼きなまし法）の実行
    print("==================================================")
    print("Starting Cython Approximate Solver (Hill Climbing/SA)")
    print(f"Items: {n_items}, Groups: {n_groups}, Iterations: 5,000,000")
    print("==================================================")

    start_time = time.time()

    # 探索の実行
    score, best_sol = solve_knapsack_cython(
        val_arr,
        weight_arr,
        cap_arr,
        group_arr,
        conf_arr,
        n_items,
        n_groups,
        group_max,
        max_iter=5000000,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # 4. 結果の出力
    print(f"Status: Completed")
    print(f"Execution Time: {elapsed:.4f} seconds")
    print(f"Best Score Found: {score}")
    print("==================================================")

    # フォーマットに合わせた出力
    result_output = f"""
    ==================================================
    Cython Local Search (Hill Climbing) Result
    ==================================================
    Status: Completed (Max Iterations reached)
    Total Objective: {score}
    Execution Time: {elapsed:.4f} seconds
    Method: Cython Optimized Local Search
    ==================================================
    """
    print(result_output)

    # 結果をファイルに保存
    with open("cython_hill_climbing_results.txt", "a", encoding="utf-8") as f:
        f.write(result_output)

    print(f"Results saved to 'cython_hill_climbing_results.txt'")


if __name__ == "__main__":
    main()
