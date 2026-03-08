import pandas as pd
import pulp
import time
import os


def solve_with_pulp(csv_file="problem_data.csv", constraints_file="constraints.txt"):
    # 1. データの読み込み
    if not os.path.exists(csv_file) or not os.path.exists(constraints_file):
        print(
            "Error: データファイルが見つかりません。先に生成スクリプトを実行してください。"
        )
        return

    df = pd.read_csv(csv_file)
    n_items = len(df)
    n_groups = df["group_id"].max() + 1

    with open(constraints_file, "r") as f:
        lines = f.readlines()
        capacities = list(map(int, lines[0].split(":")[1].split(",")))
        group_max = int(lines[1].split(":")[1])
        conflicts_line = lines[-1]
        conflicts_data = conflicts_line.split(":")[1].strip().split(";")
        conflicts = [tuple(map(int, c.split(","))) for c in conflicts_data if c]

    # 2. 問題の定義
    prob = pulp.LpProblem("MultiStep_Bonus_Knapsack", pulp.LpMaximize)

    # 3. 変数定義
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_items)]

    z = {}
    bonus_thresholds = [3, 4, 5]
    for g in range(n_groups):
        for t in bonus_thresholds:
            z[g, t] = pulp.LpVariable(f"z_g{g}_t{t}", cat="Binary")

    # 4. 目的関数の設定
    obj_items = pulp.lpSum([df.loc[i, "value"] * x[i] for i in range(n_items)])
    obj_bonus = pulp.lpSum(
        [50 * z[g, t] for g in range(n_groups) for t in bonus_thresholds]
    )
    prob += obj_items + obj_bonus

    # 5. 制約条件の設定
    # (a) 重み制約
    for j in range(3):
        prob += (
            pulp.lpSum([df.loc[i, f"weight{j}"] * x[i] for i in range(n_items)])
            <= capacities[j]
        )

    # (b) グループ内個数制限
    for g in range(n_groups):
        group_items_idx = df[df["group_id"] == g].index
        prob += pulp.lpSum([x[i] for i in group_items_idx]) <= group_max

    # (c) ボーナス発生条件
    for g in range(n_groups):
        group_items_idx = df[df["group_id"] == g].index
        actual_count = pulp.lpSum([x[i] for i in group_items_idx])
        for t in bonus_thresholds:
            prob += actual_count >= t * z[g, t]

    # (d) 排他制約
    for i, (g1, g2) in enumerate(conflicts):
        y1 = pulp.LpVariable(f"y_conflict_{i}_g{g1}", cat="Binary")
        y2 = pulp.LpVariable(f"y_conflict_{i}_g{g2}", cat="Binary")

        items1 = df[df["group_id"] == g1].index
        items2 = df[df["group_id"] == g2].index

        prob += pulp.lpSum([x[idx] for idx in items1]) <= group_max * y1
        prob += pulp.lpSum([x[idx] for idx in items2]) <= group_max * y2
        prob += y1 + y2 <= 1

    # 6. ソルバーの実行
    print(f"Solving with PuLP (CBC)... Items: {n_items}, Groups: {n_groups}")
    start_time = time.time()
    # timeLimitは秒単位。計算が終わらない場合に備えて10分に設定
    status = prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=600))
    end_time = time.time()
    solve_time = end_time - start_time

    # 7. 結果の集計
    pure_val = 0
    if status != pulp.LpStatusNotSolved:
        pure_val = sum(
            df.loc[i, "value"] for i in range(n_items) if pulp.value(x[i]) > 0.5
        )

    # 8. 結果表示とログ保存
    result_text = f"""
    ==============================
    PuLP Solver Result (Benchmark)
    ==============================
    Result Status: {pulp.LpStatus[status]}
    Total Objective: {pulp.value(prob.objective)}
    Pure Item Value: {pure_val}
    Execution Time: {solve_time:.4f} seconds
    ==============================
    """
    print(result_text)

    # 保存処理
    output_dir = "result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "pulp_results.txt")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(result_text)

    return solve_time


if __name__ == "__main__":
    solve_with_pulp()
