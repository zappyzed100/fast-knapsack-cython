import os
import datetime
import time
import numpy as np
import pandas as pd
from minizinc import Instance, Model, Solver


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def validate_solution_from_minizinc(
    result,
    csv_path=os.path.join(PROJECT_ROOT, "data", "problem_data.csv"),
    constraints_path=os.path.join(PROJECT_ROOT, "data", "constraints.txt"),
):
    """
    MiniZincの実行結果が制約を満たしているか独立して検証する
    """
    if result.solution is None:
        return False, "No solution to validate."

    # --- データのロード ---
    df = pd.read_csv(csv_path)
    values = df["value"].values
    weights = df[["weight0", "weight1", "weight2"]].values
    item_groups = df["group_id"].values

    with open(constraints_path, "r") as f:
        lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
    capacities = np.array(list(map(int, lines["capacities"].split(","))))
    conflict_pairs = [
        tuple(map(int, c.split(",")))
        for c in lines.get("conflicts", "").split(";")
        if c
    ]
    group_max = 10
    bonus_val = 50

    # MiniZincのモデルで定義されている変数名（例: x）に合わせて取得
    # 配列形式で出力されていることを想定
    sol = np.array(result.solution.x, dtype=np.int8)
    selected_indices = np.where(sol == 1)[0]

    is_valid = True
    error_msg = []

    # 1. 重量制約
    total_weights = np.sum(weights[selected_indices], axis=0)
    for i, (tw, cap) in enumerate(zip(total_weights, capacities)):
        if tw > cap:
            is_valid = False
            error_msg.append(f"Capacity {i} Over: {tw} > {cap}")

    # 2. グループ最大数制約
    u_groups, counts = np.unique(item_groups[selected_indices], return_counts=True)
    for g, count in zip(u_groups, counts):
        if count > group_max:
            is_valid = False
            error_msg.append(f"Group {g} Count Over: {count} > {group_max}")

    # --- 3. 排他制約 ---
    # conflict_pairs はアイテムIDのペア (0..1999) を含んでいる
    for idx1, idx2 in conflict_pairs:
        # MiniZincの解配列 sol において、衝突する両方のアイテムが選ばれているか確認
        if sol[idx1] == 1 and sol[idx2] == 1:
            is_valid = False
            # アイテム単位の衝突としてメッセージを出す
            error_msg.append(f"Conflict: Item {idx1} and {idx2} both selected")

    # 4. スコア再計算 (ボーナス: 3~5個選択時)
    base_score = np.sum(values[selected_indices])
    bonus_score = np.sum([bonus_val for c in counts if 3 <= c <= 5])
    total_score = base_score + bonus_score

    # ソルバーの目的関数値との照合
    if result.objective is not None and abs(total_score - result.objective) > 1e-5:
        error_msg.append(
            f"Score Mismatch: Solver={result.objective}, Validated={total_score}"
        )

    status_str = "VALID" if is_valid else "INVALID: " + " | ".join(error_msg)
    print(f"[*] Validation Result: {status_str}")
    print(f"[*] Re-calculated Score: {total_score}")

    return is_valid, status_str


def run_single_benchmark(solver_id, timeout_sec=100):
    model_path = os.path.join(PROJECT_ROOT, "src", "solver_minizinc", "problem.mzn")
    data_path = os.path.join(PROJECT_ROOT, "data", "data.dzn")
    result_dir = os.path.join(PROJECT_ROOT, "results")

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

        # --- 出力チェックの実行 ---
        is_valid, validation_msg = validate_solution_from_minizinc(result)

        objective = (
            result.objective if result.objective is not None else "No Solution Found"
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_text = f"""[{timestamp}]
Solver: {solver_id}
Status: {status}
Validation: {validation_msg}
Objective Value (Score): {objective}
Execution Time: {elapsed_time:.4f} seconds
--------------------------------------------------
"""
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
    solvers = ["cp-sat", "gecode", "cbc"]
    for solver_name in solvers:
        run_single_benchmark(solver_name)
