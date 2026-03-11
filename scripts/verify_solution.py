import pandas as pd
import numpy as np
import os
import sys

# プロジェクトルートをパスに追加して solver_cython をインポート可能にする
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from solver_cython.core import solve_knapsack_sa_parallel


def load_data():
    """scripts/ から見て ../data/ にあるファイルをロード"""
    # パスの設定
    data_dir = os.path.join(PROJECT_ROOT, "data")
    csv_path = os.path.join(data_dir, "problem_data.csv")
    constraints_path = os.path.join(data_dir, "constraints.txt")

    if not os.path.exists(csv_path) or not os.path.exists(constraints_path):
        print(f"[!] Error: Data files not found at {data_dir}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    val_arr = df["value"].values.astype(np.int32)
    weight_arr = df[["weight0", "weight1", "weight2"]].values.astype(np.int32)
    group_arr = df["group_id"].values.astype(np.int32)

    with open(constraints_path, "r") as f:
        lines = {
            line.split(":")[0]: line.split(":")[1].strip() for line in f if ":" in line
        }

    capacities = np.array(
        list(map(int, lines["capacities"].split(","))), dtype=np.int32
    )
    conf_raw = lines.get("conflicts", "")
    conflicts = (
        np.array(
            [tuple(map(int, c.split(","))) for c in conf_raw.split(";") if c],
            dtype=np.int32,
        )
        if conf_raw
        else np.zeros((0, 2), dtype=np.int32)
    )

    return (
        val_arr,
        weight_arr,
        capacities,
        group_arr,
        conflicts,
        len(df),
        int(df["group_id"].max() + 1),
        10,
    )


def verify_and_score(
    solution, values, weights, capacities, item_groups, conflict_pairs, group_max
):
    """ソルバーに依存しない独立したバリデーションロジック"""
    is_valid = True
    errors = []
    selected_indices = np.where(solution == 1)[0]

    # 1. 重量制約チェック
    total_weights = np.sum(weights[selected_indices], axis=0)
    for i, (tw, cap) in enumerate(zip(total_weights, capacities)):
        if tw > cap:
            is_valid = False
            errors.append(f"Weight Over (Dim {i}): {tw} > {cap}")

    # 2. グループ最大数チェック
    unique_groups, counts = np.unique(item_groups[selected_indices], return_counts=True)
    for gid, count in zip(unique_groups, counts):
        if count > group_max:
            is_valid = False
            errors.append(f"Group Count Over (GID {gid}): {count} > {group_max}")

    # 3. 排他制約チェック (Conflict Pairs)
    selected_groups_set = set(unique_groups)
    for i in range(conflict_pairs.shape[0]):
        g1, g2 = conflict_pairs[i]
        if g1 in selected_groups_set and g2 in selected_groups_set:
            is_valid = False
            errors.append(f"Conflict Violation: Group {g1} and {g2} both selected")

    # 4. スコア再計算 (Base Value + Group Bonus)
    base_score = np.sum(values[selected_indices])
    bonus_score = 0
    for count in counts:
        if 3 <= count <= 5:
            bonus_score += 50  # Bonus Value: 50

    return is_valid, int(base_score + bonus_score), errors


if __name__ == "__main__":
    # データロード
    v, w, c, ig, cp, ni, ng, gm = load_data()

    print("--- Running Solver for Verification ---")
    # ソルバーの実行 (デフォルトパラメータ)
    score_from_solver, solution = solve_knapsack_sa_parallel(
        v, w, c, ig, cp, ni, ng, gm
    )

    # 独立検証
    is_valid, calculated_score, errors = verify_and_score(solution, v, w, c, ig, cp, gm)

    print("\n" + "=" * 50)
    if is_valid:
        print("✅ Validation: VALID")
    else:
        print("❌ Validation: INVALID")
        for err in errors:
            print(f"   - {err}")

    print(f"\nSolver reported score      : {score_from_solver}")
    print(f"Independently calculated   : {calculated_score}")

    if int(score_from_solver) == calculated_score:
        print("✅ Score Match: SUCCESS")
    else:
        print("❌ Score Match: FAILED")
    print("=" * 50)
