from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def generate_and_save_problem(
    n_items=2000,
    n_cliques=40,
    clique_size=50,
    n_random_conflicts=80,
    n_groups=200,
    filename=None,
    constraints_filename=None,
):
    np.random.seed(42)

    if filename is None:
        filename = DATA_DIR / "problem_data.csv"
    if constraints_filename is None:
        constraints_filename = DATA_DIR / "constraints.txt"

    filename = Path(filename)
    constraints_filename = Path(constraints_filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    constraints_filename.parent.mkdir(parents=True, exist_ok=True)

    # 1. アイテム生成
    w0 = np.random.randint(50, 150, n_items)
    w1 = np.random.randint(50, 150, n_items)
    w2 = np.random.randint(50, 150, n_items)
    base_val = (w0 + w1 + w2) // 3
    value = base_val + np.random.randint(-2, 3, n_items)

    group_ids = np.repeat(np.arange(n_groups), n_items // n_groups)

    df = pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "group_id": group_ids,
            "weight0": w0,
            "weight1": w1,
            "weight2": w2,
            "value": value,
        }
    )
    df.to_csv(filename, index=False)

    # キャパシティ設定
    capacities = [int(df[f"weight{i}"].sum() * 0.02) for i in range(3)]

    # 2. 高密度クリン生成
    cliques = []
    for _ in range(n_cliques):
        members = np.random.choice(np.arange(n_items), clique_size, replace=False)
        cliques.append(",".join(map(str, members)))

    # 3. 排他ペア生成
    all_conflict_pairs = []
    for i in range(200):
        all_conflict_pairs.append((i, (i + 1) % 200))
    for _ in range(n_random_conflicts):
        p1, p2 = np.random.choice(np.arange(n_items), 2, replace=False)
        all_conflict_pairs.append((int(p1), int(p2)))

    # 4. 制約内容を data/constraints.txt へ保存
    with open(constraints_filename, "w", encoding="utf-8") as f:
        f.write(f"capacities:{','.join(map(str, capacities))}\n")
        f.write(f"cliques:{';'.join(cliques)}\n")
        conf_str = ";".join([f"{p[0]},{p[1]}" for p in all_conflict_pairs])
        f.write(f"conflicts:{conf_str}\n")
        f.write("bonus_thresholds:3,4,5\n")  # バリデーター用
        f.write("bonus_value:50\n")

    print(f"Generated {n_items} items with {n_groups} groups.")
    return df, capacities, cliques, all_conflict_pairs


def save_as_dzn(df, capacities, cliques, conflicts, filename=None):
    if filename is None:
        filename = DATA_DIR / "data.dzn"

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    n_items = len(df)
    n_groups = df["group_id"].max() + 1

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"n_items = {n_items};\n")

        # array1d(0..n_items-1, [...]) 形式に修正してインデックスを強制指定
        f.write(
            f"weight0 = array1d(0..{n_items-1}, [{', '.join(map(str, df['weight0']))}]);\n"
        )
        f.write(
            f"weight1 = array1d(0..{n_items-1}, [{', '.join(map(str, df['weight1']))}]);\n"
        )
        f.write(
            f"weight2 = array1d(0..{n_items-1}, [{', '.join(map(str, df['weight2']))}]);\n"
        )
        f.write(
            f"value = array1d(0..{n_items-1}, [{', '.join(map(str, df['value']))}]);\n"
        )
        f.write(
            f"group_id = array1d(0..{n_items-1}, [{', '.join(map(str, df['group_id']))}]);\n"
        )

        f.write(f"n_groups = {n_groups};\n")
        f.write("bonus_val = 50;\n")

        # キャパシティ
        f.write(f"cap0 = {capacities[0]};\n")
        f.write(f"cap1 = {capacities[1]};\n")
        f.write(f"cap2 = {capacities[2]};\n")

        # 制約定義
        f.write(f"n_cliques = {len(cliques)};\n")
        clique_str = ", ".join([f"{{{c}}}" for c in cliques])
        f.write(f"clique_members = [{clique_str}];\n")

        f.write(f"n_conflicts = {len(conflicts)};\n")
        f.write(
            f"conflict_pairs = array2d(1..{len(conflicts)}, 1..2, [{', '.join([f'{p[0]}, {p[1]}' for p in conflicts])}]);\n"
        )

    print(f"MiniZinc data saved to {filename}")


def save_problem_description(filename=None):
    if filename is None:
        filename = DATA_DIR / "problem_description.txt"

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    description = """
==================================================
【ベンチマーク専用】
高度過密制約・多段階ボーナス付きナップサック問題 仕様書
==================================================

1. 目的（ゴール）:
   3次元リソース制限を遵守し、合計価値（基本価値＋ボーナス）を最大化する。
   Value ≈ Weight の設定により、LP緩和による効率的な枝刈りを困難にしている。

2. 複雑な制約構造:

   (a) 高密度クリン制約 (High-Density Overlapping Cliques):
       - アイテムが複数の排他集合に重複所属。
       - 汎用ソルバーの制約伝搬コストを最大化する設計。

   (b) ハイブリッド排他制約 (Cycle & Random Conflicts):
       - 200要素の長周期循環排他に、ランダムなアイテム間排他を40ペア追加。
       - 制約グラフの疎密を不均一にし、プレソルブ（前処理）による簡略化を封じている。

   (c) 非線形多段階ボーナス (Multi-stage Bonus):
       - 同一グループから 3, 4, 5個選択するごとに順次 +50点（最大150点）。
       - MILP(CBC等)においては、判定用の補助バイナリ変数を600個(200g * 3)追加する必要があり、
         探索空間の次元を意図的に押し上げている。

3. 自作アルゴリズム（Cython/Numba）の優位性:

   - ビット並列判定:
     アイテム単位の複雑な排他判定を、定数時間のビット演算 (AND/OR) で処理。
   - インクリメンタル・アップデート:
     ボーナス計算を「全走査」ではなく「差分加算」で行うことで、
     汎用ソルバーが数百の制約式で表現するロジックを O(1) で処理。
==================================================
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(description.strip())


if __name__ == "__main__":
    df, capacities, cliques, conflicts = generate_and_save_problem()
    save_as_dzn(df, capacities, cliques, conflicts)
    save_problem_description()
