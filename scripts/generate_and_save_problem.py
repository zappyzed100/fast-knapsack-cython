from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def generate_and_save_problem(
    n_items=2000,
    n_random_conflicts=200,
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

    # 2. グループ間禁止ペア生成（重複なし）
    cliques = []  # MiniZinc互換のため保持（使用しない）
    used_pairs = set()
    all_conflict_pairs = []
    while len(all_conflict_pairs) < n_random_conflicts:
        p1, p2 = np.random.choice(np.arange(n_groups), 2, replace=False)
        pair = (min(int(p1), int(p2)), max(int(p1), int(p2)))
        if pair not in used_pairs:
            used_pairs.add(pair)
            all_conflict_pairs.append(pair)

    # 3. 制約内容を data/constraints.txt へ保存
    with open(constraints_filename, "w", encoding="utf-8") as f:
        f.write(f"capacities:{','.join(map(str, capacities))}\n")
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
ナップサック問題（多制約・ボーナス付き）仕様書
==================================================

【問題のイメージ】
  倉庫に積み込む荷物を2,000種類の候補から選ぶ選択問題です。
  「どの荷物を選べば合計価値が最大になるか」を求めます。
  ただし、容量制限・相性の悪い組み合わせ・まとめ買いボーナスが絡み合い、
  総当たりでは到底解けない難しさがあります。

【数値規模】
  - 選択候補のアイテム数 : 2,000個
  - カテゴリー（グループ）数 : 200種類
  - 容量制限の種類 : 3種（重さ・体積・電力に相当）

【制約の種類と、なぜ難しいか】

  (a) 3次元の容量制限
      重さ・体積・電力の3種類それぞれの合計が上限を超えてはいけない。
      さらに「価値と重さがほぼ同じ値」に設定してあるため、
      「重いものから優先して省く」という単純な戦略が通用しない。

  (b) グループ間禁止ペア（同時選択禁止ルール）
      「グループAとグループBのアイテムは同時に選べない」という禁止ペアが200組ある。
      禁止ペアはランダムに設定され・重複なしであり、
      制約グラフの構造を不規則にすることでソルバーの前処理による簡略化を防いでいる。

  (c) 段階ボーナス（まとめ買い加点）
      同じカテゴリーから 3個・4個・5個 選ぶごとに +50点（最大 +150点）。
      このボーナスを数式で扱うには「何個選んだか」を追跡する補助変数が
      200グループ × 3段階 = 600個 必要になり、汎用ソルバーの探索空間が大きく膨らむ。

==================================================
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(description.strip())


if __name__ == "__main__":
    df, capacities, cliques, conflicts = generate_and_save_problem()
    save_as_dzn(df, capacities, cliques, conflicts)
    save_problem_description()
