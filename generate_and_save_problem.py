import numpy as np
import pandas as pd


def generate_and_save_problem(
    n_items=2000, n_groups=200, n_conflicts=40, filename="problem_data.csv"
):
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "weight0": np.random.randint(10, 100, n_items),
            "weight1": np.random.randint(10, 100, n_items),
            "weight2": np.random.randint(10, 100, n_items),
            "value": np.random.randint(10, 200, n_items),
            "group_id": np.random.randint(0, n_groups, n_items),
        }
    )

    df.to_csv(filename, index=False)

    # キャパシティ設定（合計の5%程度に絞ると難易度が上がります）
    capacities = [int(df[f"weight{i}"].sum() * 0.05) for i in range(3)]

    # 排他ペアの生成 (ランダムに40組)
    # 矛盾（A-B, B-C, C-A）を避けるため、一貫性のあるペアを作る
    all_groups = np.arange(n_groups)
    np.random.shuffle(all_groups)

    conflicts = []
    for i in range(0, n_conflicts * 2, 2):
        if i + 1 < n_groups:
            conflicts.append((all_groups[i], all_groups[i + 1]))

    with open("constraints.txt", "w") as f:
        f.write(f"capacities:{','.join(map(str, capacities))}\n")
        f.write(f"group_max:6\n")  # 最大6個まで持てる設定
        f.write(f"bonus_step:3,4,5\n")
        f.write(f"bonus_value:50,50,50\n")  # 累積で50, 100, 150になる設定
        conf_str = ";".join([f"{a},{b}" for a, b in conflicts])
        f.write(f"conflicts:{conf_str}\n")

    print(f"Generated {n_items} items. Capacities: {capacities}")


def save_problem_description(filename="problem_description.txt"):
    description = """
==================================================
【業務シミュレーション】
最適アイテム選定・多次元コスト管理問題 仕様書
==================================================

1. 目的（ゴール）:
   限られた予算とルールの中で、「合計スコア」を最大化する組み合わせを見つける。
   - 合計スコア = 「各アイテムの価値」の合計 + 「セットボーナス」の合計

2. 守らなければならないルール（制約）:
   - 3つの厳しい制限（コスト）: 
     「重さ」「体積」「消費電力」のような3つの異なる基準が設定されており、
     そのすべてが決められた上限（キャパシティ）を超えてはいけない。
   - 同じ属性の重複制限:
     同じカテゴリー（グループ）に属するアイテムは、1つのグループにつき
     最大6個までしか選ぶことができない。
   - カテゴリー間の相性不一致（排他ルール）: 
     「Aグループを1つでも選んだら、Bグループのものは一切選べない」
     という犬猿の仲のペアが40組存在し、これらを同時に選ぶことはできない。

3. 特別加点（セットボーナス）:
   同じカテゴリーから多くのアイテムを選ぶと、ボーナスポイントが入る。
   - 3つ選んだ場合： +50点
   - 4つ選んだ場合： +100点（さらに50点追加）
   - 5つ選んだ場合： +150点（さらに50点追加）
   ※5個以上選んでもボーナスは一律150点（最大個数は6個のため）。

4. この問題の難しさ（技術的な背景）:
   アイテム数が2,000個、組み合わせのパターンは「宇宙の星の数」を遥かに超える。
   さらに「Aを選んだらBがダメ」「3つ以上でボーナス」といった複雑な条件が絡み合うため、
   一般的な計算ソフトでは「迷路」に迷い込み、解答までに膨大な時間がかかる。
==================================================
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(description.strip())


if __name__ == "__main__":
    generate_and_save_problem()
    save_problem_description()
