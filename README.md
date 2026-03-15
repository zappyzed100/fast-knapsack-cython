# fast-knapsack-cython
## 実務制約付きナップサック問題に対する時間予算ベースの解法比較

このリポジトリは、複数制約を持つ大規模ナップサック問題に対して、以下の3系統の解法を同じ時間予算で比較するための実験プロジェクトです。

- Cython による AOT コンパイル済みヒューリスティック
- Numba による JIT コンパイル済みヒューリスティック
- MiniZinc 経由で実行する汎用ソルバー群（Cbc / CP-SAT / Gecode）

現在の主眼は「最適性証明」ではなく、時間制約のある状況でどこまで良い実行可能解に到達できるかを、統計的に比較できる形で示すことです。

## 基本方針

このプロジェクトでは、アルゴリズムを次の2種類に分けて扱っています。

- 決定論的ソルバー: Cbc / CP-SAT / Gecode
- 確率的ソルバー: Cython / Numba の SA・進化計算

そのため評価方針も分けています。

- 決定論的ソルバーは同一条件なら基本的に同じ結果を返すため、各時間予算で 1 系列として扱う
- 確率的ソルバーはばらつきがあるため、複数回実行した上で平均・標準偏差を見る

現在のレポートでは、時間窓ごとのデータからソルバーごとに最新 20 件までを集計し、以下を出力します。

- 目的関数値の平均
- 目的関数値の標準偏差
- 試行数
- 実行時間の平均

## 対象問題

ベンチマーク対象は、単純な 0/1 ナップサックではなく、実務上の制約を模した複合制約付き問題です。

- アイテム数: 2,000
- グループ数: 200
- 3次元の容量制約
- 排他制約
- グループごとの上限制約
- グループ選択数に応じたボーナス

探索空間は非常に大きく、単純な全探索は不可能です。加えて、排他制約とグループ制約が入ることで、汎用ソルバー・ヒューリスティックの双方にとって探索が難しくなります。

## 現在の比較設定

時間予算は次を採用しています。

- 0.03秒: Cython / Numba の単体 SA のみ
- 1秒
- 10秒
- 30秒
- 60秒

0.03 秒は「極短時間で初期解をどこまで返せるか」を見るための枠です。1 秒以降は、ヒューリスティックと汎用ソルバーを同じ時間上限で比較します。

## 現在の読みどころ

最新レポートでは、次のような傾向が観測できます。

- 0.03秒では Cython / Numba の単体 SA が即座に実行可能解を返す
- 1秒では Cython / Numba のヒューリスティックが Cbc より高いスコア帯に到達する
- 10秒では Cython の進化計算が Cbc と競合する水準まで伸びる
- 30秒〜60秒では Cython の進化計算が Cbc にかなり近い平均スコアを出す
- Numba 実装は全体に安定して高水準だが、現状では Cython 進化計算が最も強い場面が多い
- 進化計算は単体 SA よりスコアの振れ幅が大きく、このばらつき自体が探索される解の多様性を示している
- 一方、Cbc / CP-SAT / Gecode のような決定論的ソルバーにはこの種の多様性はなく、同一条件ではほぼ同じ解へ収束する

このため、本リポジトリの主張は「常に厳密解法を上回る」ことではありません。むしろ、短時間で高品質な実行可能解を返せるヒューリスティック実装を、汎用ソルバーと比較可能な形で提示することにあります。

### 時間別スコア推移（平均 ± 標準偏差）

![時間別スコア推移](results/reports/benchmark_main_ribbon_line_jp.png)

## レポート出力

`scripts/generate_report.py` は、結果ファイルから日本語レポートと複数の図を生成します。

主な出力:

- 時間予算別の統計表
- メイン: リボン付き線グラフ（平均 ± 標準偏差）
- サブ1: 60秒時点の箱ひげ図
- サブ2: CBCとの差分スコア図

注記:

- メイン線グラフと CBC 差分スコア図の横軸は、`0.03, 1, 10, 30, 60` 秒を実時間比ではなく指定順で等間隔に配置しています
- CBC 差分スコアは `各手法平均 - CBC平均` で定義しています

### 60秒時点の目的関数値分布

![60秒箱ひげ図](results/reports/benchmark_sub_boxplot_60s_jp.png)

### CBCとの差分スコア

![CBCとの差分スコア](results/reports/benchmark_sub_delta_vs_cbc_jp.png)

## 主要スクリプト

- `scripts/solve_with_cython.py`
	Cython 実装の単体 SA / 進化計算を実行
- `scripts/solve_with_numba.py`
	Numba 実装の単体 SA / 進化計算を実行
- `scripts/solve_with_minizinc_solvers.py`
	MiniZinc 経由で Cbc / CP-SAT / Gecode を実行
- `scripts/run_timeout_experiments.py`
	複数時間予算・複数回実行の実験ランナー
- `scripts/generate_report.py`
	実行結果から統計レポートとグラフを生成

## セットアップ

`uv` を使う前提です。

```bash
uv sync
```

## 実行手順

### 1. 問題インスタンス生成

```bash
uv run python scripts/generate_and_save_problem.py
```

### 2. Cython 拡張ビルド

```bash
uv run python src/solver_cython/setup.py build_ext --inplace
Copy-Item -Force .\build\lib.win-amd64-cpython-314\solver_cython\core.cp314-win_amd64.pyd .\src\solver_cython\core.cp314-win_amd64.pyd
```

### 3. 個別実行

```bash
uv run python scripts/solve_with_cython.py --timeout 10 --no-full-output
uv run python scripts/solve_with_numba.py --timeout 10 --no-full-output
uv run python scripts/solve_with_minizinc_solvers.py --timeout 10 --no-full-output
```

### 4. 一括実験

全ソルバーを時間予算ごとにまとめて回す場合:

```bash
uv run python scripts/run_timeout_experiments.py
```

例: Cython と Numba の 100 秒だけを再実行

```bash
uv run python scripts/run_timeout_experiments.py --solvers cython,numba --timeouts 100
```

### 5. レポート生成

```bash
uv run python scripts/generate_report.py
```

## 今後の改善候補

- 箱ひげ図を確率的ソルバーのみ表示にして見やすくする
- レポートに 95% 信頼区間を追加する
- 問題インスタンスを複数用意して、単一インスタンス依存を減らす
- MiniZinc 系ソルバーの `status` と `best bound` をより丁寧に整理する

## 位置づけ

これは「Cython が常に最強」であることを示すデモではなく、以下を示すためのポートフォリオです。

- 制約の厳しい組合せ最適化問題を設計できること
- 厳密解法とヒューリスティックを同じ時間予算で比較できること
- Cython / Numba による高速化実装ができること
- 単発比較ではなく、統計的な見方を含めて結果を整理できること