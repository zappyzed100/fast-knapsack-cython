import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_result_file(file_path):
    """
    ファイルから最新の 目的関数値(Score) と 実行時間 を抽出する
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 各項目を抽出（複数ある場合は最後=最新のものを採用）
    objectives = re.findall(r"(?:Total Objective|Score):\s*([\d\.]+)", content)
    times = re.findall(r"(?:Execution Time|Elapsed):\s*([\d\.]+)", content)

    # 正規表現を修正：改行までをすべて取得するように変更（末尾の ) を含めるため）
    statuses = re.findall(r"Status:\s*([^\n\r]+)", content)

    if not objectives or not times:
        return None

    # 数値変換
    obj_val = int(float(objectives[-1]))
    time_val = float(times[-1])

    # ステータスの日本語化
    raw_status = statuses[-1].strip() if statuses else "Completed"
    status_mapping = {
        "Optimal": "最適解 (緩和モデル)",
        "Completed (Max Iterations reached)": "近似解 (完遂)",
        "Completed": "近似解 (完遂)",
        "TIMEOUT (Partial Solution)": "暫定解 (中断)",
        "Not Solved": "解なし",
    }
    # .get() で変換。マッチしない場合は元の文字列を返す
    status_val = status_mapping.get(raw_status, raw_status)

    return {"目的関数値": obj_val, "実行時間 (秒)": time_val, "計算状態": status_val}


def main():
    # 日本語文字化け対策（Windows用）
    plt.rcParams["font.family"] = "MS Gothic"

    result_dir = "result"
    data = []

    # 実装手法の短縮・日本語表示名定義
    target_configs = [
        ("cython_hill_climbing_results.txt", "Cython (勾配法/AOT)"),
        ("numba_results.txt", "Numba (勾配法/JIT)"),
        ("pulp_results.txt", "PuLP (数理最適化)"),
        ("cython_dfs_results.txt", "Cython (全探索)"),
    ]

    print(f"'{result_dir}' 内の結果を走査中...")

    for filename, display_name in target_configs:
        path = os.path.join(result_dir, filename)
        res = parse_result_file(path)
        if res:
            res["手法・実装"] = display_name
            data.append(res)
            print(f"  読み込み完了: {filename}")
        else:
            print(f"  警告: {filename} に有効なデータが見つかりません")

    if not data:
        print("表示可能な結果データがありません。")
        return

    # DataFrame作成とソート（スコア順）
    df = pd.DataFrame(data)[["手法・実装", "目的関数値", "実行時間 (秒)", "計算状態"]]
    df = df.sort_values("目的関数値", ascending=False)

    # --- Matplotlib で比較表画像を作成 ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    header_color = "#40466e"
    row_colors = ["#f1f1f2", "w"]

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=[header_color] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    # ヘッダースタイル設定
    for j in range(len(df.columns)):
        table[0, j].get_text().set_color("white")
        table[0, j].get_text().set_weight("bold")

    # 行の背景色を交互に設定
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(row_colors[i % len(row_colors)])

    plt.title(
        "ナップサック問題 ソルバー性能比較ベンチマーク",
        fontsize=15,
        pad=20,
        weight="bold",
    )

    # 画像保存
    output_png = os.path.join(result_dir, "benchmark_comparison_jp.png")
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    print(f"\n比較画像を生成しました: {output_png}")

    # Markdown レポート作成
    output_md = os.path.join(result_dir, "benchmark_report_jp.md")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# ソルバー性能比較レポート\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*異なる実装技術と最適化アプローチによる性能比較*")
    print(f"レポートを生成しました: {output_md}")

    # 標準出力
    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
