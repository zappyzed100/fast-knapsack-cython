import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_result_file(file_path):
    """
    ファイルから全実行結果をパースし、各ソルバーごとの最新データを抽出する
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ソルバーのセクションごとに分割
    sections = content.split("-" * 50)
    results = []

    for section in sections:
        if not section.strip():
            continue

        # 各項目を抽出
        solver_match = re.search(r"Solver:\s*([^\n\r]+)", section)
        obj_match = re.search(
            r"(?:Objective Value \(Score\)|Score):\s*([\d\.]+)", section
        )
        time_match = re.search(r"(?:Execution Time|Time):\s*([\d\.]+)", section)
        status_match = re.search(r"Status:\s*([^\n\r]+)", section)

        if solver_match and obj_match and time_match:
            solver_name = solver_match.group(1).strip()
            obj_val = int(float(obj_match.group(1)))
            time_val = float(time_match.group(1))
            raw_status = status_match.group(1).strip() if status_match else "SATISFIED"

            # 表示名のマッピング
            display_names = {
                "cbc": "Cbc (MILP/数理最適化)",
                "cp-sat": "CP-SAT (制約プログラミング)",
                "gecode": "Gecode (制約プログラミング)",
                "cython_single_sa": "Cython (単体SA/AOT)",
                "cython_sa_parallel_evolution": "Cython (進化計算/AOT)",
                "numba_single_sa": "Numba (単体SA/JIT)",
                "numba_hybrid_evolution": "Numba (進化計算/JIT)",
            }

            status_mapping = {
                "SATISFIED": "近似解/満足解",
                "VALID": "近似解/満足解",
                "Optimal": "最適解",
            }

            results.append(
                {
                    "手法・実装": display_names.get(solver_name, solver_name),
                    "目的関数値": obj_val,
                    "実行時間 (秒)": time_val,
                    "計算状態": status_mapping.get(raw_status, raw_status),
                    "_raw_name": solver_name,  # 重複排除用
                }
            )

    return results


def main():
    # 日本語文字化け対策（MS Gothic等が入っていない環境への配慮が必要な場合は適宜変更）
    plt.rcParams["font.family"] = "MS Gothic"

    result_dir = "result"
    target_files = [
        "cbc_results.txt",
        "cp_sat_results.txt",
        "gecode_results.txt",
        "cython_results.txt",
        "numba_results.txt",
    ]

    all_data = []
    seen_solvers = {}

    print(f"'{result_dir}' 内の結果ファイルを走査中...")

    for filename in target_files:
        path = os.path.join(result_dir, filename)
        file_results = parse_result_file(path)

        # 各ファイル内の各ソルバーについて、最新（リストの最後）のデータのみを採用
        for res in file_results:
            seen_solvers[res["_raw_name"]] = res

    all_data = list(seen_solvers.values())

    if not all_data:
        print("表示可能な結果データがありません。")
        return

    # DataFrame作成とソート（スコア順、同点なら時間順）
    df = pd.DataFrame(all_data)[
        ["手法・実装", "目的関数値", "実行時間 (秒)", "計算状態"]
    ]
    df = df.sort_values(by=["目的関数値", "実行時間 (秒)"], ascending=[False, True])

    # --- Matplotlib で比較表画像を作成 ---
    fig, ax = plt.subplots(figsize=(12, 6))
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
    table.set_fontsize(10)
    table.scale(1.2, 2.5)

    for j in range(len(df.columns)):
        table[0, j].get_text().set_color("white")
        table[0, j].get_text().set_weight("bold")

    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(row_colors[i % len(row_colors)])

    plt.title(
        "ナップサック問題 ソルバー性能比較ベンチマーク",
        fontsize=16,
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
        f.write(
            f"生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write(df.to_markdown(index=False))
        f.write(
            "\n\n*数理最適化、制約プログラミング、および焼きなまし法（SA）による性能比較*"
        )
    print(f"レポートを生成しました: {output_md}")

    # 標準出力
    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    import datetime

    main()
