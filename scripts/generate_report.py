import os
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIME_BUDGETS = [10, 30, 60, 100]

DISPLAY_NAMES = {
    "cbc": "Cbc (MILP/数理最適化)",
    "cp-sat": "CP-SAT (制約プログラミング)",
    "gecode": "Gecode (制約プログラミング)",
    "cython_single_sa": "Cython (単体SA/AOT)",
    "cython_sa_parallel_evolution": "Cython (進化計算/AOT)",
    "numba_single_sa": "Numba (単体SA/JIT)",
    "numba_hybrid_evolution": "Numba (進化計算/JIT)",
}

STATUS_MAPPING = {
    "SATISFIED": "近似解/満足解",
    "VALID": "近似解/満足解",
    "Optimal": "最適解",
}


def parse_result_file(file_path):
    """結果ファイルをパースし、実行セクションを時刻付きで返す。"""
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("-" * 50)
    results = []

    for idx, section in enumerate(sections):
        if not section.strip():
            continue

        solver_match = re.search(r"Solver:\s*([^\n\r]+)", section)
        ts_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", section)
        recalc_match = re.search(r"Recalculated Score:\s*([\d\.]+)", section)
        obj_match = re.search(
            r"Objective Value \((?:Solver|Score)\):\s*([\d\.]+)", section
        )
        score_match = re.search(r"Score:\s*([\d\.]+)", section)
        time_match = re.search(r"(?:Execution Time|Time):\s*([\d\.]+)", section)
        status_match = re.search(r"Status:\s*([^\n\r]+)", section)

        if not (
            solver_match and time_match and (recalc_match or obj_match or score_match)
        ):
            continue

        solver_name = solver_match.group(1).strip()
        if recalc_match:
            obj_val = int(float(recalc_match.group(1)))
        elif obj_match:
            obj_val = int(float(obj_match.group(1)))
        else:
            obj_val = int(float(score_match.group(1)))

        if ts_match:
            ts = datetime.datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
        else:
            ts = datetime.datetime.min

        time_val = float(time_match.group(1))
        raw_status = status_match.group(1).strip() if status_match else "SATISFIED"

        results.append(
            {
                "手法・実装": DISPLAY_NAMES.get(solver_name, solver_name),
                "目的関数値": obj_val,
                "実行時間 (秒)": time_val,
                "計算状態": STATUS_MAPPING.get(raw_status, raw_status),
                "実行日時": (
                    ts.strftime("%Y-%m-%d %H:%M:%S")
                    if ts != datetime.datetime.min
                    else "N/A"
                ),
                "_raw_name": solver_name,
                "_timestamp": ts,
                "_section_index": idx,
            }
        )

    return results


def select_latest_by_budget(records, budget_sec):
    """指定秒数以下で、各ソルバーの最新結果のみを返す。"""
    latest = {}

    for rec in records:
        if rec["実行時間 (秒)"] > budget_sec:
            continue

        key = rec["_raw_name"]
        if key not in latest:
            latest[key] = rec
            continue

        cur = latest[key]
        if (rec["_timestamp"], rec["_section_index"]) > (
            cur["_timestamp"],
            cur["_section_index"],
        ):
            latest[key] = rec

    data = list(latest.values())
    if not data:
        return pd.DataFrame(
            columns=[
                "手法・実装",
                "目的関数値",
                "実行時間 (秒)",
                "計算状態",
                "実行日時",
            ]
        )

    df = pd.DataFrame(data)[
        ["手法・実装", "目的関数値", "実行時間 (秒)", "計算状態", "実行日時"]
    ]
    return df.sort_values(by=["目的関数値", "実行時間 (秒)"], ascending=[False, True])


def save_table_png(df, title, output_png):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 6))
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
    table.set_fontsize(9)
    table.scale(1.2, 2.2)

    for j in range(len(df.columns)):
        table[0, j].get_text().set_color("white")
        table[0, j].get_text().set_weight("bold")

    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(row_colors[i % len(row_colors)])

    plt.title(title, fontsize=14, pad=16, weight="bold")
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "MS Gothic"

    results_dir = os.path.join(PROJECT_ROOT, "results")
    runs_dir = os.path.join(results_dir, "runs")
    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    target_files = [
        "cbc_results.txt",
        "cp_sat_results.txt",
        "gecode_results.txt",
        "cython_results.txt",
        "numba_results.txt",
    ]

    print(f"'{runs_dir}' 内の結果ファイルを走査中...")

    all_records = []
    for filename in target_files:
        path = os.path.join(runs_dir, filename)
        all_records.extend(parse_result_file(path))

    if not all_records:
        print("表示可能な結果データがありません。")
        return

    output_md = os.path.join(reports_dir, "benchmark_report_jp.md")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# ソルバー性能比較レポート\n\n")
        f.write(f"生成日時: {now_str}\n\n")
        f.write("## 時間予算別の最新結果\n\n")
        f.write(
            "実行時間が各予算以下のレコードから、ソルバーごとの最新データを採用しています。\n\n"
        )

        for budget in TIME_BUDGETS:
            df_budget = select_latest_by_budget(all_records, budget)
            f.write(f"### 実行時間上限 {budget} 秒\n\n")
            if df_budget.empty:
                f.write("該当データなし\n\n")
            else:
                f.write(df_budget.to_markdown(index=False))
                f.write("\n\n")

                png_path = os.path.join(
                    reports_dir, f"benchmark_comparison_{budget}s_jp.png"
                )
                save_table_png(
                    df_budget,
                    f"ナップサック問題 比較表（上限 {budget} 秒）",
                    png_path,
                )
                print(f"比較画像を生成しました: {png_path}")

        f.write(
            "*数理最適化、制約プログラミング、および焼きなまし法（SA）による性能比較*\n"
        )

    print(f"レポートを生成しました: {output_md}")

    # 標準出力は100秒テーブルを表示
    df_100 = select_latest_by_budget(all_records, 100)
    if not df_100.empty:
        print("\n[100秒上限の比較表]\n")
        print(df_100.to_markdown(index=False))


if __name__ == "__main__":
    main()
