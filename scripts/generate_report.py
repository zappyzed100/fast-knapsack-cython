import os
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIME_BUDGETS = [1, 10, 30, 60]
SA_ONLY_BUDGETS = [0.05]
SA_ONLY_SOLVERS = {"cython_single_sa", "numba_single_sa"}
MAX_SAMPLES_PER_SOLVER = 20

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


def _budget_upper(budget_sec):
    return budget_sec + min(2.0, budget_sec * 5)


def filter_records_by_budget(records, budget_sec, solver_filter=None):
    """budget_sec < 実行時間 < budget_sec + バッファ のレコードを返す。"""
    upper = _budget_upper(budget_sec)
    out = []
    for rec in records:
        if solver_filter is not None and rec["_raw_name"] not in solver_filter:
            continue
        t = rec["実行時間 (秒)"]
        if t > budget_sec and t < upper:
            out.append(rec)
    return out


def select_latest_samples(records, max_samples=MAX_SAMPLES_PER_SOLVER):
    """ソルバーごとに最新 max_samples 件だけを残す。"""
    if not records:
        return []

    df = pd.DataFrame(records)
    latest = (
        df.sort_values(by=["_timestamp", "_section_index"], ascending=[True, True])
        .groupby("_raw_name", group_keys=False)
        .tail(max_samples)
    )
    return latest.to_dict("records")


def summarize_by_budget(records, budget_sec, solver_filter=None):
    """時間窓内のデータをソルバー別に集計（平均・標準偏差・試行数）する。"""
    filtered = filter_records_by_budget(
        records, budget_sec, solver_filter=solver_filter
    )
    filtered = select_latest_samples(filtered)
    if not filtered:
        return pd.DataFrame(
            columns=[
                "手法・実装",
                "目的関数値 平均",
                "目的関数値 標準偏差",
                "試行数",
                "実行時間 平均 (秒)",
                "計算状態(最新)",
            ]
        )

    df = pd.DataFrame(filtered)
    grouped = df.groupby(["_raw_name", "手法・実装"], as_index=False).agg(
        **{
            "目的関数値 平均": ("目的関数値", "mean"),
            "目的関数値 標準偏差": ("目的関数値", "std"),
            "試行数": ("目的関数値", "count"),
            "実行時間 平均 (秒)": ("実行時間 (秒)", "mean"),
            "_latest_ts": ("_timestamp", "max"),
        }
    )

    # 最新状態を引く
    latest_rows = (
        df.sort_values(by=["_timestamp", "_section_index"], ascending=[True, True])
        .groupby("_raw_name", as_index=False)
        .tail(1)
    )[["_raw_name", "計算状態"]].rename(columns={"計算状態": "計算状態(最新)"})

    out = grouped.merge(latest_rows, on="_raw_name", how="left")
    out["目的関数値 標準偏差"] = out["目的関数値 標準偏差"].fillna(0.0)
    out["目的関数値 平均"] = out["目的関数値 平均"].round(2)
    out["目的関数値 標準偏差"] = out["目的関数値 標準偏差"].round(2)
    out["実行時間 平均 (秒)"] = out["実行時間 平均 (秒)"].round(4)

    out = out[
        [
            "_raw_name",
            "手法・実装",
            "目的関数値 平均",
            "目的関数値 標準偏差",
            "試行数",
            "実行時間 平均 (秒)",
            "計算状態(最新)",
        ]
    ].sort_values(by=["目的関数値 平均", "実行時間 平均 (秒)"], ascending=[False, True])
    return out


def collect_budget_stats(records, budgets, solver_filter=None):
    """予算ごとのソルバー統計を縦持ち形式で返す。"""
    rows = []
    for budget in budgets:
        summ = summarize_by_budget(records, budget, solver_filter=solver_filter)
        if summ.empty:
            continue
        for _, r in summ.iterrows():
            rows.append(
                {
                    "budget": budget,
                    "raw": r["_raw_name"],
                    "label": r["手法・実装"],
                    "mean": float(r["目的関数値 平均"]),
                    "std": float(r["目的関数値 標準偏差"]),
                    "count": int(r["試行数"]),
                }
            )
    return pd.DataFrame(rows)


def _budget_label(budget):
    if float(budget) < 1.0:
        return f"{float(budget):.2f}"
    return str(int(float(budget)))


def _build_budget_axis(budgets):
    """指定順で等間隔のX軸を作る。"""
    ordered = [float(b) for b in budgets]
    budget_to_x = {b: float(i) for i, b in enumerate(ordered)}
    ticks = [budget_to_x[b] for b in ordered]
    labels = [_budget_label(b) for b in ordered]
    return budget_to_x, ticks, labels


def save_ribbon_line_chart(records, budgets, output_png):
    """メイン図: 平均線 + 標準偏差リボン。"""
    plot_df = collect_budget_stats(records, budgets)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 6.2))
    labels = sorted(plot_df["label"].unique())
    budget_to_x, ticks, tick_labels = _build_budget_axis(budgets)

    for label in labels:
        s = plot_df[plot_df["label"] == label].sort_values("budget")
        if s.empty:
            continue
        x = s["budget"].astype(float).map(budget_to_x).to_numpy()
        y = s["mean"].astype(float).to_numpy()
        std = s["std"].astype(float).to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label=label)
        ax.fill_between(x, y - std, y + std, alpha=0.15)

    ax.set_xlabel("時間上限 (秒)")
    ax.set_ylabel("目的関数値")
    ax.set_title("時間別スコア推移（平均 ± 標準偏差）")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(3500, 4600)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.03,
        "注: 横軸は実時間差の比例ではなく、0.05, 1, 10, 30, 60 秒を指定順で等間隔配置",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.14)
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def save_boxplot_60s(records, output_png):
    """サブ図1: 60秒時点の箱ひげ図。"""
    b = 60
    filtered = filter_records_by_budget(records, b)
    filtered = select_latest_samples(filtered)
    if not filtered:
        return

    df = pd.DataFrame(filtered)
    labels = sorted(df["手法・実装"].unique())
    data = [
        df[df["手法・実装"] == label]["目的関数値"].astype(float).to_numpy()
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title("60秒時点の目的関数値分布（箱ひげ図）")
    ax.set_ylabel("目的関数値")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def save_delta_vs_cbc_chart(records, budgets, output_png):
    """サブ図2: CBC平均を基準としたΔスコア図。"""
    plot_df = collect_budget_stats(records, budgets)
    if plot_df.empty:
        return

    cbc_rows = plot_df[plot_df["raw"] == "cbc"]
    if cbc_rows.empty:
        return
    cbc_by_budget = dict(zip(cbc_rows["budget"], cbc_rows["mean"]))

    fig, ax = plt.subplots(figsize=(13, 6.2))
    labels = sorted(plot_df["label"].unique())
    budget_to_x, ticks, tick_labels = _build_budget_axis(budgets)

    for label in labels:
        s = plot_df[plot_df["label"] == label].sort_values("budget")
        if s.empty:
            continue

        x = []
        y = []
        for _, row in s.iterrows():
            b = row["budget"]
            if b not in cbc_by_budget:
                continue
            x.append(budget_to_x[float(b)])
            y.append(float(row["mean"] - cbc_by_budget[b]))

        if not x:
            continue
        ax.plot(x, y, marker="o", linewidth=2, label=label)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("時間上限 (秒)")
    ax.set_ylabel("CBCとの差分スコア (各手法平均 - CBC平均)")
    ax.set_title("CBCとの差分スコア")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-1000, 400)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.03,
        "注: 横軸は実時間差の比例ではなく、1, 10, 30, 60 秒を指定順で等間隔配置",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.14)
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


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
        f.write("## 時間予算別の統計結果\n\n")
        f.write(
            "各予算について、時間窓内 (budget < 実行時間 < budget+バッファ) のデータから、ソルバーごとに最新20件までを集計し、目的関数値の平均・標準偏差・試行数を表示しています。\n\n"
        )
        f.write(
            "注: メイン線グラフとΔスコア図の横軸は、0.05・1・10・30・60秒を実時間比ではなく指定順で等間隔に配置しています。\n\n"
        )

        for budget in SA_ONLY_BUDGETS:
            df_budget = summarize_by_budget(
                all_records, budget, solver_filter=SA_ONLY_SOLVERS
            )
            f.write(f"### 実行時間上限 {budget} 秒 (Cython/Numba の単体SAのみ)\n\n")
            if df_budget.empty:
                f.write("該当データなし\n\n")
            else:
                f.write(
                    df_budget[
                        [
                            "手法・実装",
                            "目的関数値 平均",
                            "目的関数値 標準偏差",
                            "試行数",
                            "実行時間 平均 (秒)",
                            "計算状態(最新)",
                        ]
                    ].to_markdown(index=False)
                )
                f.write("\n\n")

                png_path = os.path.join(
                    reports_dir, f"benchmark_comparison_{budget}s_sa_jp.png"
                )
                save_table_png(
                    df_budget[
                        [
                            "手法・実装",
                            "目的関数値 平均",
                            "目的関数値 標準偏差",
                            "試行数",
                            "実行時間 平均 (秒)",
                            "計算状態(最新)",
                        ]
                    ],
                    f"ナップサック問題 SA比較表（上限 {budget} 秒）",
                    png_path,
                )
                print(f"比較画像を生成しました: {png_path}")

        for budget in TIME_BUDGETS:
            df_budget = summarize_by_budget(all_records, budget)
            f.write(f"### 実行時間上限 {budget} 秒\n\n")
            if df_budget.empty:
                f.write("該当データなし\n\n")
            else:
                f.write(
                    df_budget[
                        [
                            "手法・実装",
                            "目的関数値 平均",
                            "目的関数値 標準偏差",
                            "試行数",
                            "実行時間 平均 (秒)",
                            "計算状態(最新)",
                        ]
                    ].to_markdown(index=False)
                )
                f.write("\n\n")

                png_path = os.path.join(
                    reports_dir, f"benchmark_comparison_{budget}s_jp.png"
                )
                save_table_png(
                    df_budget[
                        [
                            "手法・実装",
                            "目的関数値 平均",
                            "目的関数値 標準偏差",
                            "試行数",
                            "実行時間 平均 (秒)",
                            "計算状態(最新)",
                        ]
                    ],
                    f"ナップサック問題 比較表（上限 {budget} 秒）",
                    png_path,
                )
                print(f"比較画像を生成しました: {png_path}")

        # 追加グラフ群
        graph_budgets = SA_ONLY_BUDGETS + TIME_BUDGETS

        main_line_png = os.path.join(reports_dir, "benchmark_main_ribbon_line_jp.png")
        save_ribbon_line_chart(all_records, graph_budgets, main_line_png)
        print(f"メイングラフを生成しました: {main_line_png}")

        sub_box_png = os.path.join(reports_dir, "benchmark_sub_boxplot_60s_jp.png")
        save_boxplot_60s(all_records, sub_box_png)
        print(f"サブグラフ(箱ひげ)を生成しました: {sub_box_png}")

        sub_delta_png = os.path.join(reports_dir, "benchmark_sub_delta_vs_cbc_jp.png")
        save_delta_vs_cbc_chart(all_records, TIME_BUDGETS, sub_delta_png)
        print(f"サブグラフ(Δスコア)を生成しました: {sub_delta_png}")

        f.write(
            "*数理最適化、制約プログラミング、および焼きなまし法（SA）による性能比較*\n"
        )

    print(f"レポートを生成しました: {output_md}")

    # 標準出力は60秒テーブルを表示
    df_60 = summarize_by_budget(all_records, 60)
    if not df_60.empty:
        print("\n[60秒上限の統計比較表]\n")
        print(
            df_60[
                [
                    "手法・実装",
                    "目的関数値 平均",
                    "目的関数値 標準偏差",
                    "試行数",
                    "実行時間 平均 (秒)",
                    "計算状態(最新)",
                ]
            ].to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
