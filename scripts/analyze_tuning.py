import optuna
import pandas as pd
import numpy as np
import os
import sys

# プロジェクトルートのパスを取得
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def analyze_and_determine_params():
    # 1. Studyのロード (ルートにあるDBを参照)
    db_path = os.path.join(PROJECT_ROOT, "knapsack_tuning.db")
    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(
            study_name="knapsack_production_tuning", storage=storage_url
        )
    except Exception as e:
        print(f"Error: データベースの読み込みに失敗しました。\nパス: {db_path}\n{e}")
        return

    # 2. 全試行データをPandas DataFrameで出力
    df = study.trials_dataframe()
    df_sorted = df.sort_values(by="value", ascending=False)

    # 保存先フォルダの作成 (data/tuning_results/)
    output_dir = os.path.join(PROJECT_ROOT, "data", "tuning_results")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "tuning_full_results.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"--- 全 {len(df)} 試行のデータを {csv_path} に保存しました ---")

    # 3. 上位試行の分析
    top_n = 10
    top_trials = df_sorted.head(top_n)

    print(
        f"\n[上位 {top_n} 件のスコア範囲: {top_trials['value'].min()} ～ {top_trials['value'].max()}]"
    )

    # 4. 適切なパラメータの決定
    param_cols = [c for c in df.columns if c.startswith("params_")]
    recommended_params = {}
    print("\n--- 推奨パラメータの決定ロジック (上位平均) ---")

    for col in param_cols:
        param_name = col.replace("params_", "")
        mean_val = top_trials[col].mean()
        std_val = top_trials[col].std()

        if "size" in param_name or "gen" in param_name or "iter" in param_name:
            recommended_params[param_name] = int(round(mean_val))
        else:
            recommended_params[param_name] = round(float(mean_val), 4)

        print(
            f"{param_name:15}: {recommended_params[param_name]} (上位分散: ±{std_val:.2f})"
        )

    return recommended_params


if __name__ == "__main__":
    best_config = analyze_and_determine_params()

    if best_config:
        print("\n--- ソルバー用貼り付けコード ---")
        print("final_params = {")
        for k, v in best_config.items():
            print(f"    '{k}': {v},")
        print("}")
