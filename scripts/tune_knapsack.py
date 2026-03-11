import optuna
import numpy as np
import pandas as pd
import os
import sys

# プロジェクトルートをパスに追加して solver_cython をインポート可能にする
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from solver_cython.core import solve_knapsack_sa_parallel


def load_problem_data():
    """scripts/ から見て ../data/ にあるファイルをロード"""
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


# グローバル変数の初期化
G_DATA = None


def objective(trial):
    global G_DATA
    if G_DATA is None:
        G_DATA = load_problem_data()

    params = {
        "pop_size": trial.suggest_int("pop_size", 40, 100),
        "elite_size": trial.suggest_int("elite_size", 10, 30),
        "num_new_gen": trial.suggest_int("num_new_gen", 20, 50),
        "max_generations": trial.suggest_int("max_generations", 100, 200),
        "iter_per_ind": trial.suggest_int("iter_per_ind", 10000, 60000),
        "prob_crossover": trial.suggest_float("prob_crossover", 0.2, 0.7),
        "prob_mut_repair": trial.suggest_float("prob_mut_repair", 0.05, 0.3),
        "prob_mut_30": trial.suggest_float("prob_mut_30", 0.2, 0.6),
        "w_slack": trial.suggest_float("w_slack", 0.1, 5.0),
        "w_density": trial.suggest_float("w_density", 100.0, 5000.0),
    }

    score, _ = solve_knapsack_sa_parallel(
        G_DATA[0],
        G_DATA[1],
        G_DATA[2],
        G_DATA[3],
        G_DATA[4],
        G_DATA[5],
        G_DATA[6],
        G_DATA[7],
        **params,
    )
    return float(score)


if __name__ == "__main__":
    # DBファイルはプロジェクトルートに配置
    db_path = os.path.join(PROJECT_ROOT, "knapsack_tuning.db")
    storage_url = f"sqlite:///{db_path}?timeout=60"

    study = optuna.create_study(
        study_name="knapsack_production_tuning",
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"Trial starts... (DB: {db_path})")
    # 安定性のためシングルプロセス。複数ターミナルで実行して並列化してください
    study.optimize(objective, n_trials=300)
