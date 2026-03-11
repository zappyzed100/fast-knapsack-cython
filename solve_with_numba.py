import pandas as pd
import numpy as np
import time
import os
import argparse
import sys
import datetime
from numba import njit, uint64, int32, int8, float32, float64

# ---------------------------------------------------------
# ソルバー本体 (ロジック変更なし)
# ---------------------------------------------------------


@njit(inline="always")
def get_group_bonus(count, bonus_val):
    if 3 <= count <= 5:
        return float64(bonus_val)
    return 0.0


@njit(cache=True)
def solve_knapsack_sa_numba(
    values,
    weights,
    capacities,
    item_groups,
    conflict_pairs,
    n_items,
    n_groups,
    group_max,
    max_iter,
):
    conflict_masks = np.zeros((1000, 16), dtype=uint64)
    current_groups_bit = np.zeros(16, dtype=uint64)

    for i in range(conflict_pairs.shape[0]):
        g1, g2 = conflict_pairs[i, 0], conflict_pairs[i, 1]
        if g1 < 1000 and g2 < 1000:
            conflict_masks[g1, g2 // 64] |= uint64(1) << (g2 % 64)
            conflict_masks[g2, g1 // 64] |= uint64(1) << (g1 % 64)

    current_sol = np.zeros(n_items, dtype=int8)
    group_counts = np.zeros(n_groups, dtype=int32)
    current_weights = np.zeros(3, dtype=int32)
    current_score = 0.0
    best_score = 0.0
    best_sol = np.zeros(n_items, dtype=int8)
    BONUS_VAL = 50.0

    rand_add = np.random.randint(0, n_items, size=max_iter).astype(int32)
    rand_rem = np.random.randint(0, n_items, size=max_iter).astype(int32)
    rand_flt = np.random.random(size=max_iter).astype(float32)

    for it in range(max_iter):
        add_idx = rand_add[it]
        g_add = item_groups[add_idx]
        temp = 1.0 - (float64(it) / max_iter)
        r_val = rand_flt[it]

        if current_sol[add_idx] == 0:
            over_weight = False
            for j in range(3):
                if current_weights[j] + weights[add_idx, j] > capacities[j]:
                    over_weight = True
                    break

            if over_weight:
                rem_idx = rand_rem[it]
                if current_sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    w_ok = True
                    for j in range(3):
                        if (
                            current_weights[j]
                            - weights[rem_idx, j]
                            + weights[add_idx, j]
                            > capacities[j]
                        ):
                            w_ok = False
                            break
                    if w_ok:
                        current_groups_bit[g_rem // 64] &= ~(uint64(1) << (g_rem % 64))
                        conflict = False
                        for k in range(16):
                            if current_groups_bit[k] & conflict_masks[g_add, k]:
                                conflict = True
                                break
                        if not conflict:
                            group_counts[g_add] += 1
                            group_counts[g_rem] -= 1
                            new_total_bonus = 0.0
                            for g_idx in range(n_groups):
                                if 3 <= group_counts[g_idx] <= 5:
                                    new_total_bonus += BONUS_VAL
                            val_diff = float64(values[add_idx] - values[rem_idx])
                            group_counts[g_add] -= 1
                            group_counts[g_rem] += 1
                            old_total_bonus = 0.0
                            for g_idx in range(n_groups):
                                if 3 <= group_counts[g_idx] <= 5:
                                    old_total_bonus += BONUS_VAL
                            diff = (val_diff + new_total_bonus) - old_total_bonus
                            if diff > 0 or r_val < 0.05 * temp:
                                current_sol[rem_idx], current_sol[add_idx] = 0, 1
                                for j in range(3):
                                    current_weights[j] += (
                                        weights[add_idx, j] - weights[rem_idx, j]
                                    )
                                group_counts[g_add] += 1
                                group_counts[g_rem] -= 1
                                current_score += val_diff
                                if group_counts[g_rem] > 0:
                                    current_groups_bit[g_rem // 64] |= uint64(1) << (
                                        g_rem % 64
                                    )
                                current_groups_bit[g_add // 64] |= uint64(1) << (
                                    g_add % 64
                                )
                                final_sc = current_score + new_total_bonus
                                if final_sc > best_score:
                                    best_score = final_sc
                                    best_sol[:] = current_sol[:]
                                continue
                        if group_counts[g_rem] > 0:
                            current_groups_bit[g_rem // 64] |= uint64(1) << (g_rem % 64)
            elif group_counts[g_add] < group_max:
                conflict = False
                for k in range(16):
                    if current_groups_bit[k] & conflict_masks[g_add, k]:
                        conflict = True
                        break
                if not conflict:
                    val_diff = float64(values[add_idx])
                    group_counts[g_add] += 1
                    total_b = 0.0
                    for g_idx in range(n_groups):
                        if 3 <= group_counts[g_idx] <= 5:
                            total_b += BONUS_VAL
                    group_counts[g_add] -= 1
                    old_total_b = 0.0
                    for g_idx in range(n_groups):
                        if 3 <= group_counts[g_idx] <= 5:
                            old_total_b += BONUS_VAL
                    diff = (val_diff + total_b) - old_total_b
                    if diff > 0 or r_val < 0.1 * temp:
                        current_sol[add_idx] = 1
                        for j in range(3):
                            current_weights[j] += weights[add_idx, j]
                        group_counts[g_add] += 1
                        current_score += val_diff
                        current_groups_bit[g_add // 64] |= uint64(1) << (g_add % 64)
                        final_sc = current_score + total_b
                        if final_sc > best_score:
                            best_score = final_sc
                            best_sol[:] = current_sol[:]
        else:
            if r_val < 0.02 * temp:
                g_rem = item_groups[add_idx]
                val_diff = float64(-values[add_idx])
                old_total_b = 0.0
                for g_idx in range(n_groups):
                    if 3 <= group_counts[g_idx] <= 5:
                        old_total_b += BONUS_VAL
                group_counts[g_rem] -= 1
                new_total_b = 0.0
                for g_idx in range(n_groups):
                    if 3 <= group_counts[g_idx] <= 5:
                        new_total_b += BONUS_VAL
                diff = (val_diff + new_total_b) - old_total_b
                current_sol[add_idx] = 0
                for j in range(3):
                    current_weights[j] -= weights[add_idx, j]
                current_score += val_diff
                if group_counts[g_rem] == 0:
                    current_groups_bit[g_rem // 64] &= ~(uint64(1) << (g_rem % 64))

    return best_score, best_sol


# ---------------------------------------------------------
# バリデーター
# ---------------------------------------------------------


def validate_solution(
    solution,
    values,
    weights,
    capacities,
    item_groups,
    conflict_pairs,
    group_max,
    bonus_val=50,
):
    is_valid = True
    selected_indices = np.where(solution == 1)[0]
    total_weights = np.zeros(weights.shape[1], dtype=np.int64)
    for idx in selected_indices:
        total_weights += weights[idx]
    for tw, cap in zip(total_weights, capacities):
        if tw > cap:
            is_valid = False
    u_groups, counts = np.unique(item_groups[selected_indices], return_counts=True)
    for count in counts:
        if count > group_max:
            is_valid = False
    g_set = set(u_groups)
    for g1, g2 in conflict_pairs:
        if g1 in g_set and g2 in g_set:
            is_valid = False
    base_score = np.sum(values[selected_indices])
    bonus_score = 0
    for count in counts:
        if 3 <= count <= 5:
            bonus_score += bonus_val
    return is_valid, int(base_score + bonus_score)


# ---------------------------------------------------------
# ベンチマーカー
# ---------------------------------------------------------


class NumbaBenchmarker:
    def __init__(self, csv_path="problem_data.csv", constraints_path="constraints.txt"):
        self.csv_path = csv_path
        self.constraints_path = constraints_path

    def run(self, iterations=10000000):
        df = pd.read_csv(self.csv_path)
        with open(self.constraints_path, "r") as f:
            lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}

        caps = np.array(list(map(int, lines["capacities"].split(","))), dtype=np.int32)
        confs = np.array(
            [
                tuple(map(int, c.split(",")))
                for c in lines.get("conflicts", "").split(";")
                if c
            ],
            dtype=np.int32,
        )
        v = df["value"].values.astype(np.int32)
        w = df[["weight0", "weight1", "weight2"]].values.astype(np.int32)
        g = df["group_id"].values.astype(np.int32)

        start_time = time.perf_counter()
        score, sol = solve_knapsack_sa_numba(
            v, w, caps, g, confs, len(df), int(g.max() + 1), 10, iterations
        )
        elapsed = time.perf_counter() - start_time

        valid, v_score = validate_solution(sol, v, w, caps, g, confs, 10)

        # 結果の保存
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        output_path = os.path.join(result_dir, "numba_results.txt")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_text = f"""[{timestamp}]
Solver: numba
Status: SATISFIED
Validation: {'VALID' if valid else 'INVALID'}
Objective Value (Score): {v_score}
Execution Time: {elapsed:.4f} seconds
--------------------------------------------------
"""
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result_text)

        print(result_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=10000000)
    args = parser.parse_args()
    NumbaBenchmarker().run(iterations=args.iter)
