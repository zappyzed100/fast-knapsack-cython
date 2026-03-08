import pandas as pd
import numpy as np
import time
import os
from numba import njit, uint64, int32, int8


@njit(cache=True)
def solve_knapsack_numba_sa(
    values,
    weights,
    capacities,
    item_groups,
    conflict_pairs,
    n_items,
    n_groups,
    group_max,
    max_iter=5000000,
):
    # --- ビットセットの準備 ---
    current_groups_bit = np.zeros(4, dtype=uint64)
    conflict_masks = np.zeros((200, 4), dtype=uint64)

    for i in range(conflict_pairs.shape[0]):
        g1 = conflict_pairs[i, 0]
        g2 = conflict_pairs[i, 1]
        conflict_masks[g1, g2 // 64] |= uint64(1) << uint64(g2 % 64)
        conflict_masks[g2, g1 // 64] |= uint64(1) << uint64(g1 % 64)

    # --- 状態管理 ---
    current_sol = np.zeros(n_items, dtype=int8)
    group_counts = np.zeros(n_groups, dtype=int32)
    current_weights = np.zeros(3, dtype=int32)
    current_score = 0
    best_score = 0
    best_sol = np.zeros(n_items, dtype=int8)

    # 乱数生成
    rand_add = np.random.randint(0, n_items, max_iter)
    rand_rem = np.random.randint(0, n_items, max_iter)
    rand_flt = np.random.random(max_iter)

    for it in range(max_iter):
        add_idx = rand_add[it]
        g_add = item_groups[add_idx]
        temp = 1.0 - (float(it) / max_iter)
        r_val = rand_flt[it]

        if current_sol[add_idx] == 0:
            # 重量チェック
            if (
                (current_weights[0] + weights[add_idx, 0] > capacities[0])
                or (current_weights[1] + weights[add_idx, 1] > capacities[1])
                or (current_weights[2] + weights[add_idx, 2] > capacities[2])
            ):

                # --- A. 1個スワップ ---
                rem_idx = rand_rem[it]
                if current_sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    if (
                        (
                            current_weights[0]
                            - weights[rem_idx, 0]
                            + weights[add_idx, 0]
                            <= capacities[0]
                        )
                        and (
                            current_weights[1]
                            - weights[rem_idx, 1]
                            + weights[add_idx, 1]
                            <= capacities[1]
                        )
                        and (
                            current_weights[2]
                            - weights[rem_idx, 2]
                            + weights[add_idx, 2]
                            <= capacities[2]
                        )
                    ):

                        current_groups_bit[g_rem // 64] &= ~(
                            uint64(1) << uint64(g_rem % 64)
                        )
                        if not (
                            (current_groups_bit[0] & conflict_masks[g_add, 0])
                            or (current_groups_bit[1] & conflict_masks[g_add, 1])
                            or (current_groups_bit[2] & conflict_masks[g_add, 2])
                            or (current_groups_bit[3] & conflict_masks[g_add, 3])
                        ):

                            score_diff = values[add_idx] - values[rem_idx]
                            if 2 <= group_counts[g_add] <= 4:
                                score_diff += 50
                            if 3 <= group_counts[g_rem] <= 5:
                                score_diff -= 50

                            if score_diff > 0 or r_val < 0.05 * temp:
                                current_sol[rem_idx] = 0
                                current_sol[add_idx] = 1
                                for j in range(3):
                                    current_weights[j] = (
                                        current_weights[j]
                                        - weights[rem_idx, j]
                                        + weights[add_idx, j]
                                    )
                                group_counts[g_rem] -= 1
                                group_counts[g_add] += 1
                                current_score += score_diff
                                current_groups_bit[g_add // 64] |= uint64(1) << uint64(
                                    g_add % 64
                                )
                                if current_score > best_score:
                                    best_score = current_score
                                    best_sol[:] = current_sol[:]
                                continue
                        if group_counts[g_rem] > 0:
                            current_groups_bit[g_rem // 64] |= uint64(1) << uint64(
                                g_rem % 64
                            )

                # --- B. 2個捨て1個入れスワップ ---
                if r_val < 0.1:
                    rem_idx = rand_rem[it]
                    rem_idx2 = (rem_idx + 7) % n_items
                    if (
                        current_sol[rem_idx] == 1
                        and current_sol[rem_idx2] == 1
                        and rem_idx != rem_idx2
                    ):
                        g_rem = item_groups[rem_idx]
                        g_rem2 = item_groups[rem_idx2]

                        can_swap2 = True
                        for j in range(3):
                            if (
                                current_weights[j]
                                - weights[rem_idx, j]
                                - weights[rem_idx2, j]
                                + weights[add_idx, j]
                                > capacities[j]
                            ):
                                can_swap2 = False
                                break

                        if can_swap2:
                            current_groups_bit[g_rem // 64] &= ~(
                                uint64(1) << uint64(g_rem % 64)
                            )
                            current_groups_bit[g_rem2 // 64] &= ~(
                                uint64(1) << uint64(g_rem2 % 64)
                            )

                            if not (
                                (current_groups_bit[0] & conflict_masks[g_add, 0])
                                or (current_groups_bit[1] & conflict_masks[g_add, 1])
                                or (current_groups_bit[2] & conflict_masks[g_add, 2])
                                or (current_groups_bit[3] & conflict_masks[g_add, 3])
                            ):

                                score_diff = (
                                    values[add_idx] - values[rem_idx] - values[rem_idx2]
                                )
                                if 2 <= group_counts[g_add] <= 4:
                                    score_diff += 50

                                if score_diff > 0 or r_val < 0.01 * temp:
                                    current_sol[rem_idx] = 0
                                    current_sol[rem_idx2] = 0
                                    current_sol[add_idx] = 1
                                    for j in range(3):
                                        current_weights[j] = (
                                            current_weights[j]
                                            - weights[rem_idx, j]
                                            - weights[rem_idx2, j]
                                            + weights[add_idx, j]
                                        )
                                    group_counts[g_rem] -= 1
                                    group_counts[g_rem2] -= 1
                                    group_counts[g_add] += 1
                                    current_score += score_diff
                                    current_groups_bit[g_add // 64] |= uint64(
                                        1
                                    ) << uint64(g_add % 64)
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_sol[:] = current_sol[:]
                                    continue

                            if group_counts[g_rem] > 0:
                                current_groups_bit[g_rem // 64] |= uint64(1) << uint64(
                                    g_rem % 64
                                )
                            if group_counts[g_rem2] > 0:
                                current_groups_bit[g_rem2 // 64] |= uint64(1) << uint64(
                                    g_rem2 % 64
                                )
                continue

            # 単純追加
            if group_counts[g_add] < group_max:
                if not (
                    (current_groups_bit[0] & conflict_masks[g_add, 0])
                    or (current_groups_bit[1] & conflict_masks[g_add, 1])
                    or (current_groups_bit[2] & conflict_masks[g_add, 2])
                    or (current_groups_bit[3] & conflict_masks[g_add, 3])
                ):

                    current_sol[add_idx] = 1
                    for j in range(3):
                        current_weights[j] += weights[add_idx, j]
                    score_diff = values[add_idx]
                    if 2 <= group_counts[g_add] <= 4:
                        score_diff += 50
                    group_counts[g_add] += 1
                    current_score += score_diff
                    current_groups_bit[g_add // 64] |= uint64(1) << uint64(g_add % 64)
                    if current_score > best_score:
                        best_score = current_score
                        best_sol[:] = current_sol[:]
        else:
            if r_val < 0.02 * temp:
                current_sol[add_idx] = 0
                for j in range(3):
                    current_weights[j] -= weights[add_idx, j]
                score_diff = -values[add_idx]
                if 3 <= group_counts[g_add] <= 5:
                    score_diff -= 50
                group_counts[g_add] -= 1
                current_score += score_diff
                if group_counts[g_add] == 0:
                    current_groups_bit[g_add // 64] &= ~(
                        uint64(1) << uint64(g_add % 64)
                    )

    return best_score, best_sol


def load_constraints(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        capacities = list(map(int, lines[0].split(":")[1].split(",")))
        group_max = int(lines[1].split(":")[1])
        conf_line = [l for l in lines if l.startswith("conflicts:")][0]
        conf_data = conf_line.split(":")[1].strip().split(";")
        conflicts = [tuple(map(int, c.split(","))) for c in conf_data if c]
    return capacities, group_max, conflicts


def main():
    df = pd.read_csv("problem_data.csv")
    capacities, group_max, conflicts = load_constraints("constraints.txt")

    val_arr = df["value"].values.astype(np.int32)
    weight_arr = df[["weight0", "weight1", "weight2"]].values.astype(np.int32)
    group_arr = df["group_id"].values.astype(np.int32)
    conf_arr = np.array(conflicts, dtype=np.int32)
    cap_arr = np.array(capacities, dtype=np.int32)

    n_items, n_groups = len(df), df["group_id"].max() + 1

    start_time = time.time()
    score, _ = solve_knapsack_numba_sa(
        val_arr,
        weight_arr,
        cap_arr,
        group_arr,
        conf_arr,
        n_items,
        n_groups,
        group_max,
        max_iter=5000000,
    )
    elapsed = time.time() - start_time

    result_output = f"""
==================================================
Numba Local Search (Hill Climbing) Result
==================================================
Status: Completed (Max Iterations reached)
Total Objective: {score}
Execution Time: {elapsed:.4f} seconds
Method: Numba Optimized JIT Solver
==================================================
"""
    print(result_output)

    # ディレクトリ作成と保存
    output_dir = "result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "numba_results.txt")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(result_output)


if __name__ == "__main__":
    main()
