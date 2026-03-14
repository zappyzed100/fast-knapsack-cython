import pandas as pd
import numpy as np
import time
import os
import argparse
import sys
import datetime
from numba import njit, uint64, int32, int8, float64, prange


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from utils.solution_eval import (
    evaluate_solution,
    format_solution_report,
    parse_constraints,
)


# ---------------------------------------------------------
# 1. 高速乱数 (Xorshift)
# ---------------------------------------------------------
@njit(inline="always")
def _xs_next(state):
    x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x


@njit(inline="always")
def _xs_double(state):
    return float64(_xs_next(state) & uint64(0xFFFFFFF)) / 268435456.0


# ---------------------------------------------------------
# 2. 高速 SA エンジン (Cython 版の移植)
# ---------------------------------------------------------
@njit(nogil=True)
def _run_sa_numba(
    sol,
    values,
    weights,
    capacities,
    item_groups,
    n_items,
    n_groups,
    group_max,
    iterations,
    conflict_masks,
    gen,
    seed,
):
    xs_state = np.array([uint64(seed + 1)], dtype=uint64)
    BONUS_VAL = 50.0

    g_bits = np.zeros(16, dtype=uint64)
    cur_w = np.zeros(3, dtype=int32)
    gc_buf = np.zeros(n_groups, dtype=int32)
    cur_val_sum = 0.0

    # 初期解の修復と状態の構築
    for j in range(n_items):
        if sol[j] == 1:
            g_idx = item_groups[j]
            conflict = False
            if 0 <= g_idx < 1024:
                for k in range(16):
                    if g_bits[k] & conflict_masks[g_idx, k]:
                        conflict = True
                        break

            if (
                conflict
                or cur_w[0] + weights[j, 0] > capacities[0]
                or cur_w[1] + weights[j, 1] > capacities[1]
                or cur_w[2] + weights[j, 2] > capacities[2]
                or (g_idx >= 0 and gc_buf[g_idx] >= group_max)
            ):
                sol[j] = 0
            else:
                cur_val_sum += float64(values[j])
                for k in range(3):
                    cur_w[k] += weights[j, k]
                if g_idx >= 0:
                    gc_buf[g_idx] += 1
                    g_bits[g_idx // 64] |= uint64(1) << (g_idx % 64)

    cur_bonus = 0.0
    for j in range(n_groups):
        if 3 <= gc_buf[j] <= 5:
            cur_bonus += BONUS_VAL

    best_total = cur_val_sum + cur_bonus
    best_sol_tmp = sol.copy()

    for it in range(iterations):
        temp = (1.0 - (float64(it) / iterations)) * (1.0 / (1.0 + gen * 0.1))
        add_idx = _xs_next(xs_state) % uint64(n_items)
        g_add = item_groups[add_idx]
        r = _xs_double(xs_state)

        if sol[add_idx] == 0:
            if (
                cur_w[0] + weights[add_idx, 0] <= capacities[0]
                and cur_w[1] + weights[add_idx, 1] <= capacities[1]
                and cur_w[2] + weights[add_idx, 2] <= capacities[2]
            ):

                conflict = False
                if 0 <= g_add < 1024:
                    for k in range(16):
                        if g_bits[k] & conflict_masks[g_add, k]:
                            conflict = True
                            break

                if not conflict and (g_add < 0 or gc_buf[g_add] < group_max):
                    bonus_diff = 0.0
                    if g_add >= 0:
                        if gc_buf[g_add] == 2:
                            bonus_diff = BONUS_VAL
                        elif gc_buf[g_add] == 5:
                            bonus_diff = -BONUS_VAL

                    diff = float64(values[add_idx]) + bonus_diff
                    if diff > 0 or (temp > 0 and r < np.exp(diff / (temp * 100.0))):
                        sol[add_idx] = 1
                        for k in range(3):
                            cur_w[k] += weights[add_idx, k]
                        if g_add >= 0:
                            gc_buf[g_add] += 1
                            g_bits[g_add // 64] |= uint64(1) << (g_add % 64)
                        cur_val_sum += float64(values[add_idx])
                        cur_bonus += bonus_diff
            else:
                rem_idx = _xs_next(xs_state) % uint64(n_items)
                if sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    if (
                        cur_w[0] - weights[rem_idx, 0] + weights[add_idx, 0]
                        <= capacities[0]
                        and cur_w[1] - weights[rem_idx, 1] + weights[add_idx, 1]
                        <= capacities[1]
                        and cur_w[2] - weights[rem_idx, 2] + weights[add_idx, 2]
                        <= capacities[2]
                    ):

                        if g_rem >= 0 and gc_buf[g_rem] == 1:
                            g_bits[g_rem // 64] &= ~(uint64(1) << (g_rem % 64))

                        conflict = False
                        if 0 <= g_add < 1024:
                            for k in range(16):
                                if g_bits[k] & conflict_masks[g_add, k]:
                                    conflict = True
                                    break

                        if not conflict and (
                            g_add < 0
                            or gc_buf[g_add]
                            < (group_max + (1 if g_add == g_rem else 0))
                        ):
                            bonus_diff = 0.0
                            if g_add != g_rem:
                                if g_rem >= 0:
                                    if gc_buf[g_rem] == 3:
                                        bonus_diff -= BONUS_VAL
                                    elif gc_buf[g_rem] == 6:
                                        bonus_diff += BONUS_VAL
                                if g_add >= 0:
                                    if gc_buf[g_add] == 2:
                                        bonus_diff += BONUS_VAL
                                    elif gc_buf[g_add] == 5:
                                        bonus_diff -= BONUS_VAL

                            diff = (
                                float64(values[add_idx] - values[rem_idx]) + bonus_diff
                            )
                            if diff > 0 or (
                                temp > 0 and r < np.exp(diff / (temp * 100.0))
                            ):
                                sol[rem_idx], sol[add_idx] = 0, 1
                                for k in range(3):
                                    cur_w[k] += (
                                        weights[add_idx, k] - weights[rem_idx, k]
                                    )
                                if g_add >= 0:
                                    gc_buf[g_add] += 1
                                if g_rem >= 0:
                                    gc_buf[g_rem] -= 1
                                if g_add >= 0:
                                    g_bits[g_add // 64] |= uint64(1) << (g_add % 64)
                                cur_val_sum += float64(
                                    values[add_idx] - values[rem_idx]
                                )
                                cur_bonus += bonus_diff
                                if cur_val_sum + cur_bonus > best_total:
                                    best_total = cur_val_sum + cur_bonus
                                    best_sol_tmp[:] = sol[:]
                                continue
                        if g_rem >= 0 and gc_buf[g_rem] >= 1:
                            g_bits[g_rem // 64] |= uint64(1) << (g_rem % 64)
        else:
            if r < 0.02 * temp:
                g_rem = item_groups[add_idx]
                bonus_diff = 0.0
                if g_rem >= 0:
                    if gc_buf[g_rem] == 3:
                        bonus_diff = -BONUS_VAL
                    elif gc_buf[g_rem] == 6:
                        bonus_diff = BONUS_VAL
                sol[add_idx] = 0
                for k in range(3):
                    cur_w[k] -= weights[add_idx, k]
                if g_rem >= 0:
                    gc_buf[g_rem] -= 1
                    if gc_buf[g_rem] == 0:
                        g_bits[g_rem // 64] &= ~(uint64(1) << (g_rem % 64))
                cur_val_sum -= float64(values[add_idx])
                cur_bonus += bonus_diff

        if cur_val_sum + cur_bonus > best_total:
            best_total = cur_val_sum + cur_bonus
            best_sol_tmp[:] = sol[:]

    return best_total, best_sol_tmp


# ---------------------------------------------------------
# 3. 進化計算・交叉ロジック
# ---------------------------------------------------------
@njit
def _init_masks_numba(confs):
    masks = np.zeros((1024, 16), dtype=uint64)
    for i in range(confs.shape[0]):
        u, v = confs[i, 0], confs[i, 1]
        if u < 1024 and v < 1024:
            masks[u, v // 64] |= uint64(1) << (v % 64)
            masks[v, u // 64] |= uint64(1) << (u % 64)
    return masks


@njit
def _greedy_crossover_numba(
    p1,
    p2,
    item_groups,
    weights,
    capacities,
    conflict_masks,
    n_items,
    n_groups,
    group_max,
    s_idx_desc,
):
    child = np.zeros(n_items, dtype=int8)
    cw = np.zeros(3, dtype=int32)
    gc = np.zeros(n_groups, dtype=int32)
    gbits = np.zeros(16, dtype=uint64)
    for k in range(n_items):
        idx = s_idx_desc[k]
        if p1[idx] == 1 or p2[idx] == 1:
            gid = item_groups[idx]
            conflict = False
            if 0 <= gid < 1024:
                for m in range(16):
                    if gbits[m] & conflict_masks[gid, m]:
                        conflict = True
                        break
            if conflict:
                continue
            if (
                cw[0] + weights[idx, 0] <= capacities[0]
                and cw[1] + weights[idx, 1] <= capacities[1]
                and cw[2] + weights[idx, 2] <= capacities[2]
                and (gid < 0 or gc[gid] < group_max)
            ):
                child[idx] = 1
                for m in range(3):
                    cw[m] += weights[idx, m]
                if gid >= 0:
                    gc[gid] += 1
                    gbits[gid // 64] |= uint64(1) << (gid % 64)
    return child


@njit(parallel=True)
def solve_knapsack_evolution_numba(
    initial_sol,
    values,
    weights,
    capacities,
    item_groups,
    conflict_pairs,
    n_items,
    n_groups,
    group_max,
    pop_size=20,
    rand_add_size=20,
    crossover_size=50,
    max_generations=1000,
    iter_per_ind=1000000,
    patience=10,
):
    total_pop = pop_size + rand_add_size + crossover_size
    conflict_masks = _init_masks_numba(conflict_pairs)
    densities = values.astype(float64) / (
        1.0 + weights[:, 0] + weights[:, 1] + weights[:, 2]
    )
    s_idx_desc = np.argsort(-densities).astype(int32)

    pops = np.zeros((total_pop, n_items), dtype=int8)
    temp_pops = np.zeros((pop_size, n_items), dtype=int8)
    scores = np.zeros(total_pop, dtype=float64)
    temp_scores = np.zeros(pop_size, dtype=float64)
    if np.any(initial_sol):
        for i in range(pop_size):
            pops[i] = initial_sol.copy()

    last_best = -1.0
    no_improvement = 0
    base_seed = 42

    for gen in range(max_generations):
        for i in prange(pop_size, pop_size + rand_add_size):
            pops[i] = np.zeros(n_items, dtype=int8)
            s, sol = _run_sa_numba(
                pops[i],
                values,
                weights,
                capacities,
                item_groups,
                n_items,
                n_groups,
                group_max,
                iter_per_ind,
                conflict_masks,
                gen,
                base_seed ^ (i * 777 + gen),
            )
            scores[i] = s
            pops[i] = sol

        cx1 = np.random.randint(0, pop_size + rand_add_size, crossover_size).astype(
            int32
        )
        cx2 = np.random.randint(0, pop_size + rand_add_size, crossover_size).astype(
            int32
        )
        for i in prange(crossover_size):
            pop_idx = pop_size + rand_add_size + i
            pops[pop_idx] = _greedy_crossover_numba(
                pops[cx1[i]],
                pops[cx2[i]],
                item_groups,
                weights,
                capacities,
                conflict_masks,
                n_items,
                n_groups,
                group_max,
                s_idx_desc,
            )

        for i in prange(total_pop):
            s, sol = _run_sa_numba(
                pops[i],
                values,
                weights,
                capacities,
                item_groups,
                n_items,
                n_groups,
                group_max,
                iter_per_ind,
                conflict_masks,
                gen,
                base_seed ^ (i * 123 + gen),
            )
            scores[i] = s
            pops[i] = sol

        sorted_idx = np.argsort(-scores)
        if scores[sorted_idx[0]] > last_best:
            last_best = scores[sorted_idx[0]]
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

        for i in range(pop_size):
            temp_pops[i] = pops[sorted_idx[i]].copy()
            temp_scores[i] = scores[sorted_idx[i]]

        for i in range(pop_size):
            pops[i] = temp_pops[i]
            scores[i] = temp_scores[i]

    return last_best, pops[0]


def save_result(
    solver_name,
    elapsed,
    status,
    evaluation,
    objective_value=None,
    full_output=True,
):
    result_dir = os.path.join(PROJECT_ROOT, "results", "runs")
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, "numba_results.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_text = format_solution_report(
        solver_name=solver_name,
        elapsed_sec=elapsed,
        status=status,
        evaluation=evaluation,
        objective_value=objective_value,
        timestamp=timestamp,
        full_output=full_output,
    )
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(result_text)
    print(result_text)


class NumbaBenchmarker:
    def __init__(self):
        self.csv_path = os.path.join(PROJECT_ROOT, "data", "problem_data.csv")
        self.constraints_path = os.path.join(PROJECT_ROOT, "data", "constraints.txt")

    def run(self, iterations=10000000, patience=10, full_output=True):
        df = pd.read_csv(self.csv_path)
        caps, confs = parse_constraints(self.constraints_path)
        v, w, g = (
            df["value"].values.astype(np.int32),
            df[["weight0", "weight1", "weight2"]].values.astype(np.int32),
            df["group_id"].values.astype(np.int32),
        )
        n_items, n_groups = len(df), int(g.max() + 1)

        # 1. 単体 SA
        st1 = time.perf_counter()
        mask = _init_masks_numba(confs)
        score_sa, sol_sa = _run_sa_numba(
            np.zeros(n_items, dtype=np.int8),
            v,
            w,
            caps,
            g,
            n_items,
            n_groups,
            10,
            iterations,
            mask,
            0,
            42,
        )
        el1 = time.perf_counter() - st1
        eval1 = evaluate_solution(sol_sa, v, w, caps, g, confs, 10)
        status1 = "SATISFIED" if eval1["is_valid"] else "INFEASIBLE"
        save_result(
            "numba_single_sa",
            el1,
            status1,
            eval1,
            objective_value=int(score_sa),
            full_output=full_output,
        )

        # 2. 進化計算
        st2 = time.perf_counter()
        score_evo, sol_evo = solve_knapsack_evolution_numba(
            sol_sa,
            v,
            w,
            caps,
            g,
            confs,
            n_items,
            n_groups,
            10,
            pop_size=20,
            rand_add_size=20,
            crossover_size=50,
            max_generations=1000,
            iter_per_ind=iterations,
            patience=patience,
        )
        el2 = time.perf_counter() - st2
        eval2 = evaluate_solution(sol_evo, v, w, caps, g, confs, 10)
        status2 = "SATISFIED" if eval2["is_valid"] else "INFEASIBLE"
        save_result(
            "numba_hybrid_evolution",
            el2,
            status2,
            eval2,
            objective_value=int(score_evo),
            full_output=full_output,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=10000000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--full-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Output full selected item/group lists (default: true). Use --no-full-output for preview mode.",
    )
    args = parser.parse_args()
    NumbaBenchmarker().run(
        iterations=args.iter,
        patience=args.patience,
        full_output=args.full_output,
    )
