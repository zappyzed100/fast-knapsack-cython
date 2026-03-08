# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
import time

# --- 近傍探索 (Hill Climbing) ---
def solve_knapsack_sa(
    int[:] values,
    int[:, :] weights,
    int[:] capacities,
    int[:] item_groups,
    int[:, :] conflict_pairs,
    int n_items,
    int n_groups,
    int group_max,
    int max_iter = 2000000
):
    cdef unsigned long long[4] current_groups_bit = [0, 0, 0, 0]
    cdef unsigned long long[200][4] conflict_masks
    
    cdef int g, b, i
    for g in range(200):
        for b in range(4): conflict_masks[g][b] = 0
        
    cdef int g1, g2
    for i in range(conflict_pairs.shape[0]):
        g1 = conflict_pairs[i, 0]
        g2 = conflict_pairs[i, 1]
        conflict_masks[g1][g2 // 64] |= (1ULL << (g2 % 64))
        conflict_masks[g2][g1 // 64] |= (1ULL << (g1 % 64))

    cdef char[:] current_sol = np.zeros(n_items, dtype=np.int8)
    cdef int[:] group_counts = np.zeros(n_groups, dtype=np.int32)
    cdef int[:] current_weights = np.zeros(3, dtype=np.int32)
    cdef long current_score = 0
    cdef long best_score = 0
    cdef char[:] best_sol = np.zeros(n_items, dtype=np.int8)
    
    cdef int it, add_idx, rem_idx, rem_idx2, j, g_add, g_rem, g_rem2
    cdef float temp, r_val
    cdef long score_diff
    cdef int can_swap2

    rng = np.random.default_rng()
    cdef int[:] rand_add = rng.integers(0, n_items, size=max_iter, dtype=np.int32)
    cdef int[:] rand_rem = rng.integers(0, n_items, size=max_iter, dtype=np.int32)
    cdef float[:] rand_flt = rng.random(size=max_iter, dtype=np.float32)

    for it in range(max_iter):
        add_idx = rand_add[it]
        g_add = item_groups[add_idx]
        temp = 1.0 - (<float>it / max_iter)
        r_val = rand_flt[it]

        if current_sol[add_idx] == 0:
            if (current_weights[0] + weights[add_idx, 0] > capacities[0]) or \
               (current_weights[1] + weights[add_idx, 1] > capacities[1]) or \
               (current_weights[2] + weights[add_idx, 2] > capacities[2]):
                
                rem_idx = rand_rem[it]
                if current_sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    if (current_weights[0] - weights[rem_idx, 0] + weights[add_idx, 0] <= capacities[0]) and \
                       (current_weights[1] - weights[rem_idx, 1] + weights[add_idx, 1] <= capacities[1]) and \
                       (current_weights[2] - weights[rem_idx, 2] + weights[add_idx, 2] <= capacities[2]):
                        
                        current_groups_bit[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                        if not ((current_groups_bit[0] & conflict_masks[g_add][0]) or \
                                (current_groups_bit[1] & conflict_masks[g_add][1]) or \
                                (current_groups_bit[2] & conflict_masks[g_add][2]) or \
                                (current_groups_bit[3] & conflict_masks[g_add][3])):
                            
                            score_diff = values[add_idx] - values[rem_idx]
                            if group_counts[g_add] in [2, 3, 4]: score_diff += 50
                            if group_counts[g_rem] in [3, 4, 5]: score_diff -= 50
                            
                            if score_diff > 0 or r_val < 0.05 * temp:
                                current_sol[rem_idx] = 0
                                current_sol[add_idx] = 1
                                for j in range(3):
                                    current_weights[j] = current_weights[j] - weights[rem_idx, j] + weights[add_idx, j]
                                group_counts[g_rem] -= 1
                                group_counts[g_add] += 1
                                current_score += score_diff
                                current_groups_bit[g_add // 64] |= (1ULL << (g_add % 64))
                                if current_score > best_score:
                                    best_score = current_score
                                    best_sol[:] = current_sol[:]
                                continue
                        if group_counts[g_rem] > 0: current_groups_bit[g_rem // 64] |= (1ULL << (g_rem % 64))

                if r_val < 0.1: 
                    rem_idx = rand_rem[it]
                    rem_idx2 = (rem_idx + 7) % n_items
                    if current_sol[rem_idx] == 1 and current_sol[rem_idx2] == 1 and rem_idx != rem_idx2:
                        g_rem = item_groups[rem_idx]
                        g_rem2 = item_groups[rem_idx2]
                        can_swap2 = 1
                        for j in range(3):
                            if current_weights[j] - weights[rem_idx, j] - weights[rem_idx2, j] + weights[add_idx, j] > capacities[j]:
                                can_swap2 = 0; break
                        if can_swap2:
                            current_groups_bit[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                            current_groups_bit[g_rem2 // 64] &= ~(1ULL << (g_rem2 % 64))
                            if not ((current_groups_bit[0] & conflict_masks[g_add][0]) or \
                                    (current_groups_bit[1] & conflict_masks[g_add][1]) or \
                                    (current_groups_bit[2] & conflict_masks[g_add][2]) or \
                                    (current_groups_bit[3] & conflict_masks[g_add][3])):
                                score_diff = values[add_idx] - values[rem_idx] - values[rem_idx2]
                                if group_counts[g_add] in [2, 3, 4]: score_diff += 50
                                if score_diff > 0 or r_val < 0.01 * temp:
                                    current_sol[rem_idx] = 0
                                    current_sol[rem_idx2] = 0
                                    current_sol[add_idx] = 1
                                    for j in range(3):
                                        current_weights[j] = current_weights[j] - weights[rem_idx, j] - weights[rem_idx2, j] + weights[add_idx, j]
                                    group_counts[g_rem] -= 1
                                    group_counts[g_rem2] -= 1
                                    group_counts[g_add] += 1
                                    current_score += score_diff
                                    current_groups_bit[g_add // 64] |= (1ULL << (g_add % 64))
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_sol[:] = current_sol[:]
                                    continue
                            if group_counts[g_rem] > 0: current_groups_bit[g_rem // 64] |= (1ULL << (g_rem % 64))
                            if group_counts[g_rem2] > 0: current_groups_bit[g_rem2 // 64] |= (1ULL << (g_rem2 % 64))
                continue

            if group_counts[g_add] < group_max:
                if not ((current_groups_bit[0] & conflict_masks[g_add][0]) or \
                        (current_groups_bit[1] & conflict_masks[g_add][1]) or \
                        (current_groups_bit[2] & conflict_masks[g_add][2]) or \
                        (current_groups_bit[3] & conflict_masks[g_add][3])):
                    current_sol[add_idx] = 1
                    for j in range(3): current_weights[j] += weights[add_idx, j]
                    score_diff = values[add_idx]
                    if group_counts[g_add] in [2, 3, 4]: score_diff += 50
                    group_counts[g_add] += 1
                    current_score += score_diff
                    current_groups_bit[g_add // 64] |= (1ULL << (g_add % 64))
                    if current_score > best_score:
                        best_score = current_score
                        best_sol[:] = current_sol[:]
        else:
            if r_val < 0.02 * temp:
                current_sol[add_idx] = 0
                for j in range(3): current_weights[j] -= weights[add_idx, j]
                score_diff = -values[add_idx]
                if group_counts[g_add] in [3, 4, 5]: score_diff -= 50
                group_counts[g_add] -= 1
                current_score += score_diff
                if group_counts[g_add] == 0:
                    current_groups_bit[g_add // 64] &= ~(1ULL << (g_add % 64))

    return best_score, np.array(best_sol)

# --- 全探索 (DFS with Pruning) ---
def solve_knapsack_dfs(
    int[:] values,
    int[:, :] weights,
    int[:] capacities,
    int[:] item_groups,
    int[:, :] conflict_pairs,
    int n_items,
    int n_groups,
    int group_max
):
    cdef unsigned long long[4] cur_bits = [0, 0, 0, 0]
    cdef unsigned long long[200][4] conflict_masks
    cdef int i, j, b
    for i in range(200):
        for b in range(4): conflict_masks[i][b] = 0
    for i in range(conflict_pairs.shape[0]):
        conflict_masks[conflict_pairs[i, 0]][conflict_pairs[i, 1] // 64] |= (1ULL << (conflict_pairs[i, 1] % 64))
        conflict_masks[conflict_pairs[i, 1]][conflict_pairs[i, 0] // 64] |= (1ULL << (conflict_pairs[i, 0] % 64))

    cdef char[:] current_sol = np.zeros(n_items, dtype=np.int8)
    cdef int[:] group_counts = np.zeros(n_groups, dtype=np.int32)
    cdef int[:] cur_w = np.zeros(3, dtype=np.int32)
    cdef long best_score = 0
    cdef char[:] best_sol = np.zeros(n_items, dtype=np.int8)
    
    cdef double start_time = time.time()
    cdef long nodes_visited = 0
    cdef int timeout_flag = 0

    cdef long long[:] remain_max = np.zeros(n_items + 1, dtype=np.int64)
    for i in range(n_items - 1, -1, -1):
        remain_max[i] = remain_max[i+1] + values[i] + 50

    def backtrack(int idx, long current_score):
        nonlocal best_score, nodes_visited, timeout_flag
        if timeout_flag: return

        nodes_visited += 1
        if nodes_visited % 100000 == 0:
            if time.time() - start_time > 360:
                timeout_flag = 1
                return

        if current_score > best_score:
            best_score = current_score
            best_sol[:] = current_sol[:]

        if idx == n_items: return
        if current_score + remain_max[idx] <= best_score: return

        # 1. 入れる
        cdef int g = item_groups[idx]
        cdef int can_add = 1
        for j in range(3):
            if cur_w[j] + weights[idx, j] > capacities[j]:
                can_add = 0; break
        
        if can_add and group_counts[g] < group_max:
            if not ((cur_bits[0] & conflict_masks[g][0]) or \
                    (cur_bits[1] & conflict_masks[g][1]) or \
                    (cur_bits[2] & conflict_masks[g][2]) or \
                    (cur_bits[3] & conflict_masks[g][3])):
                
                current_sol[idx] = 1
                for j in range(3): cur_w[j] += weights[idx, j]
                
                group_counts[g] += 1
                cur_bits[g // 64] |= (1ULL << (g % 64))
                
                score_inc = values[idx]
                if group_counts[g] in [3, 4, 5]: score_inc += 50
                
                backtrack(idx + 1, current_score + score_inc)
                
                # 復元 (バグ修正: 個数が0になったときだけビットを下ろす)
                group_counts[g] -= 1
                if group_counts[g] == 0:
                    cur_bits[g // 64] &= ~(1ULL << (g % 64))
                for j in range(3): cur_w[j] -= weights[idx, j]
                current_sol[idx] = 0

        # 2. 入れない
        backtrack(idx + 1, current_score)

    backtrack(0, 0)
    return best_score, np.array(best_sol), (timeout_flag == 1)