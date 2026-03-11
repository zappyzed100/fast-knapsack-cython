# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

def solve_knapsack_sa(
    int[:] values,
    int[:, :] weights,
    int[:] capacities,
    int[:] item_groups,
    int[:, :] conflict_pairs,
    int n_items,
    int n_groups,
    int group_max,
    int max_iter = 10000000
):
    cdef int i, j, k, g_idx, g1, g2
    cdef int add_idx, rem_idx, g_add, g_rem
    cdef int it, over_weight, conflict
    cdef double temp, r_val_d
    cdef double val_diff, old_total_bonus, new_total_bonus, diff
    cdef double current_base_score = 0 
    cdef double best_score = 0
    cdef double current_score_with_bonus = 0
    cdef int BONUS_VAL = 50

    cdef unsigned long long[:, :] conflict_masks = np.zeros((1000, 16), dtype=np.uint64)
    cdef unsigned long long[16] current_groups_bit
    for i in range(16): current_groups_bit[i] = 0

    for i in range(conflict_pairs.shape[0]):
        g1 = conflict_pairs[i, 0]
        g2 = conflict_pairs[i, 1]
        if g1 < 1000 and g2 < 1000:
            conflict_masks[g1, g2 // 64] |= (1ULL << (g2 % 64))
            conflict_masks[g2, g1 // 64] |= (1ULL << (g1 % 64))

    cdef char[:] current_sol = np.zeros(n_items, dtype=np.int8)
    cdef int[:] group_counts = np.zeros(n_groups, dtype=np.int32)
    cdef int[:] current_weights = np.zeros(3, dtype=np.int32)
    cdef char[:] best_sol = np.zeros(n_items, dtype=np.int8)
    
    cdef int[:] rand_add = np.random.randint(0, n_items, size=max_iter).astype(np.int32)
    cdef int[:] rand_rem = np.random.randint(0, n_items, size=max_iter).astype(np.int32)
    cdef double[:] rand_flt = np.random.random(size=max_iter).astype(np.float64)

    for it in range(max_iter):
        add_idx = rand_add[it]
        g_add = item_groups[add_idx]
        temp = 1.0 - (<double>it / <double>max_iter)
        r_val_d = rand_flt[it]

        if current_sol[add_idx] == 0:
            over_weight = 0
            for j in range(3):
                if current_weights[j] + weights[add_idx, j] > capacities[j]:
                    over_weight = 1
                    break

            if over_weight == 1:
                rem_idx = rand_rem[it]
                if current_sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    if (current_weights[0] - weights[rem_idx, 0] + weights[add_idx, 0] <= capacities[0]) and \
                       (current_weights[1] - weights[rem_idx, 1] + weights[add_idx, 1] <= capacities[1]) and \
                       (current_weights[2] - weights[rem_idx, 2] + weights[add_idx, 2] <= capacities[2]):
                        
                        current_groups_bit[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                        conflict = 0
                        for k in range(16):
                            if current_groups_bit[k] & conflict_masks[g_add, k]:
                                conflict = 1
                                break
                        
                        if conflict == 0:
                            # 旧ボーナス計算
                            old_total_bonus = 0
                            for g_idx in range(n_groups):
                                if group_counts[g_idx] >= 3 and group_counts[g_idx] <= 5: old_total_bonus += BONUS_VAL
                            
                            # 新ボーナス計算
                            group_counts[g_add] += 1
                            group_counts[g_rem] -= 1
                            new_total_bonus = 0
                            for g_idx in range(n_groups):
                                if group_counts[g_idx] >= 3 and group_counts[g_idx] <= 5: new_total_bonus += BONUS_VAL
                            group_counts[g_add] -= 1
                            group_counts[g_rem] += 1

                            val_diff = <double>(values[add_idx] - values[rem_idx])
                            diff = (val_diff + new_total_bonus) - old_total_bonus

                            if diff > 0 or r_val_d < 0.05 * temp:
                                current_sol[rem_idx] = 0
                                current_sol[add_idx] = 1
                                for j in range(3):
                                    current_weights[j] += weights[add_idx, j] - weights[rem_idx, j]
                                group_counts[g_rem] -= 1
                                group_counts[g_add] += 1
                                current_base_score += val_diff
                                if group_counts[g_rem] > 0: current_groups_bit[g_rem // 64] |= (1ULL << (g_rem % 64))
                                current_groups_bit[g_add // 64] |= (1ULL << (g_add % 64))
                                if (current_base_score + new_total_bonus) > best_score:
                                    best_score = current_base_score + new_total_bonus
                                    best_sol[:] = current_sol[:]
                                continue
                        if group_counts[g_rem] > 0: current_groups_bit[g_rem // 64] |= (1ULL << (g_rem % 64))

            elif group_counts[g_add] < group_max:
                conflict = 0
                for k in range(16):
                    if current_groups_bit[k] & conflict_masks[g_add, k]:
                        conflict = 1
                        break
                
                if conflict == 0:
                    old_total_bonus = 0
                    for g_idx in range(n_groups):
                        if group_counts[g_idx] >= 3 and group_counts[g_idx] <= 5: old_total_bonus += BONUS_VAL
                    
                    group_counts[g_add] += 1
                    new_total_bonus = 0
                    for g_idx in range(n_groups):
                        if group_counts[g_idx] >= 3 and group_counts[g_idx] <= 5: new_total_bonus += BONUS_VAL
                    group_counts[g_add] -= 1

                    val_diff = <double>values[add_idx]
                    diff = (val_diff + new_total_bonus) - old_total_bonus
                    
                    if diff > 0 or r_val_d < 0.1 * temp:
                        current_sol[add_idx] = 1
                        for j in range(3): current_weights[j] += weights[add_idx, j]
                        group_counts[g_add] += 1
                        current_base_score += val_diff
                        current_groups_bit[g_add // 64] |= (1ULL << (g_add % 64))
                        if (current_base_score + new_total_bonus) > best_score:
                            best_score = current_base_score + new_total_bonus
                            best_sol[:] = current_sol[:]
        else:
            if r_val_d < 0.02 * temp:
                g_rem = item_groups[add_idx]
                old_total_bonus = 0
                for g_idx in range(n_groups):
                    if group_counts[g_idx] >= 3 and group_counts[g_idx] <= 5: old_total_bonus += BONUS_VAL
                
                group_counts[g_rem] -= 1
                new_total_bonus = 0
                for g_idx in range(n_groups):
                    if group_counts[g_idx] >= 3 and group_counts[g_idx] <= 5: new_total_bonus += BONUS_VAL
                
                val_diff = <double>(-values[add_idx])
                current_sol[add_idx] = 0
                for j in range(3): current_weights[j] -= weights[add_idx, j]
                current_base_score += val_diff
                if group_counts[g_rem] == 0: current_groups_bit[g_rem // 64] &= ~(1ULL << (g_rem % 64))

    return best_score, np.array(best_sol)