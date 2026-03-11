# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time as c_time

# ---------------------------------------------------------
# 1. Cレベルのヘルパー関数 (nogil)
# ---------------------------------------------------------

cdef double _calc_b_delta(int ca, int cb, int da, int db, int bv) noexcept nogil:
    cdef double b = 0
    if 3 <= ca <= 5: b -= bv
    if 3 <= (ca + da) <= 5: b += bv
    if 3 <= cb <= 5: b -= bv
    if 3 <= (cb + db) <= 5: b += bv
    return b

cdef double _calc_b_single(int c, int d, int bv) noexcept nogil:
    cdef double b = 0
    if 3 <= c <= 5: b -= bv
    if 3 <= (c + d) <= 5: b += bv
    return b

cdef void _repair_individual(char[:] sol, int[:] values, int[:, :] weights, int[:] capacities, 
                             int[:] item_groups, int group_max, int n_items, int n_groups, 
                             int[:] s_idx_desc, int[:] s_idx_asc) noexcept nogil:
    cdef int k, idx, gid, j
    cdef int g_counts[1000] 
    cdef int cur_w[3]
    for j in range(3): cur_w[j] = 0
    for j in range(1000): g_counts[j] = 0
    
    for k in range(n_items):
        if sol[k] == 1:
            gid = item_groups[k]
            for j in range(3): cur_w[j] += weights[k, j]
            if gid < 1000: g_counts[gid] += 1

    # 重過剰・グループ数過剰を削除
    for k in range(n_items):
        idx = s_idx_asc[k]
        if sol[idx] == 1:
            gid = item_groups[idx]
            if cur_w[0] > capacities[0] or cur_w[1] > capacities[1] or cur_w[2] > capacities[2] or (gid < 1000 and g_counts[gid] > group_max):
                sol[idx] = 0
                for j in range(3): cur_w[j] -= weights[idx, j]
                if gid < 1000: g_counts[gid] -= 1

    # 空きに充填
    for k in range(n_items):
        idx = s_idx_desc[k]
        if sol[idx] == 0:
            gid = item_groups[idx]
            if cur_w[0] + weights[idx,0] <= capacities[0] and \
               cur_w[1] + weights[idx,1] <= capacities[1] and \
               cur_w[2] + weights[idx,2] <= capacities[2] and (gid < 1000 and g_counts[gid] < group_max):
                sol[idx] = 1
                for j in range(3): cur_w[j] += weights[idx, j]
                if gid < 1000: g_counts[gid] += 1

# ---------------------------------------------------------
# 2. SAコアエンジン (Hard Constraint)
# ---------------------------------------------------------

cdef void _run_advanced_sa(
    int island_type, char[:] sol, double[:, :] scores_base, double[:, :] scores_bonus, int isol_idx, int ind_idx,
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups, int n_items, int n_groups, int group_max,
    int iterations, int[:] g_counts, int[:] cur_w, int[:] conf_ptr, int[:] conf_data,
    unsigned long long[:, :] conflict_masks, int gen
) noexcept nogil:
    cdef int it, j, k, add_idx, rem_idx, conflict, gid, g_add, g_rem
    cdef double r, eval_diff, val_diff, b_diff, temp, dens_add, dens_rem
    cdef double current_base = 0, current_bonus = 0
    cdef int b_val = 50
    cdef unsigned long long g_bits[16]
    
    for j in range(16): g_bits[j] = 0
    for j in range(3): cur_w[j] = 0
    for j in range(n_groups): g_counts[j] = 0
    for j in range(n_items):
        if sol[j] == 1:
            gid = item_groups[j]
            current_base += values[j]; g_counts[gid] += 1
            for k in range(3): cur_w[k] += weights[j, k]
            if gid < 1024: g_bits[gid // 64] |= (1ULL << (gid % 64))
    for j in range(n_groups):
        if 3 <= g_counts[j] <= 5: current_bonus += b_val

    for it in range(iterations):
        temp = (1.0 - (<double>it / iterations)) * (2.0 / (2.0 + gen * 0.15))
        add_idx = rand() % n_items; g_add = item_groups[add_idx]; r = <double>rand() / RAND_MAX

        if sol[add_idx] == 0:
            conflict = 0
            if g_add < 1024:
                for k in range(16):
                    if g_bits[k] & conflict_masks[g_add, k]: conflict = 1; break
            if conflict == 0:
                for k in range(conf_ptr[add_idx], conf_ptr[add_idx+1]):
                    if sol[conf_data[k]] == 1: conflict = 1; break
            if conflict == 1: continue

            rem_idx = rand() % n_items if r < 0.5 else -1
            if rem_idx != -1 and sol[rem_idx] == 0: rem_idx = -1

            # Hard Constraint Check
            if rem_idx != -1:
                if (cur_w[0] + weights[add_idx,0] - weights[rem_idx,0] > capacities[0]) or \
                   (cur_w[1] + weights[add_idx,1] - weights[rem_idx,1] > capacities[1]) or \
                   (cur_w[2] + weights[add_idx,2] - weights[rem_idx,2] > capacities[2]) or \
                   (g_add != item_groups[rem_idx] and g_counts[g_add] >= group_max):
                    continue
            else:
                if (cur_w[0] + weights[add_idx,0] > capacities[0]) or \
                   (cur_w[1] + weights[add_idx,1] > capacities[1]) or \
                   (cur_w[2] + weights[add_idx,2] > capacities[2]) or \
                   (g_counts[g_add] >= group_max):
                    continue

            val_diff = values[add_idx] - (values[rem_idx] if rem_idx != -1 else 0)
            b_diff = _calc_b_delta(g_counts[g_add], (g_counts[item_groups[rem_idx]] if rem_idx != -1 else 0), 1, (-1 if rem_idx != -1 else 0), b_val)
            
            # 各島の特殊評価
            if island_type == 0: eval_diff = val_diff
            elif island_type == 1: eval_diff = b_diff
            elif island_type == 3: # Slack
                eval_diff = val_diff * 0.1 + b_diff * 0.1 + (weights[rem_idx,0] if rem_idx != -1 else 0) - weights[add_idx,0]
            elif island_type == 5: # Density
                dens_add = values[add_idx] / (1.0 + weights[add_idx,0] + weights[add_idx,1] + weights[add_idx,2])
                dens_rem = (values[rem_idx] / (1.0 + weights[rem_idx,0] + weights[rem_idx,1] + weights[rem_idx,2])) if rem_idx != -1 else 0
                eval_diff = (dens_add - dens_rem) * 1000.0
            else: eval_diff = val_diff + b_diff

            if eval_diff > 0 or r < 0.05 * temp:
                if rem_idx != -1: 
                    sol[rem_idx] = 0; g_counts[item_groups[rem_idx]] -= 1
                    for j in range(3): cur_w[j] -= weights[rem_idx, j]
                sol[add_idx] = 1; g_counts[g_add] += 1
                for j in range(3): cur_w[j] += weights[add_idx, j]
                current_base += val_diff; current_bonus += b_diff
                if g_add < 1024: g_bits[g_add // 64] |= (1ULL << (g_add % 64))
        else:
            if r < 0.03 * temp:
                gid = item_groups[add_idx]; sol[add_idx] = 0; g_counts[gid] -= 1
                for j in range(3): cur_w[j] -= weights[add_idx, j]
                current_base -= values[add_idx]
                current_bonus += _calc_b_single(g_counts[gid]+1, -1, b_val)
                if g_counts[gid] == 0 and gid < 1024: g_bits[gid // 64] &= ~(1ULL << (gid % 64))

    scores_base[isol_idx, ind_idx] = current_base
    scores_bonus[isol_idx, ind_idx] = current_bonus

# ---------------------------------------------------------
# 3. 進化・初期化
# ---------------------------------------------------------

def _initialize_populations(char[:, :, :] pops, int[:, :] weights, int[:] capacities, int n_isl, int p_size, int n_items):
    cdef int i, j, k, idx, cw0, cw1, cw2
    for i in range(n_isl):
        for j in range(p_size):
            indices = np.random.permutation(n_items).astype(np.int32)
            cw0=0; cw1=0; cw2=0
            for k in range(n_items):
                idx = indices[k]
                if cw0 + weights[idx,0] <= capacities[0] and cw1 + weights[idx,1] <= capacities[1] and cw2 + weights[idx,2] <= capacities[2]:
                    pops[i, j, idx] = 1
                    cw0 += weights[idx,0]; cw1 += weights[idx,1]; cw2 += weights[idx,2]

def _perform_advanced_evolution(char[:, :, :] pops, double[:, :] s_base, double[:, :] s_bonus, 
                                char[:, :] hof_sols, double[:] hof_scores,
                                int n_isl, int p_size, int e_size, 
                                int[:] item_groups, int n_groups, int group_max,
                                int[:, :] weights, int[:] capacities, int[:] values,
                                int[:] s_idx_desc, int[:] s_idx_asc):
    cdef int i, j, k, target_idx, m_idx, best_m_idx, n_items = pops.shape[2]
    cdef double r, best_diff, diff, cur_score
    
    rank_matrix = np.zeros((n_isl, p_size), dtype=np.int32)
    for i in range(n_isl):
        combined = np.array(s_base[i]) + np.array(s_bonus[i])
        rank_matrix[i] = np.argsort(np.argsort(-combined))
    
    elite_indices = np.argsort(np.prod(rank_matrix + 1, axis=0)).astype(np.int32)
    cur_score = s_base[0, elite_indices[0]] + s_bonus[0, elite_indices[0]]
    if cur_score > hof_scores[0]:
        hof_scores[0] = cur_score; hof_sols[0, :] = pops[0, elite_indices[0], :]

    new_pops = np.zeros_like(pops)
    cdef char[:, :, :] next_pops = new_pops

    for i in range(n_isl):
        p_list = np.where(rank_matrix[i] < (p_size // 5))[0].astype(np.int32)
        for j in range(p_size):
            r = np.random.rand()
            target_idx = p_list[np.random.randint(0, len(p_list))]
            next_pops[i, j, :] = pops[i, target_idx, :]

            if r < 0.15: # 改良
                _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc)
            elif r < 0.55: # 30倍変異
                best_diff = -1e9; best_m_idx = -1
                for _ in range(30):
                    m_idx = np.random.randint(0, n_items)
                    diff = (1 - 2*next_pops[i, j, m_idx]) * values[m_idx]
                    if diff > best_diff:
                        best_diff = diff; best_m_idx = m_idx
                if best_m_idx != -1: next_pops[i, j, best_m_idx] = 1 - next_pops[i, j, best_m_idx]
                _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc)
            else: # 通常変異
                k = np.random.randint(0, n_items)
                next_pops[i, j, k] = 1 - next_pops[i, j, k]
                _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc)
    pops[:, :, :] = next_pops[:, :, :]

# ---------------------------------------------------------
# 4. メインソルバー
# ---------------------------------------------------------

def solve_knapsack_sa_parallel(
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups,
    int[:, :] conflict_pairs, int n_items, int n_groups, int group_max,
    int max_iter = 10000000
):
    cdef int n_islands = 6, pop_size = 50, elite_size = 20, max_generations = 150
    cdef int gen, isol, ind, i, u, v
    cdef unsigned int base_seed = <unsigned int>c_time(NULL)

    densities = np.array(values) / (1.0 + np.sum(np.array(weights), axis=1))
    cdef int[:] s_idx_desc = np.argsort(-densities).astype(np.int32)
    cdef int[:] s_idx_asc = np.argsort(densities).astype(np.int32)

    cdef char[:, :, :] populations = np.zeros((n_islands, pop_size, n_items), dtype=np.int8)
    cdef double[:, :] scores_base = np.zeros((n_islands, pop_size), dtype=np.float64)
    cdef double[:, :] scores_bonus = np.zeros((n_islands, pop_size), dtype=np.float64)
    cdef int[:, :, :] island_g_counts = np.zeros((n_islands, pop_size, n_groups), dtype=np.int32)
    cdef int[:, :, :] island_cur_w = np.zeros((n_islands, pop_size, 3), dtype=np.int32)
    cdef char[:, :] hof_sols = np.zeros((3, n_items), dtype=np.int8)
    cdef double[:] hof_scores = np.zeros(3, dtype=np.float64)

    # 衝突構造
    cdef int[:] item_conf_ptr = np.zeros(n_items + 1, dtype=np.int32)
    for i in range(conflict_pairs.shape[0]):
        item_conf_ptr[conflict_pairs[i, 0] + 1] += 1
        item_conf_ptr[conflict_pairs[i, 1] + 1] += 1
    for i in range(n_items): item_conf_ptr[i+1] += item_conf_ptr[i]
    cdef int[:] item_conf_data = np.zeros(item_conf_ptr[n_items], dtype=np.int32)
    cdef int[:] item_conf_cur = item_conf_ptr.copy()
    for i in range(conflict_pairs.shape[0]):
        u = conflict_pairs[i, 0]; v = conflict_pairs[i, 1]
        item_conf_data[item_conf_cur[u]] = v; item_conf_cur[u] += 1
        item_conf_data[item_conf_cur[v]] = u; item_conf_cur[v] += 1
    cdef unsigned long long[:, :] conflict_masks = np.zeros((1024, 16), dtype=np.uint64)
    for i in range(conflict_pairs.shape[0]):
        u = conflict_pairs[i, 0]; v = conflict_pairs[i, 1]
        if u < 1024 and v < 1024:
            conflict_masks[u, v // 64] |= (1ULL << (v % 64)); conflict_masks[v, u // 64] |= (1ULL << (u % 64))

    _initialize_populations(populations, weights, capacities, n_islands, pop_size, n_items)

    for gen in range(max_generations):
        for isol in prange(n_islands, nogil=True):
            srand(base_seed ^ <unsigned int>(isol * 888 + gen))
            for ind in range(pop_size):
                _run_advanced_sa(
                    isol, populations[isol, ind, :], scores_base, scores_bonus, isol, ind,
                    values, weights, capacities, item_groups, n_items, n_groups, group_max,
                    30000, island_g_counts[isol, ind, :], island_cur_w[isol, ind, :],
                    item_conf_ptr, item_conf_data, conflict_masks, gen
                )

        _perform_advanced_evolution(populations, scores_base, scores_bonus, hof_sols, hof_scores, 
                                   n_islands, pop_size, elite_size, item_groups, n_groups, 
                                   group_max, weights, capacities, values, s_idx_desc, s_idx_asc)

    total_scores = np.array(scores_base) + np.array(scores_bonus)
    best_tup = np.unravel_index(np.argmax(total_scores), (n_islands, pop_size))
    # 最終チェック
    cdef int b_isl = best_tup[0], b_ind = best_tup[1]
    _repair_individual(populations[b_isl, b_ind, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc)
    
    return np.max(total_scores), np.array(populations[b_isl, b_ind, :])