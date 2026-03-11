# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time as c_time
from libc.string cimport memset

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
                             int[:] s_idx_desc, int[:] s_idx_asc, 
                             unsigned long long[:, :] conflict_masks) noexcept nogil:
    cdef int k, idx, gid, j, m, conflict
    cdef int g_counts[2000]
    cdef int cur_w[3]
    cdef unsigned long long g_bits[16]
    cdef char val
    
    memset(cur_w, 0, sizeof(cur_w))
    memset(g_counts, 0, sizeof(g_counts))
    memset(g_bits, 0, sizeof(g_bits))
    
    for k in range(n_items):
        val = sol[k]
        if val == 1:
            gid = item_groups[k]
            for j in range(3): cur_w[j] += weights[k, j]
            if 0 <= gid < 2000: 
                g_counts[gid] += 1
                if gid < 1024: g_bits[gid // 64] |= (1ULL << (gid % 64))

    for k in range(n_items):
        idx = s_idx_asc[k]
        if sol[idx] == 1:
            gid = item_groups[idx]
            conflict = 0
            if 0 <= gid < 1024:
                for m in range(16):
                    if g_bits[m] & conflict_masks[gid, m]:
                        conflict = 1
                        break
            
            if cur_w[0] > capacities[0] or cur_w[1] > capacities[1] or cur_w[2] > capacities[2] or \
               (0 <= gid < 2000 and g_counts[gid] > group_max) or conflict:
                sol[idx] = 0
                for j in range(3): cur_w[j] -= weights[idx, j]
                if 0 <= gid < 2000:
                    g_counts[gid] -= 1
                    if g_counts[gid] == 0 and gid < 1024:
                        g_bits[gid // 64] &= ~(1ULL << (gid % 64))

    for k in range(n_items):
        idx = s_idx_desc[k]
        if sol[idx] == 0:
            gid = item_groups[idx]
            conflict = 0
            if 0 <= gid < 1024:
                for m in range(16):
                    if g_bits[m] & conflict_masks[gid, m]:
                        conflict = 1
                        break
            if conflict: continue

            if cur_w[0] + weights[idx,0] <= capacities[0] and \
               cur_w[1] + weights[idx,1] <= capacities[1] and \
               cur_w[2] + weights[idx,2] <= capacities[2] and \
               (gid < 0 or (gid < 2000 and g_counts[gid] < group_max)):
                sol[idx] = 1
                for j in range(3): cur_w[j] += weights[idx, j]
                if 0 <= gid < 2000:
                    g_counts[gid] += 1
                    if gid < 1024: g_bits[gid // 64] |= (1ULL << (gid % 64))

cdef void _run_advanced_sa(
    int island_type, char[:] sol, double[:, :] scores_base, double[:, :] scores_bonus, int isol_idx, int ind_idx,
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups, int n_items, int n_groups, int group_max,
    int iterations, int[:] g_counts_ptr, int[:] cur_w_ptr, int[:] conf_ptr, int[:] conf_data,
    unsigned long long[:, :] conflict_masks, int gen, double w_slack, double w_density
) noexcept nogil:
    cdef int it, j, k, add_idx, rem_idx, conflict, gid, g_add, g_rem
    cdef double r, eval_diff, val_diff, b_diff, temp, dens_add, dens_rem
    cdef double current_base = 0, current_bonus = 0
    cdef int b_val = 50
    cdef unsigned long long g_bits[16]
    
    memset(g_bits, 0, sizeof(g_bits))
    for j in range(3): cur_w_ptr[j] = 0
    for j in range(n_groups): g_counts_ptr[j] = 0
    
    for j in range(n_items):
        if sol[j] == 1:
            gid = item_groups[j]
            current_base += values[j]
            if gid >= 0: g_counts_ptr[gid] += 1
            for k in range(3): cur_w_ptr[k] += weights[j, k]
            if 0 <= gid < 1024: g_bits[gid // 64] |= (1ULL << (gid % 64))
            
    for j in range(n_groups):
        if 3 <= g_counts_ptr[j] <= 5: current_bonus += b_val

    for it in range(iterations):
        temp = (1.0 - (<double>it / iterations)) * (2.0 / (2.0 + gen * 0.15))
        add_idx = rand() % n_items; g_add = item_groups[add_idx]; r = <double>rand() / RAND_MAX

        if sol[add_idx] == 0:
            conflict = 0
            if 0 <= g_add < 1024:
                for k in range(16):
                    if g_bits[k] & conflict_masks[g_add, k]: conflict = 1; break
            if conflict: continue

            rem_idx = rand() % n_items if r < 0.5 else -1
            if rem_idx != -1 and sol[rem_idx] == 0: rem_idx = -1

            if rem_idx != -1:
                g_rem = item_groups[rem_idx]
                if (cur_w_ptr[0] + weights[add_idx,0] - weights[rem_idx,0] > capacities[0]) or \
                   (cur_w_ptr[1] + weights[add_idx,1] - weights[rem_idx,1] > capacities[1]) or \
                   (cur_w_ptr[2] + weights[add_idx,2] - weights[rem_idx,2] > capacities[2]) or \
                   (g_add != g_rem and g_add >= 0 and g_counts_ptr[g_add] >= group_max):
                    continue
            else:
                if (cur_w_ptr[0] + weights[add_idx,0] > capacities[0]) or \
                   (cur_w_ptr[1] + weights[add_idx,1] > capacities[1]) or \
                   (cur_w_ptr[2] + weights[add_idx,2] > capacities[2]) or \
                   (g_add >= 0 and g_counts_ptr[g_add] >= group_max):
                    continue

            val_diff = values[add_idx] - (values[rem_idx] if rem_idx != -1 else 0)
            b_diff = _calc_b_delta((g_counts_ptr[g_add] if g_add >= 0 else 0), 
                                   (g_counts_ptr[item_groups[rem_idx]] if rem_idx != -1 and item_groups[rem_idx] >= 0 else 0), 
                                   1, (-1 if rem_idx != -1 else 0), b_val)
            
            if island_type == 0: eval_diff = val_diff
            elif island_type == 1: eval_diff = b_diff
            elif island_type == 3:
                eval_diff = val_diff * 0.1 + b_diff * 0.1 + ((weights[rem_idx,0] if rem_idx != -1 else 0) - weights[add_idx,0]) * w_slack
            elif island_type == 5:
                dens_add = values[add_idx] / (1.0 + weights[add_idx,0] + weights[add_idx,1] + weights[add_idx,2])
                dens_rem = (values[rem_idx] / (1.0 + weights[rem_idx,0] + weights[rem_idx,1] + weights[rem_idx,2])) if rem_idx != -1 else 0
                eval_diff = (dens_add - dens_rem) * w_density
            else: eval_diff = val_diff + b_diff

            if eval_diff > 0 or r < 0.05 * temp:
                if rem_idx != -1: 
                    sol[rem_idx] = 0; g_rem = item_groups[rem_idx]
                    if g_rem >= 0:
                        g_counts_ptr[g_rem] -= 1
                        if g_counts_ptr[g_rem] == 0 and g_rem < 1024:
                            g_bits[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                    for j in range(3): cur_w_ptr[j] -= weights[rem_idx, j]
                
                sol[add_idx] = 1
                if g_add >= 0:
                    g_counts_ptr[g_add] += 1
                    if g_add < 1024: g_bits[g_add // 64] |= (1ULL << (g_add % 64))
                for j in range(3): cur_w_ptr[j] += weights[add_idx, j]
                current_base += val_diff; current_bonus += b_diff
        else:
            if r < 0.03 * temp:
                gid = item_groups[add_idx]
                sol[add_idx] = 0
                if gid >= 0:
                    g_counts_ptr[gid] -= 1
                    if g_counts_ptr[gid] == 0 and gid < 1024:
                        g_bits[gid // 64] &= ~(1ULL << (gid % 64))
                for j in range(3): cur_w_ptr[j] -= weights[add_idx, j]
                current_base -= values[add_idx]
                current_bonus += _calc_b_single((g_counts_ptr[gid]+1 if gid >= 0 else 0), -1, b_val)

    scores_base[isol_idx, ind_idx] = current_base
    scores_bonus[isol_idx, ind_idx] = current_bonus

def _initialize_populations(char[:, :, :] pops, int[:, :] weights, int[:] capacities, 
                             int[:] item_groups, unsigned long long[:, :] conflict_masks,
                             int n_isl, int p_size, int n_items):
    cdef int i, j, k, m, idx, cw0, cw1, cw2, gid, conflict
    cdef unsigned long long g_bits[16]
    for i in range(n_isl):
        for j in range(p_size):
            indices = np.random.permutation(n_items).astype(np.int32)
            cw0=0; cw1=0; cw2=0
            memset(g_bits, 0, sizeof(g_bits))
            for k in range(n_items):
                idx = indices[k]; gid = item_groups[idx]
                conflict = 0
                if 0 <= gid < 1024:
                    for m in range(16):
                        if g_bits[m] & conflict_masks[gid, m]: conflict = 1; break
                if conflict: continue
                if cw0 + weights[idx,0] <= capacities[0] and cw1 + weights[idx,1] <= capacities[1] and cw2 + weights[idx,2] <= capacities[2]:
                    pops[i, j, idx] = 1
                    cw0 += weights[idx,0]; cw1 += weights[idx,1]; cw2 += weights[idx,2]
                    if 0 <= gid < 1024: g_bits[gid // 64] |= (1ULL << (gid % 64))

def _perform_advanced_evolution(char[:, :, :] pops, double[:, :] s_base, double[:, :] s_bonus, 
                                char[:, :] hof_sols, double[:] hof_scores,
                                int n_isl, int p_size, int e_size, int num_new_gen,
                                double prob_crossover, double prob_mut_repair, double prob_mut_30,
                                int[:] item_groups, int n_groups, int group_max,
                                int[:, :] weights, int[:] capacities, int[:] values,
                                int[:] s_idx_desc, int[:] s_idx_asc, unsigned long long[:, :] conflict_masks):
    cdef int i, j, k, target_idx, m_idx, best_m_idx, n_items = pops.shape[2]
    cdef int p1, p2, cw0, cw1, cw2, gid
    cdef double r, best_diff, diff, cur_score, r_mut
    cdef int[:] child_g_counts = np.zeros(n_groups, dtype=np.int32)
    
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
        for j in range(num_new_gen):
            r = np.random.rand()
            if r < prob_crossover:
                p1 = p_list[np.random.randint(0, len(p_list))]
                p2 = elite_indices[np.random.randint(0, min(e_size, p_size))]
                cw0=0; cw1=0; cw2=0; child_g_counts[:] = 0
                for k in range(n_items):
                    if pops[i, p1, k] == 1 and pops[0, p2, k] == 1:
                        gid = item_groups[k]
                        if cw0+weights[k,0]<=capacities[0] and cw1+weights[k,1]<=capacities[1] and cw2+weights[k,2]<=capacities[2] and (gid < 0 or child_g_counts[gid]<group_max):
                            next_pops[i, j, k] = 1
                            cw0+=weights[k,0]; cw1+=weights[k,1]; cw2+=weights[k,2]
                            if gid >= 0: child_g_counts[gid]+=1
                for k in range(n_items):
                    if next_pops[i, j, k] == 1: continue
                    if pops[i, p1, k] == 1 or pops[0, p2, k] == 1:
                        gid = item_groups[k]
                        if np.random.rand() < (0.8 if (gid >= 0 and child_g_counts[gid] > 0) else 0.3):
                            if cw0+weights[k,0]<=capacities[0] and cw1+weights[k,1]<=capacities[1] and cw2+weights[k,2]<=capacities[2] and (gid < 0 or child_g_counts[gid]<group_max):
                                next_pops[i, j, k] = 1
                                cw0+=weights[k,0]; cw1+=weights[k,1]; cw2+=weights[k,2]
                                if gid >= 0: child_g_counts[gid]+=1
                _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc, conflict_masks)
            else:
                target_idx = p_list[np.random.randint(0, len(p_list))]
                next_pops[i, j, :] = pops[i, target_idx, :]
                r_mut = np.random.rand()
                if r_mut < prob_mut_repair:
                    _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc, conflict_masks)
                elif r_mut < (prob_mut_repair + prob_mut_30):
                    best_diff = -1e9; best_m_idx = -1
                    for _ in range(30):
                        m_idx = np.random.randint(0, n_items)
                        diff = (1 - 2*next_pops[i, j, m_idx]) * values[m_idx]
                        if diff > best_diff:
                            best_diff = diff; best_m_idx = m_idx
                    if best_m_idx != -1: next_pops[i, j, best_m_idx] = 1 - next_pops[i, j, best_m_idx]
                    _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc, conflict_masks)
                else:
                    k = np.random.randint(0, n_items)
                    next_pops[i, j, k] = 1 - next_pops[i, j, k]
                    _repair_individual(next_pops[i, j, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc, conflict_masks)
    pops[:, :num_new_gen, :] = next_pops[:, :num_new_gen, :]

def solve_knapsack_sa_parallel(
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups,
    int[:, :] conflict_pairs, int n_items, int n_groups, int group_max,
    int pop_size = 50, int elite_size = 20, int num_new_gen = 30,
    int max_generations = 150, int iter_per_ind = 30000,
    double prob_crossover = 0.4, double prob_mut_repair = 0.15, double prob_mut_30 = 0.40,
    double w_slack = 1.0, double w_density = 1000.0
):
    cdef int n_islands = 6
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

    cdef int[:] item_conf_ptr = np.zeros(n_items + 1, dtype=np.int32)
    cdef int[:] item_conf_data = np.zeros(1, dtype=np.int32)
    
    cdef unsigned long long[:, :] conflict_masks = np.zeros((1024, 16), dtype=np.uint64)
    for i in range(conflict_pairs.shape[0]):
        u = conflict_pairs[i, 0]; v = conflict_pairs[i, 1]
        if u < 1024 and v < 1024:
            conflict_masks[u, v // 64] |= (1ULL << (v % 64))
            conflict_masks[v, u // 64] |= (1ULL << (u % 64))

    _initialize_populations(populations, weights, capacities, item_groups, conflict_masks, n_islands, pop_size, n_items)

    for gen in range(max_generations):
        for isol in prange(n_islands, nogil=True):
            srand(base_seed ^ <unsigned int>(isol * 888 + gen))
            for ind in range(pop_size):
                _run_advanced_sa(
                    isol, populations[isol, ind, :], scores_base, scores_bonus, isol, ind,
                    values, weights, capacities, item_groups, n_items, n_groups, group_max,
                    iter_per_ind, island_g_counts[isol, ind, :], island_cur_w[isol, ind, :],
                    item_conf_ptr, item_conf_data, conflict_masks, gen, w_slack, w_density
                )

        _perform_advanced_evolution(populations, scores_base, scores_bonus, hof_sols, hof_scores, 
                                   n_islands, pop_size, elite_size, num_new_gen, 
                                   prob_crossover, prob_mut_repair, prob_mut_30,
                                   item_groups, n_groups, group_max, weights, capacities, values, 
                                   s_idx_desc, s_idx_asc, conflict_masks)

    total_scores = np.array(scores_base) + np.array(scores_bonus)
    best_tup = np.unravel_index(np.argmax(total_scores), (n_islands, pop_size))
    cdef int b_isl = best_tup[0], b_ind = best_tup[1]
    _repair_individual(populations[b_isl, b_ind, :], values, weights, capacities, item_groups, group_max, n_items, n_groups, s_idx_desc, s_idx_asc, conflict_masks)
    
    cdef char[:] best_sol = populations[b_isl, b_ind, :]
    cdef double final_base = 0, final_bonus = 0
    cdef int[:] final_g_counts = np.zeros(n_groups, dtype=np.int32)
    cdef int gid_f, it_idx
    
    for it_idx in range(n_items):
        if best_sol[it_idx] == 1:
            final_base += values[it_idx]
            gid_f = item_groups[it_idx]
            if 0 <= gid_f < n_groups: final_g_counts[gid_f] += 1
                
    for gid_f in range(n_groups):
        if 3 <= final_g_counts[gid_f] <= 5: final_bonus += 50
            
    return final_base + final_bonus, np.array(best_sol)