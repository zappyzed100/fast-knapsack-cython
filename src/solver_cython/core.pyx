# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.time cimport time as c_time
from libc.string cimport memset, memcpy
from libc.math cimport exp

# ---------------------------------------------------------
# 1. 高速乱数 (Xorshift)
# ---------------------------------------------------------
cdef struct xorshift_state:
    unsigned int a

cdef inline unsigned int xorshift_next(xorshift_state* state) noexcept nogil:
    state.a ^= state.a << 13
    state.a ^= state.a >> 17
    state.a ^= state.a << 5
    return state.a

cdef inline double xorshift_double(xorshift_state* state) noexcept nogil:
    return <double>(xorshift_next(state) & 0xFFFFFFF) / 268435456.0

# ---------------------------------------------------------
# 2. SAコアエンジン (不整合解消版)
# ---------------------------------------------------------
cdef void _run_sa_on_block(
    char* sol, double* score_ptr,
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups,
    int n_items, int n_groups, int group_max, int iterations,
    unsigned long long[:, :] conflict_masks, int gen,
    int bonus_t1, int bonus_t2, int bonus_t3, double bonus_val,
    int* gc_buf, char* best_sol_buf, unsigned int seed
) noexcept nogil:
    cdef int it, j, k, add_idx, rem_idx, g_add, g_rem, conflict
    cdef double r, temp, cur_val_sum, cur_bonus, bonus_diff, diff
    cdef unsigned long long g_bits[16]
    cdef int cur_w[3]
    cdef xorshift_state xsr
    xsr.a = seed
    
    # --- 状態の完全リセットと初期解の修復 ---
    memset(g_bits, 0, sizeof(g_bits))
    memset(cur_w, 0, sizeof(cur_w))
    memset(gc_buf, 0, n_groups * sizeof(int))
    cur_val_sum = 0.0

    for j in range(n_items):
        if sol[j] == 1:
            g_add = item_groups[j]
            conflict = 0
            if 0 <= g_add < 1024:
                for k in range(16):
                    if g_bits[k] & conflict_masks[g_add, k]:
                        conflict = 1
                        break
            
            # 制約違反（重量、排他、上限）がある場合は deselect
            if (conflict or 
                cur_w[0] + weights[j, 0] > capacities[0] or 
                cur_w[1] + weights[j, 1] > capacities[1] or 
                cur_w[2] + weights[j, 2] > capacities[2] or
                (g_add >= 0 and gc_buf[g_add] >= group_max)):
                sol[j] = 0
            else:
                cur_val_sum += <double>values[j]
                for k in range(3): cur_w[k] += weights[j, k]
                if g_add >= 0:
                    gc_buf[g_add] += 1
                    if g_add < 1024:
                        g_bits[g_add // 64] |= (1ULL << (g_add % 64))

    cur_bonus = 0.0
    for j in range(n_groups):
        if gc_buf[j] >= bonus_t1:
            cur_bonus += bonus_val
        if gc_buf[j] >= bonus_t2:
            cur_bonus += bonus_val
        if gc_buf[j] >= bonus_t3:
            cur_bonus += bonus_val

    cdef double best_total = cur_val_sum + cur_bonus
    memcpy(best_sol_buf, sol, n_items * sizeof(char))

    # --- SAメインループ ---
    for it in range(iterations):
        temp = (1.0 - (<double>it / iterations)) * (1.0 / (1.0 + gen * 0.1))
        add_idx = xorshift_next(&xsr) % n_items
        g_add = item_groups[add_idx]
        r = xorshift_double(&xsr)

        if sol[add_idx] == 0:
            # 追加試行
            if (cur_w[0] + weights[add_idx, 0] <= capacities[0] and
                cur_w[1] + weights[add_idx, 1] <= capacities[1] and
                cur_w[2] + weights[add_idx, 2] <= capacities[2]):
                
                conflict = 0
                if 0 <= g_add < 1024:
                    for k in range(16):
                        if g_bits[k] & conflict_masks[g_add, k]:
                            conflict = 1; break
                
                if not conflict and (g_add < 0 or gc_buf[g_add] < group_max):
                    bonus_diff = 0.0
                    if g_add >= 0:
                        if gc_buf[g_add] == bonus_t1 - 1 or gc_buf[g_add] == bonus_t2 - 1 or gc_buf[g_add] == bonus_t3 - 1:
                            bonus_diff = bonus_val
                    
                    diff = <double>values[add_idx] + bonus_diff
                    if diff > 0 or (temp > 0 and r < exp(diff / (temp * 100.0))):
                        sol[add_idx] = 1
                        for k in range(3): cur_w[k] += weights[add_idx, k]
                        if g_add >= 0:
                            gc_buf[g_add] += 1
                            if g_add < 1024:
                                g_bits[g_add // 64] |= (1ULL << (g_add % 64))
                        cur_val_sum += <double>values[add_idx]
                        cur_bonus += bonus_diff
            
            else:
                # 入れ替え(Swap)試行
                rem_idx = xorshift_next(&xsr) % n_items
                if sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    if (cur_w[0] - weights[rem_idx, 0] + weights[add_idx, 0] <= capacities[0] and
                        cur_w[1] - weights[rem_idx, 1] + weights[add_idx, 1] <= capacities[1] and
                        cur_w[2] - weights[rem_idx, 2] + weights[add_idx, 2] <= capacities[2]):
                        
                        if 0 <= g_rem < 1024 and gc_buf[g_rem] == 1:
                            g_bits[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                        
                        conflict = 0
                        if 0 <= g_add < 1024:
                            for k in range(16):
                                if g_bits[k] & conflict_masks[g_add, k]:
                                    conflict = 1; break
                        
                        # 同一グループ内交換なら上限チェックを緩和
                        if not conflict and (g_add < 0 or gc_buf[g_add] < (group_max + (1 if g_add == g_rem else 0))):
                            bonus_diff = 0.0
                            if g_add != g_rem:
                                if g_rem >= 0:
                                    if gc_buf[g_rem] == bonus_t1 or gc_buf[g_rem] == bonus_t2 or gc_buf[g_rem] == bonus_t3:
                                        bonus_diff -= bonus_val
                                if g_add >= 0:
                                    if gc_buf[g_add] == bonus_t1 - 1 or gc_buf[g_add] == bonus_t2 - 1 or gc_buf[g_add] == bonus_t3 - 1:
                                        bonus_diff += bonus_val
                            
                            diff = <double>(values[add_idx] - values[rem_idx]) + bonus_diff
                            if diff > 0 or (temp > 0 and r < exp(diff / (temp * 100.0))):
                                sol[rem_idx] = 0; sol[add_idx] = 1
                                for k in range(3): cur_w[k] += weights[add_idx, k] - weights[rem_idx, k]
                                if g_add >= 0: gc_buf[g_add] += 1
                                if g_rem >= 0: gc_buf[g_rem] -= 1
                                if 0 <= g_add < 1024: g_bits[g_add // 64] |= (1ULL << (g_add % 64))
                                cur_val_sum += <double>(values[add_idx] - values[rem_idx])
                                cur_bonus += bonus_diff
                                if cur_val_sum + cur_bonus > best_total:
                                    best_total = cur_val_sum + cur_bonus
                                    memcpy(best_sol_buf, sol, n_items * sizeof(char))
                                continue
                        if 0 <= g_rem < 1024 and gc_buf[g_rem] >= 1:
                            g_bits[g_rem // 64] |= (1ULL << (g_rem % 64))
        else:
            # 削除
            if r < 0.02 * temp:
                g_rem = item_groups[add_idx]
                bonus_diff = 0.0
                if g_rem >= 0:
                    if gc_buf[g_rem] == bonus_t1 or gc_buf[g_rem] == bonus_t2 or gc_buf[g_rem] == bonus_t3:
                        bonus_diff = -bonus_val
                sol[add_idx] = 0
                for k in range(3): cur_w[k] -= weights[add_idx, k]
                if g_rem >= 0:
                    gc_buf[g_rem] -= 1
                    if gc_buf[g_rem] == 0 and g_rem < 1024:
                        g_bits[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                cur_val_sum -= <double>values[add_idx]
                cur_bonus += bonus_diff

        if cur_val_sum + cur_bonus > best_total:
            best_total = cur_val_sum + cur_bonus
            memcpy(best_sol_buf, sol, n_items * sizeof(char))

    memcpy(sol, best_sol_buf, n_items * sizeof(char))
    score_ptr[0] = best_total

# ---------------------------------------------------------
# 3. Greedy交叉 (そのまま)
# ---------------------------------------------------------
cdef void _greedy_crossover(
    char* p1, char* p2, char* child,
    int[:] item_groups, int[:, :] weights, int[:] capacities,
    unsigned long long[:, :] conflict_masks,
    int n_items, int n_groups, int group_max,
    int[:] s_idx_desc, int* gc_buf
) noexcept nogil:
    cdef int k, idx, gid, m, conflict
    cdef int cur_w[3]
    cdef unsigned long long g_bits[16]
    memset(cur_w, 0, sizeof(cur_w))
    memset(gc_buf, 0, n_groups * sizeof(int))
    memset(g_bits, 0, sizeof(g_bits))
    memset(child, 0, n_items * sizeof(char))
    for k in range(n_items):
        idx = s_idx_desc[k]
        if p1[idx] == 1 or p2[idx] == 1:
            gid = item_groups[idx]
            conflict = 0
            if gid >= 0 and gid < 1024:
                for m in range(16):
                    if g_bits[m] & conflict_masks[gid, m]:
                        conflict = 1
                        break
            if conflict: continue
            if (cur_w[0] + weights[idx, 0] <= capacities[0] and
                cur_w[1] + weights[idx, 1] <= capacities[1] and
                cur_w[2] + weights[idx, 2] <= capacities[2] and
                (gid < 0 or gc_buf[gid] < group_max)):
                child[idx] = 1
                for m in range(3): cur_w[m] += weights[idx, m]
                if gid >= 0:
                    gc_buf[gid] += 1
                    if gid < 1024:
                        g_bits[gid // 64] |= (1ULL << (gid % 64))

# ---------------------------------------------------------
# 4. 並列進化計算メイン
# ---------------------------------------------------------
def solve_knapsack_sa_parallel(
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups,
    int[:, :] conflict_pairs, int n_items, int n_groups, int group_max,
    int bonus_t1,
    int bonus_t2,
    int bonus_t3,
    double bonus_val,
    int pop_size = 20,
    int rand_add_size = 20,
    int crossover_size = 50,
    int max_generations = 1000,
    int iter_per_ind = 1000000,
    int patience = 10,
    double timeout_sec = 0.0,
    bint verbose = True
):
    import time as _time
    cdef int total_pop = pop_size + rand_add_size + crossover_size
    cdef int gen, i, s_idx, best_idx
    cdef int pool_size = pop_size + rand_add_size
    cdef int[:] cx1
    cdef int[:] cx2
    cdef double last_best = -1.0
    cdef int no_improvement = 0
    cdef unsigned int base_seed = <unsigned int>c_time(NULL)
    cdef xorshift_state sel_rng
    deadline = _time.perf_counter() + timeout_sec if timeout_sec > 0.0 else None

    cdef char[:, :] pops = np.zeros((total_pop, n_items), dtype=np.int8)
    scores_arr = np.zeros(total_pop, dtype=np.float64)
    cdef double[:] scores = scores_arr
    cdef char[:, :] temp_pops = np.zeros((pop_size, n_items), dtype=np.int8)
    cdef double[:] temp_scores = np.zeros(pop_size, dtype=np.float64)
    cx1_arr = np.empty(crossover_size, dtype=np.int32)
    cx2_arr = np.empty(crossover_size, dtype=np.int32)
    cx1 = cx1_arr
    cx2 = cx2_arr

    cdef int** gc_buffers = <int**>malloc(total_pop * sizeof(int*))
    for i in range(total_pop):
        gc_buffers[i] = <int*>malloc(n_groups * sizeof(int))

    cdef char** best_sol_buffers = <char**>malloc(total_pop * sizeof(char*))
    for i in range(total_pop):
        best_sol_buffers[i] = <char*>malloc(n_items * sizeof(char))

    densities = np.array(values) / (1.0 + np.sum(np.array(weights), axis=1))
    cdef int[:] s_idx_desc = np.argsort(-densities).astype(np.int32)

    cdef unsigned long long[:, :] conflict_masks = np.zeros((1024, 16), dtype=np.uint64)
    for i in range(conflict_pairs.shape[0]):
        if conflict_pairs[i, 0] < 1024 and conflict_pairs[i, 1] < 1024:
            conflict_masks[conflict_pairs[i,0], conflict_pairs[i,1]//64] |= (1ULL << (conflict_pairs[i,1]%64))
            conflict_masks[conflict_pairs[i,1], conflict_pairs[i,0]//64] |= (1ULL << (conflict_pairs[i,0]%64))

    for gen in range(max_generations):
        # 1. 新規個体
        for i in prange(pop_size, pop_size + rand_add_size, nogil=True):
            memset(&pops[i, 0], 0, n_items * sizeof(char))
            _run_sa_on_block(&pops[i, 0], &scores[i], values, weights, capacities, item_groups, n_items, n_groups, group_max, iter_per_ind, conflict_masks, gen, bonus_t1, bonus_t2, bonus_t3, bonus_val, gc_buffers[i], best_sol_buffers[i], base_seed ^ <unsigned int>(i * 777 + gen))

        # 2. 交叉 (Python/NumPy RNG 呼び出しを避け、C側乱数で親を選択)
        sel_rng.a = base_seed ^ <unsigned int>(gen * 1000003 + 97)
        for i in range(crossover_size):
            cx1[i] = <int>(xorshift_next(&sel_rng) % <unsigned int>pool_size)
            cx2[i] = <int>(xorshift_next(&sel_rng) % <unsigned int>pool_size)
        for i in prange(pop_size + rand_add_size, total_pop, nogil=True):
            _greedy_crossover(&pops[<int>cx1[i - pop_size - rand_add_size], 0], &pops[<int>cx2[i - pop_size - rand_add_size], 0], &pops[i, 0], item_groups, weights, capacities, conflict_masks, n_items, n_groups, group_max, s_idx_desc, gc_buffers[i])

        # 3. ブラッシュアップ SA
        for i in prange(total_pop, nogil=True):
            _run_sa_on_block(&pops[i, 0], &scores[i], values, weights, capacities, item_groups, n_items, n_groups, group_max, iter_per_ind, conflict_masks, gen, bonus_t1, bonus_t2, bonus_t3, bonus_val, gc_buffers[i], best_sol_buffers[i], base_seed ^ <unsigned int>(i * 123 + gen))

        sorted_indices = np.argsort(-scores_arr)
        best_idx = sorted_indices[0]
        if scores[best_idx] > last_best:
            last_best = scores[best_idx]
            no_improvement = 0
            if verbose:
                print(f"Gen {gen:03d}: Improved! Best Score = {last_best:.1f}")
        else:
            no_improvement += 1
            if verbose:
                print(f"Gen {gen:03d}: No Improvement ({no_improvement}/{patience})")

        # 世代交代
        for i in range(pop_size):
            s_idx = sorted_indices[i]
            memcpy(&temp_pops[i, 0], &pops[s_idx, 0], n_items * sizeof(char))
            temp_scores[i] = scores[s_idx]
        for i in range(pop_size):
            memcpy(&pops[i, 0], &temp_pops[i, 0], n_items * sizeof(char))
            scores[i] = temp_scores[i]
        
        if deadline is not None and _time.perf_counter() >= deadline:
            if verbose:
                print(f"Gen {gen:03d}: Time limit reached.")
            break

        if no_improvement >= patience:
            break

    for i in range(total_pop):
        free(gc_buffers[i])
        free(best_sol_buffers[i])
    free(gc_buffers)
    free(best_sol_buffers)
    return last_best, np.array(pops[0])

# ---------------------------------------------------------
# 5. SAソルバー (修正版)
# ---------------------------------------------------------
def solve_knapsack_sa(
    int[:] values, 
    int[:, :] weights, 
    int[:] capacities, 
    int[:] item_groups,
    int[:, :] conflict_pairs, 
    int n_items, 
    int n_groups, 
    int group_max, 
    int bonus_t1,
    int bonus_t2,
    int bonus_t3,
    double bonus_val,
    long long max_iter,
    int[:] rand_add,
    int[:] rand_rem,
    double[:] rand_flt
):
    # コンフリクトマスク (グループビットマスク)
    cdef unsigned long long[:, :] conflict_masks = np.zeros((1024, 16), dtype=np.uint64)
    cdef unsigned long long g_bits[16]
    memset(g_bits, 0, sizeof(g_bits))
    
    cdef int i, j, k, it, g_add, g_rem, add_idx, rem_idx, u, v
    cdef int over_weight, w_ok, conflict
    cdef double r_val, temp, diff, val_diff, current_bonus, bonus_diff

    # マスク作成
    for i in range(conflict_pairs.shape[0]):
        u = conflict_pairs[i, 0]
        v = conflict_pairs[i, 1]
        if u < 1024 and v < 1024:
            conflict_masks[u, v // 64] |= (1ULL << (v % 64))
            conflict_masks[v, u // 64] |= (1ULL << (u % 64))

    cdef char[:] current_sol = np.zeros(n_items, dtype=np.int8)
    cdef int[:] group_counts = np.zeros(n_groups, dtype=np.int32)
    cdef int cur_w[3]
    memset(cur_w, 0, sizeof(cur_w))

    cdef double current_score = 0.0
    cdef double best_score = 0.0
    cdef char[:] best_sol = np.zeros(n_items, dtype=np.int8)
    
    # ボーナスの初期計算
    current_bonus = 0.0
    # 初期解が0なので0

    for it in range(max_iter):
        add_idx = rand_add[it]
        g_add = item_groups[add_idx]
        temp = 1.0 - (<double>it / <double>max_iter) 
        r_val = rand_flt[it]

        if current_sol[add_idx] == 0:
            # 1. 重みチェック
            over_weight = 0
            for j in range(3):
                if cur_w[j] + weights[add_idx, j] > capacities[j]:
                    over_weight = 1
                    break

            if over_weight:
                # 入れ替え戦略
                rem_idx = rand_rem[it]
                if current_sol[rem_idx] == 1:
                    g_rem = item_groups[rem_idx]
                    w_ok = 1
                    for j in range(3):
                        if cur_w[j] - weights[rem_idx, j] + weights[add_idx, j] > capacities[j]:
                            w_ok = 0
                            break
                    
                    if w_ok:
                        # 差分でのコンフリクト判定
                        # 削除するグループが他に存在しない場合のみビットを一時的に落とす
                        if 0 <= g_rem < 1024 and group_counts[g_rem] == 1:
                            g_bits[g_rem // 64] &= ~(1ULL << (g_rem % 64))
                        
                        conflict = 0
                        if 0 <= g_add < 1024:
                            for k in range(16):
                                if g_bits[k] & conflict_masks[g_add, k]:
                                    conflict = 1
                                    break
                        
                        if not conflict:
                            # ボーナス差分計算
                            bonus_diff = 0.0
                            if g_add != g_rem:
                                # 削除の影響
                                if group_counts[g_rem] == bonus_t1 or group_counts[g_rem] == bonus_t2 or group_counts[g_rem] == bonus_t3:
                                    bonus_diff -= bonus_val
                                # 追加の影響
                                if group_counts[g_add] == bonus_t1 - 1 or group_counts[g_add] == bonus_t2 - 1 or group_counts[g_add] == bonus_t3 - 1:
                                    bonus_diff += bonus_val
                            
                            val_diff = <double>(values[add_idx] - values[rem_idx])
                            diff = val_diff + bonus_diff
                            
                            if diff > 0 or (temp > 0 and r_val < exp(diff / (temp * 100.0))):
                                current_sol[rem_idx] = 0
                                current_sol[add_idx] = 1
                                for j in range(3):
                                    cur_w[j] += weights[add_idx, j] - weights[rem_idx, j]
                                group_counts[g_add] += 1
                                group_counts[g_rem] -= 1
                                current_score += val_diff
                                current_bonus += bonus_diff
                                
                                # ビット確定更新 (g_remのビットは事前クリア済み)
                                if 0 <= g_add < 1024:
                                    g_bits[g_add // 64] |= (1ULL << (g_add % 64))
                                
                                if current_score + current_bonus > best_score:
                                    best_score = current_score + current_bonus
                                    memcpy(&best_sol[0], &current_sol[0], n_items * sizeof(char))
                                continue
                        
                        # 失敗時はビットを戻す
                        if 0 <= g_rem < 1024 and group_counts[g_rem] >= 1:
                            g_bits[g_rem // 64] |= (1ULL << (g_rem % 64))
            
            elif group_counts[g_add] < group_max:
                # 単純追加
                conflict = 0
                if 0 <= g_add < 1024:
                    for k in range(16):
                        if g_bits[k] & conflict_masks[g_add, k]:
                            conflict = 1
                            break
                if not conflict:
                    bonus_diff = 0.0
                    if group_counts[g_add] == bonus_t1 - 1 or group_counts[g_add] == bonus_t2 - 1 or group_counts[g_add] == bonus_t3 - 1:
                        bonus_diff += bonus_val
                    
                    val_diff = <double>values[add_idx]
                    diff = val_diff + bonus_diff
                    
                    if diff > 0 or (temp > 0 and r_val < exp(diff / (temp * 100.0))):
                        current_sol[add_idx] = 1
                        for j in range(3): cur_w[j] += weights[add_idx, j]
                        group_counts[g_add] += 1
                        current_score += val_diff
                        current_bonus += bonus_diff
                        if 0 <= g_add < 1024:
                            g_bits[g_add // 64] |= (1ULL << (g_add % 64))
                        
                        if current_score + current_bonus > best_score:
                            best_score = current_score + current_bonus
                            memcpy(&best_sol[0], &current_sol[0], n_items * sizeof(char))
        else:
            # 削除
            if r_val < 0.02 * temp:
                g_rem = item_groups[add_idx]
                bonus_diff = 0.0
                if group_counts[g_rem] == bonus_t1 or group_counts[g_rem] == bonus_t2 or group_counts[g_rem] == bonus_t3:
                    bonus_diff -= bonus_val
                
                current_sol[add_idx] = 0
                for j in range(3): cur_w[j] -= weights[add_idx, j]
                group_counts[g_rem] -= 1
                current_score -= <double>values[add_idx]
                current_bonus += bonus_diff
                
                if group_counts[g_rem] == 0 and 0 <= g_rem < 1024:
                    g_bits[g_rem // 64] &= ~(1ULL << (g_rem % 64))

    return best_score, np.array(best_sol)

# ---------------------------------------------------------
# 6. 時間指定SAソルバー (マルチスタート・外部乱数配列不要)
# ---------------------------------------------------------
def solve_knapsack_sa_timed(
    int[:] values, int[:, :] weights, int[:] capacities, int[:] item_groups,
    int[:, :] conflict_pairs, int n_items, int n_groups, int group_max,
    int bonus_t1, int bonus_t2, int bonus_t3, double bonus_val,
    double timeout_sec, int chunk_iter=5000000, bint verbose=True
):
    """timeout_sec 秒間、SAをチャンク単位で繰り返して最良解を返す。"""
    import time as _time
    cdef int i, u, v
    cdef double chunk_score = -1.0
    cdef double best_score_global = -1.0
    cdef int run_idx = 0

    cdef unsigned long long[:, :] conflict_masks = np.zeros((1024, 16), dtype=np.uint64)
    for i in range(conflict_pairs.shape[0]):
        u = conflict_pairs[i, 0]
        v = conflict_pairs[i, 1]
        if u < 1024 and v < 1024:
            conflict_masks[u, v // 64] |= (1ULL << (v % 64))
            conflict_masks[v, u // 64] |= (1ULL << (u % 64))

    cdef char[:] sol = np.zeros(n_items, dtype=np.int8)
    cdef char[:] best_sol_global = np.zeros(n_items, dtype=np.int8)
    cdef int* gc_buf = <int*>malloc(n_groups * sizeof(int))
    cdef char* best_sol_buf = <char*>malloc(n_items * sizeof(char))

    cdef unsigned int seed_base = <unsigned int>c_time(NULL)
    deadline = _time.perf_counter() + timeout_sec

    try:
        while _time.perf_counter() < deadline:
            memset(&sol[0], 0, n_items * sizeof(char))
            chunk_score = -1.0
            _run_sa_on_block(
                &sol[0], &chunk_score,
                values, weights, capacities, item_groups,
                n_items, n_groups, group_max, chunk_iter,
                conflict_masks, run_idx,
                bonus_t1, bonus_t2, bonus_t3, bonus_val,
                gc_buf, best_sol_buf,
                seed_base ^ <unsigned int>(run_idx * 1000003)
            )
            if chunk_score > best_score_global:
                best_score_global = chunk_score
                memcpy(&best_sol_global[0], &sol[0], n_items * sizeof(char))
                if verbose:
                    print(f"  Run {run_idx}: New best = {best_score_global:.1f}")
            run_idx += 1
    finally:
        free(gc_buf)
        free(best_sol_buf)

    return best_score_global, np.array(best_sol_global)