from typing import Tuple
import numpy as np

def solve_knapsack_sa_single(
    values: np.ndarray,
    weights: np.ndarray,
    capacities: np.ndarray,
    item_groups: np.ndarray,
    conflict_pairs: np.ndarray,
    n_items: int,
    n_groups: int,
    group_max: int,
    bonus_t1: int,
    bonus_t2: int,
    bonus_t3: int,
    bonus_val: float,
    iterations: int,
    seed: int = 42,
) -> Tuple[float, np.ndarray]: ...
def solve_knapsack_sa_parallel(
    values: np.ndarray,
    weights: np.ndarray,
    capacities: np.ndarray,
    item_groups: np.ndarray,
    conflict_pairs: np.ndarray,
    n_items: int,
    n_groups: int,
    group_max: int,
    pop_size: int = 20,
    rand_add_size: int = 20,
    crossover_size: int = 50,
    max_generations: int = 1000,
    iter_per_ind: int = 1000000,
    patience: int = 10,
) -> Tuple[float, np.ndarray]: ...
