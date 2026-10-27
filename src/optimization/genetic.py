from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import time

from src.qubo.objective import compute_qubo_component_stats, extract_qubo_total, print_qubo_component_stats


def solve_qubo_ga_energy_soft_budget(
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Dict[str, Any],
    K_budget: int,  # 仅用于初始化/统计；Q_total里已编码K也没问题
    # --- GA hyperparams ---
    pop_size: int = 80,
    max_iters: int = 300,          # generation 数（对齐A1命名）
    elite_frac: float = 0.1,
    tournament_k: int = 3,
    cx_rate: float = 0.9,
    mut_rate: float = 0.02,        # bit-flip 概率（对每个位）
    # --- init control (soft-budget 但建议围绕K初始化) ---
    init_mode: str = "bernoulli",  # "bernoulli" | "near_k"
    init_p: Optional[float] = None,  # bernoulli概率，None则用K/N
    near_k_jitter: int = 2,          # near_k: 在K的基础上±jitter个bit扰动
    # --- rng / budget ---
    seed: int = 0,
    time_budget_sec: Optional[float] = None,
    # --- convergence early-stop ---
    early_stop_patience: int = 200,
    early_stop_min_iters: int = 200,
    early_stop_abs_tol: float = 1e-8,
    # --- history ---
    keep_histories: bool = True,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, float, Optional[List[List[float]]], Dict[str, Any]]:
    """
    GA (no restart) minimize E(x)=x^T Q x on bitstrings x∈{0,1}^N.
    Soft budget is assumed to be inside Q (Q_total = Q_det + Q_budget).

    Returns (A1-compatible):
      x_best: (N,) int {0,1}
      best_E: float
      histories: [curve] where curve = best energy each generation (len = gens_done+1)
      info: {
        "restarts_done": 1,
        "iters_total": gens_done,
        "evals_total": total_evals,
        "stop_reason": "max_iters"|"time_budget"|"early_stop",
        "elapsed_sec": elapsed,
        "curve_time_sec_list": [elapsed],
        "pop_size": pop_size,
        "gens_done": gens_done,
        "num_sensors_best": sum(x_best)
      }
    """
    Q, qubo_const = extract_qubo_total(Q_dist_np, const_dist)
    N = Q.shape[0]
    assert Q.shape == (N, N), "Q must be square"
    assert N >= 2, "N too small"
    if not (0 <= int(K_budget) <= N):
        raise ValueError(f"K_budget must be in [0, {N}], got {K_budget}.")

    rng = np.random.default_rng(seed)
    print("seed=", seed, "first_rand=", rng.random())
    t0 = time.perf_counter()

    def time_up() -> bool:
        return (time_budget_sec is not None) and ((time.perf_counter() - t0) >= time_budget_sec)

    # --- energy ---
    def energy(x: np.ndarray) -> float:
        # x: (N,) {0,1}
        x_f = x.astype(float, copy=False)
        return float(x_f @ (Q @ x_f))+qubo_const

    def repair_khot(x: np.ndarray) -> np.ndarray:
        x = (np.asarray(x, dtype=np.int8).reshape(-1) > 0).astype(np.int8)
        s = int(x.sum())
        if s > int(K_budget):
            ones = np.flatnonzero(x == 1)
            drop = rng.choice(ones, size=(s - int(K_budget)), replace=False)
            x[drop] = 0
        elif s < int(K_budget):
            zeros = np.flatnonzero(x == 0)
            add = rng.choice(zeros, size=(int(K_budget) - s), replace=False)
            x[add] = 1
        return x

    # --- init ---
    def init_individual() -> np.ndarray:
        if init_mode == "near_k":
            # start from exactly-K, then apply K-preserving swaps as jitter
            x = np.zeros(N, dtype=np.int8)
            if 0 < int(K_budget) <= N:
                ones = rng.choice(N, size=K_budget, replace=False)
                x[ones] = 1
            m = int(rng.integers(0, max(1, int(near_k_jitter) + 1)))
            for _ in range(m):
                ones_idx = np.flatnonzero(x == 1)
                zeros_idx = np.flatnonzero(x == 0)
                if (ones_idx.size == 0) or (zeros_idx.size == 0):
                    break
                i = int(rng.choice(ones_idx))
                j = int(rng.choice(zeros_idx))
                x[i] = 0
                x[j] = 1
            return x

        # bernoulli
        p = float(init_p) if init_p is not None else float(K_budget) / float(max(1, N))
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        x = (rng.random(N) < p).astype(np.int8)
        return repair_khot(x)

    # --- selection ---
    def tournament_select(pop: List[np.ndarray], fit: np.ndarray) -> np.ndarray:
        idx = rng.integers(0, len(pop), size=int(tournament_k))
        best = idx[np.argmin(fit[idx])]
        return pop[int(best)]

    # --- crossover ---
    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if rng.random() >= float(cx_rate):
            return p1.copy()
        # K-preserving set crossover:
        # keep common active genes, then fill from parent-difference genes.
        if int(K_budget) == 0:
            return np.zeros(N, dtype=np.int8)

        ones1 = np.flatnonzero(p1 == 1)
        ones2 = np.flatnonzero(p2 == 1)
        common = np.intersect1d(ones1, ones2, assume_unique=True)

        child = np.zeros(N, dtype=np.int8)
        if common.size > 0:
            child[common] = 1

        need = int(K_budget) - int(common.size)
        if need <= 0:
            return child

        only1 = np.setdiff1d(ones1, common, assume_unique=True)
        only2 = np.setdiff1d(ones2, common, assume_unique=True)
        pool = np.concatenate([only1, only2])

        if pool.size >= need:
            pick = rng.choice(pool, size=need, replace=False)
            child[pick] = 1
        else:
            # Defensive fallback: fill from remaining zeros if parent pool is insufficient.
            if pool.size > 0:
                child[pool] = 1
            need2 = int(K_budget) - int(child.sum())
            if need2 > 0:
                zeros = np.flatnonzero(child == 0)
                add = rng.choice(zeros, size=need2, replace=False)
                child[add] = 1
        return child

    # --- mutation ---
    def mutate(x: np.ndarray) -> np.ndarray:
        # K-preserving mutation via 1<->0 swaps
        if mut_rate <= 0:
            return x
        if int(K_budget) == 0 or int(K_budget) == N:
            return x

        x = x.copy()
        ones_idx = np.flatnonzero(x == 1)
        zeros_idx = np.flatnonzero(x == 0)
        if (ones_idx.size == 0) or (zeros_idx.size == 0):
            return x

        sel_ones = ones_idx[rng.random(ones_idx.size) < float(mut_rate)]
        sel_zeros = zeros_idx[rng.random(zeros_idx.size) < float(mut_rate)]
        n_swap = int(min(sel_ones.size, sel_zeros.size))

        # Keep occasional exploration even when Bernoulli sampling yields 0 swaps.
        if n_swap == 0:
            if rng.random() < float(mut_rate):
                sel_ones = np.array([int(rng.choice(ones_idx))], dtype=int)
                sel_zeros = np.array([int(rng.choice(zeros_idx))], dtype=int)
                n_swap = 1
            else:
                return x
        else:
            if sel_ones.size > n_swap:
                sel_ones = rng.choice(sel_ones, size=n_swap, replace=False)
            if sel_zeros.size > n_swap:
                sel_zeros = rng.choice(sel_zeros, size=n_swap, replace=False)

        x[sel_ones] = 0
        x[sel_zeros] = 1
        return x

    # --- init pop ---
    pop: List[np.ndarray] = [repair_khot(init_individual()) for _ in range(int(pop_size))]
    fit = np.array([energy(ind) for ind in pop], dtype=float)
    evals_total = int(pop_size)

    order = np.argsort(fit)
    pop = [pop[int(i)] for i in order]
    fit = fit[order]
    print_qubo_component_stats(
        "[Init-Full][A2-pop-best]",
        compute_qubo_component_stats(pop[0].astype(float), Q_dist_np, const_dist),
    )

    elite_count = max(1, int(round(float(elite_frac) * float(pop_size))))

    curve: List[float] = [float(fit[0])]
    best_so_far = float(fit[0])
    stagnant_generations = 0
    gens_done = 0
    stop_reason = "max_iters"

    # --- GA loop ---
    for g in range(int(max_iters)):
        if time_up():
            stop_reason = "time_budget"
            break

        new_pop: List[np.ndarray] = pop[:elite_count]

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fit)
            p2 = tournament_select(pop, fit)

            child = crossover(p1, p2)
            child = mutate(child)
            if int(child.sum()) != int(K_budget):
                child = repair_khot(child)
            new_pop.append(child)


        pop = new_pop
        fit = np.array([energy(ind) for ind in pop], dtype=float)
        evals_total += int(pop_size)

        order = np.argsort(fit)
        pop = [pop[int(i)] for i in order]
        fit = fit[order]

        # record best
        current_best = float(fit[0])
        curve.append(current_best)
        gens_done += 1

        if current_best < (best_so_far - float(early_stop_abs_tol)):
            best_so_far = current_best
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if (
            int(early_stop_patience) > 0
            and gens_done >= int(early_stop_min_iters)
            and stagnant_generations >= int(early_stop_patience)
        ):
            stop_reason = "early_stop"
            break

    x_best = pop[0].astype(int, copy=True)
    best_E = float(fit[0])

    elapsed = float(time.perf_counter() - t0)
    histories = [curve] if keep_histories else None

    info: Dict[str, Any] = {
        "restarts_done": 1,  # ★ no restart
        "iters_total": int(gens_done),
        "evals_total": int(evals_total),
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed,
        "curve_time_sec_list": [elapsed] if keep_histories else None,
        "pop_size": int(pop_size),
        "gens_done": int(gens_done),
        "num_sensors_best": int(x_best.sum()),
    }
    return x_best, best_E, histories, info
