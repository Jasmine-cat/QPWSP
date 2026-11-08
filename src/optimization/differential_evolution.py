from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

from src.qubo.objective import compute_qubo_component_stats, extract_qubo_total, print_qubo_component_stats


def solve_qubo_de_topk(
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Dict[str, Any],
    K_budget: int,
    pop_size: int = 80,
    max_iters: int = 1000,
    F: float = 0.5,
    CR: float = 0.9,
    seed: int = 0,
    time_budget_sec: Optional[float] = None,
    early_stop_patience: int = 200,
    early_stop_min_iters: int = 200,
    early_stop_abs_tol: float = 1e-8,
    keep_histories: bool = True,
) -> Tuple[np.ndarray, float, Optional[List[List[float]]], Dict[str, Any]]:
    """
    Differential Evolution baseline for fixed-budget QUBO.

    The population lives in a continuous score space [0, 1]^N. Each score
    vector is decoded by top-K projection before QUBO evaluation, so every
    accepted solution corresponds to a strict K-hot sensor placement.
    """
    Q, qubo_const = extract_qubo_total(Q_dist_np, const_dist)
    N = int(Q.shape[0])
    if Q.shape != (N, N):
        raise ValueError(f"Q must be square, got {Q.shape}.")
    if not (0 <= int(K_budget) <= N):
        raise ValueError(f"K_budget must be in [0, {N}], got {K_budget}.")
    if int(pop_size) < 4:
        raise ValueError("DE requires pop_size >= 4.")

    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    K_budget = int(K_budget)
    pop_size = int(pop_size)
    max_iters = int(max_iters)
    F = float(F)
    CR = float(CR)

    def time_up() -> bool:
        return (time_budget_sec is not None) and ((time.perf_counter() - t0) >= float(time_budget_sec))

    def topk_indices(score: np.ndarray) -> np.ndarray:
        if K_budget <= 0:
            return np.empty(0, dtype=int)
        if K_budget >= N:
            return np.arange(N, dtype=int)
        return np.argpartition(np.asarray(score, dtype=float), -K_budget)[-K_budget:].astype(int, copy=False)

    def decode_topk(score: np.ndarray) -> np.ndarray:
        x = np.zeros(N, dtype=np.int8)
        if K_budget <= 0:
            return x
        if K_budget >= N:
            x[:] = 1
            return x
        x[topk_indices(score)] = 1
        return x

    def energy_from_x(x: np.ndarray) -> float:
        xf = x.astype(float, copy=False)
        return float(xf @ (Q @ xf)) + float(qubo_const)

    def evaluate(score: np.ndarray) -> float:
        idx = topk_indices(score)
        if idx.size == 0:
            return float(qubo_const)


        return float(Q[np.ix_(idx, idx)].sum()) + float(qubo_const)

    pop = rng.random((pop_size, N), dtype=np.float64)
    fit = np.array([evaluate(pop[i]) for i in range(pop_size)], dtype=float)
    evals_total = int(pop_size)

    best_idx = int(np.argmin(fit))
    best_E = float(fit[best_idx])
    best_score = pop[best_idx].copy()
    best_x = decode_topk(best_score)
    print_qubo_component_stats(
        "[Init-Full][A5-de-best]",
        compute_qubo_component_stats(best_x.astype(float), Q_dist_np, const_dist),
    )

    curve: List[float] = [best_E]
    stagnant = 0
    gens_done = 0
    stop_reason = "max_iters"

    for _gen in range(max_iters):
        if time_up():
            stop_reason = "time_budget"
            break

        for i in range(pop_size):
            if time_up():
                stop_reason = "time_budget"
                break

            candidates = np.delete(np.arange(pop_size), i)
            a_idx, b_idx, c_idx = rng.choice(candidates, size=3, replace=False)
            mutant = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
            mutant = np.clip(mutant, 0.0, 1.0)

            cross_mask = rng.random(N) < CR
            if N > 0:
                cross_mask[int(rng.integers(0, N))] = True
            trial = np.where(cross_mask, mutant, pop[i])
            trial_fit = evaluate(trial)
            evals_total += 1

            if trial_fit <= fit[i]:
                pop[i] = trial
                fit[i] = trial_fit
                if trial_fit < best_E - float(early_stop_abs_tol):
                    best_E = float(trial_fit)
                    best_score = trial.copy()
                    best_x = decode_topk(best_score)
                    stagnant = 0

        gens_done += 1
        current_best = float(np.min(fit))
        if current_best < best_E - float(early_stop_abs_tol):
            best_idx = int(np.argmin(fit))
            best_E = current_best
            best_score = pop[best_idx].copy()
            best_x = decode_topk(best_score)
            stagnant = 0
        else:
            stagnant += 1
        curve.append(best_E)

        if (
            int(early_stop_patience) > 0
            and gens_done >= int(early_stop_min_iters)
            and stagnant >= int(early_stop_patience)
        ):
            stop_reason = "early_stop"
            break

    elapsed = float(time.perf_counter() - t0)
    histories = [curve] if keep_histories else None
    info: Dict[str, Any] = {
        "restarts_done": 1,
        "iters_total": int(gens_done),
        "evals_total": int(evals_total),
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed,
        "curve_time_sec_list": [elapsed] if keep_histories else None,
        "pop_size": int(pop_size),
        "gens_done": int(gens_done),
        "num_sensors_best": int(best_x.sum()),
        "F": float(F),
        "CR": float(CR),
    }
    return best_x.astype(int, copy=False), float(best_E), histories, info
