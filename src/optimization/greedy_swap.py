from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
import time

import numpy as np

from src.qubo.objective import compute_qubo_component_stats, extract_qubo_total, print_qubo_component_stats


def solve_qubo_greedy_swap_refine(
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Dict[str, Any],
    K_budget: int,
    seed: int = 0,
    swap_max_rounds: int = 30,
    swap_zero_sample: Optional[int] = None,
    improve_tol: float = 1e-12,
    random_tie_break: bool = True,
    time_budget_sec: Optional[float] = None,
    keep_histories: bool = True,
) -> Tuple[np.ndarray, float, Optional[List[List[float]]], Dict[str, Any]]:
    """
    A4 baseline:
      1) greedy marginal-add construction to K-hot
      2) 1-out-1-in swap refinement until no improving swap or max rounds
    Objective: minimize x^T Q x + qubo_const
    """
    Q, qubo_const = extract_qubo_total(Q_dist_np, const_dist)
    N = Q.shape[0]
    assert Q.shape == (N, N)
    if not (0 <= int(K_budget) <= N):
        raise ValueError(f"K_budget must be in [0, {N}], got {K_budget}")

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    diag_q = np.diag(Q)
    improve_tol = float(max(improve_tol, 0.0))

    def time_up() -> bool:
        return (time_budget_sec is not None) and ((time.perf_counter() - t0) >= float(time_budget_sec))

    x = np.zeros(N, dtype=int)
    Qx = np.zeros(N, dtype=float)
    E = float(qubo_const)

    curve: List[float] = [E]
    iters_total = 0
    swap_rounds_done = 0
    stop_reason = "swap_converged"

    # Phase-1: greedy build
    for _ in range(int(K_budget)):
        if time_up():
            stop_reason = "time_budget"
            break

        zeros_idx = np.flatnonzero(x == 0)
        deltas = 2.0 * Qx[zeros_idx] + diag_q[zeros_idx]
        best_delta = float(np.min(deltas))

        if random_tie_break:
            tied = np.where(deltas <= best_delta + improve_tol)[0]
            pick = int(rng.choice(tied))
        else:
            pick = int(np.argmin(deltas))

        j = int(zeros_idx[pick])
        chosen_delta = float(2.0 * Qx[j] + diag_q[j])
        x[j] = 1
        Qx = Qx + Q[:, j]
        E = E + chosen_delta
        iters_total += 1

        if keep_histories:
            curve.append(float(E))

    # Keep strict K-hot (mainly defensive for explicit time_budget mode)
    if int(x.sum()) < int(K_budget):
        need = int(K_budget - x.sum())
        zeros_idx = np.flatnonzero(x == 0)
        if need > 0:
            fill = rng.choice(zeros_idx, size=need, replace=False)
            for j in fill:
                jj = int(j)
                chosen_delta = float(2.0 * Qx[jj] + diag_q[jj])
                x[jj] = 1
                Qx = Qx + Q[:, jj]
                E = E + chosen_delta
                iters_total += 1
                if keep_histories:
                    curve.append(float(E))

    print_qubo_component_stats(
        "[Init-Full][A4-after-greedy]",
        compute_qubo_component_stats(x.astype(float), Q_dist_np, const_dist),
    )

    # Phase-2: swap refine
    if stop_reason != "time_budget" and int(K_budget) > 0 and int(K_budget) < N:
        for _ in range(int(max(1, swap_max_rounds))):
            if time_up():
                stop_reason = "time_budget"
                break

            ones_idx = np.flatnonzero(x == 1)
            zeros_idx = np.flatnonzero(x == 0)
            if (swap_zero_sample is not None) and (int(swap_zero_sample) < zeros_idx.size):
                zeros_cand = rng.choice(zeros_idx, size=int(swap_zero_sample), replace=False)
            else:
                zeros_cand = zeros_idx

            best_delta = 0.0
            best_pairs: List[Tuple[int, int]] = []

            for i in ones_idx:
                ii = int(i)
                qii = Q[ii, ii]
                qx_i = Qx[ii]
                for j in zeros_cand:
                    jj = int(j)
                    delta = 2.0 * (Qx[jj] - qx_i) + (qii + Q[jj, jj] - 2.0 * Q[ii, jj])
                    if delta < best_delta - improve_tol:
                        best_delta = float(delta)
                        best_pairs = [(ii, jj)]
                    elif random_tie_break and abs(delta - best_delta) <= improve_tol and best_pairs:
                        best_pairs.append((ii, jj))

            if not best_pairs:
                stop_reason = "swap_converged"
                break

            if random_tie_break:
                i, j = best_pairs[int(rng.integers(0, len(best_pairs)))]
            else:
                i, j = best_pairs[0]

            x[i] = 0
            x[j] = 1
            Qx = Qx - Q[:, i] + Q[:, j]
            E = E + best_delta
            swap_rounds_done += 1
            iters_total += 1
            if keep_histories:
                curve.append(float(E))
        else:
            stop_reason = "swap_max_rounds"

    elapsed = float(time.perf_counter() - t0)
    histories = [curve] if keep_histories else None
    info = {
        "restarts_done": 1,
        "iters_total": int(iters_total),
        "iters_done": int(iters_total),
        "swap_rounds_done": int(swap_rounds_done),
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed,
        "curve_time_sec_list": [elapsed] if keep_histories else None,
        "num_sensors_best": int(x.sum()),
    }
    return x.astype(int, copy=False), float(E), histories, info
