from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
import time

import numpy as np

from src.qubo.objective import compute_qubo_component_stats, extract_qubo_total, print_qubo_component_stats


def solve_qubo_khot_swap_local_search(
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Dict[str, Any],
    K_budget: int,
    num_restarts: int = 50,
    max_iters: int = 2000,
    seed: int = 0,
    zero_sample: Optional[int] = None,
    time_budget_sec: Optional[float] = None,
    keep_histories: bool = True,
    max_histories: int = 50,
    tol: float = 1e-12,
    # New convergence controls (compatible defaults)
    min_restarts: Optional[int] = None,
    max_restarts: Optional[int] = None,
    restart_patience: int = 12,
    restart_improve_tol: float = 1e-10,
    max_iters_per_restart: Optional[int] = None,
) -> Tuple[np.ndarray, float, Optional[List[List[float]]], Dict[str, Any]]:
    """
    K-hot swap local search with restart-level convergence stop.

    Stop criteria preference:
      1) restart_plateau (after min_restarts)
      2) max_restarts
      3) time_budget (only if explicitly provided)

    Per-restart loop still has a hard cap max_iters_per_restart.
    """
    Q, qubo_const = extract_qubo_total(Q_dist_np, const_dist)
    N = Q.shape[0]
    assert Q.shape == (N, N)

    if max_restarts is None:
        max_restarts = int(num_restarts)
    max_restarts = max(1, int(max_restarts))

    if min_restarts is None:
        min_restarts = min(5, max_restarts)
    min_restarts = max(1, min(int(min_restarts), max_restarts))

    if max_iters_per_restart is None:
        max_iters_per_restart = int(max_iters)
    max_iters_per_restart = max(1, int(max_iters_per_restart))

    restart_patience = max(1, int(restart_patience))
    restart_improve_tol = float(max(restart_improve_tol, 0.0))

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    def time_up() -> bool:
        return (time_budget_sec is not None) and ((time.perf_counter() - t0) >= float(time_budget_sec))

    x_best = None
    best_E = float("inf")
    E_ends: List[float] = []

    best_curve: Optional[List[float]] = None
    best_curve_time_sec: Optional[float] = None
    histories: Optional[List[List[float]]] = [] if keep_histories else None
    curve_time_sec_list: Optional[List[float]] = [] if keep_histories else None

    restarts_done = 0
    iters_total = 0
    max_iters_hit_restarts = 0
    no_improve_restarts = 0
    stop_reason = "max_restarts"

    for _ in range(max_restarts):
        if time_up():
            stop_reason = "time_budget"
            break

        r_t0 = time.perf_counter()

        x = np.zeros(N, dtype=int)
        ones = rng.choice(N, size=K_budget, replace=False)
        x[ones] = 1
        if restarts_done == 0:
            print_qubo_component_stats(
                "[Init-Full][A1-restart0]",
                compute_qubo_component_stats(x.astype(float), Q_dist_np, const_dist),
            )
        Qx = Q @ x
        E = float(x @ Qx) + float(qubo_const)
        curve: List[float] = [E]

        restart_hit_iter_cap = True
        for _ in range(max_iters_per_restart):
            if time_up():
                stop_reason = "time_budget"
                restart_hit_iter_cap = False
                break

            ones_idx = np.flatnonzero(x == 1)
            zeros_idx = np.flatnonzero(x == 0)

            if (zero_sample is not None) and (int(zero_sample) < zeros_idx.size):
                zeros_cand = rng.choice(zeros_idx, size=int(zero_sample), replace=False)
            else:
                zeros_cand = zeros_idx

            best_delta = 0.0
            best_i = None
            best_j = None

            for i in ones_idx:
                ii = int(i)
                qii = Q[ii, ii]
                qx_i = Qx[ii]
                for j in zeros_cand:
                    jj = int(j)
                    delta = 2.0 * (Qx[jj] - qx_i) + (qii + Q[jj, jj] - 2.0 * Q[ii, jj])
                    if delta < best_delta - tol:
                        best_delta = float(delta)
                        best_i = ii
                        best_j = jj

            if best_i is None:
                restart_hit_iter_cap = False
                break

            i = int(best_i)
            j = int(best_j)
            x[i] = 0
            x[j] = 1
            Qx = Qx - Q[:, i] + Q[:, j]
            E = E + best_delta
            iters_total += 1
            curve.append(float(E))

        if restart_hit_iter_cap:
            max_iters_hit_restarts += 1

        E_ends.append(float(E))
        r_elapsed = float(time.perf_counter() - r_t0)
        restarts_done += 1

        if E < best_E - restart_improve_tol:
            best_E = float(E)
            x_best = x.copy()
            best_curve = curve.copy()
            best_curve_time_sec = r_elapsed
            no_improve_restarts = 0
        else:
            no_improve_restarts += 1

        if keep_histories and histories is not None and curve_time_sec_list is not None:
            if len(histories) < int(max_histories):
                histories.append(curve)
                curve_time_sec_list.append(r_elapsed)

        if stop_reason == "time_budget":
            break

        if (restarts_done >= min_restarts) and (no_improve_restarts >= restart_patience):
            stop_reason = "restart_plateau"
            break

    if x_best is None:
        x_best = np.zeros(N, dtype=int)
        x_best[rng.choice(N, size=K_budget, replace=False)] = 1
        best_E = float(x_best @ (Q @ x_best)) + float(qubo_const)
        best_curve = [best_E]
        best_curve_time_sec = 0.0
        if stop_reason != "time_budget":
            stop_reason = "max_iters_hit"

    if (
        stop_reason == "max_restarts"
        and restarts_done > 0
        and max_iters_hit_restarts == restarts_done
    ):
        stop_reason = "max_iters_hit"

    elapsed = float(time.perf_counter() - t0)
    info = {
        "restarts_done": int(restarts_done),
        "iters_total": int(iters_total),
        "stop_reason": stop_reason,
        "elapsed_sec": elapsed,
        "best_curve": best_curve,
        "best_curve_len": 0 if best_curve is None else int(len(best_curve)),
        "best_curve_time_sec": float(best_curve_time_sec) if best_curve_time_sec is not None else None,
        "curve_time_sec_list": curve_time_sec_list,
        "max_iters_hit_restarts": int(max_iters_hit_restarts),
        "no_improve_restarts": int(no_improve_restarts),
        "unique_final_energies": int(len(set(np.round(np.asarray(E_ends, dtype=float), 9)))) if E_ends else 0,
    }
    return x_best, float(best_E), histories, info
