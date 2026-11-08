"""Construct QUBO components and report energy statistics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.data.response_stats import compute_leak_importance_and_redundancy_from_H
from src.qubo.demand import normalize_demand_clip, build_Q_size_demand


QUBO_COMPONENT_SPECS = (
    ("cov", "Q_cov", ("c_cov", "const_cov")),
    ("bud", "Q_budget", ("c_budget", "const_budget", "const_bud")),
    ("leak", "Q_leak", ("c_leak", "const_leak")),
    ("demand", "Q_demand", ("c_demand",)),
)


def _validate_square_finite(Q: np.ndarray, name: str) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"[{name}] Q must be a square matrix, got {Q.shape}")
    if not np.isfinite(Q).all():
        raise ValueError(f"[{name}] Q contains non-finite values")
    return Q


def _symmetrize_checked(Q: np.ndarray, name: str) -> np.ndarray:
    Q = _validate_square_finite(Q, name)
    return _validate_square_finite(0.5 * (Q + Q.T), name)


def _const_from_dist(
    const_dist: Optional[Dict[str, Any]], keys: Sequence[str], default: float = 0.0
) -> float:
    if const_dist is None:
        return float(default)
    for key in keys:
        if key in const_dist:
            return float(const_dist[key])
    return float(default)


def extract_qubo_total(
    Q_dist_np: Dict[str, np.ndarray], const_dist: Optional[Dict[str, Any]]
) -> Tuple[np.ndarray, float]:
    """Return the total QUBO matrix and constant."""
    if not isinstance(Q_dist_np, dict) or "Q_total" not in Q_dist_np:
        raise ValueError("Q_dist_np must contain 'Q_total'.")
    return (
        _validate_square_finite(Q_dist_np["Q_total"], "extract_qubo_total.Q_total"),
        _const_from_dist(const_dist, ("c_total", "const")),
    )


def compute_qubo_component_stats(
    x_or_p: np.ndarray,
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute soft or binary energies for each QUBO component."""
    Q_total, c_total = extract_qubo_total(Q_dist_np, const_dist)
    v = np.asarray(x_or_p, dtype=float).reshape(-1)
    if v.size != Q_total.shape[0]:
        raise ValueError(f"vector length={v.size} does not match Q size={Q_total.shape[0]}")

    total_nc = float(v @ Q_total @ v)
    stats: Dict[str, float] = {
        "energy_total_nc": total_nc,
        "energy_total_full": total_nc + float(c_total),
    }
    abs_sum = 0.0
    for short, q_key, const_keys in QUBO_COMPONENT_SPECS:
        if q_key in Q_dist_np:
            Q_i = _validate_square_finite(Q_dist_np[q_key], f"component.{q_key}")
            if Q_i.shape != Q_total.shape:
                raise ValueError(f"{q_key} shape does not match Q_total")
            e_nc = float(v @ Q_i @ v)
        else:
            e_nc = 0.0
        e_full = e_nc + _const_from_dist(const_dist, const_keys)
        stats[f"energy_{short}_nc"] = e_nc
        stats[f"energy_{short}_full"] = e_full
        abs_sum += abs(e_full)

    for short, _q_key, _const_keys in QUBO_COMPONENT_SPECS:
        stats[f"ratio_{short}_full"] = (
            abs(float(stats[f"energy_{short}_full"])) / abs_sum if abs_sum > 0.0 else 0.0
        )
    return stats


def print_qubo_component_stats(label: str, stats: Dict[str, float]) -> None:
    """Print energy contributions from active QUBO components."""
    values = {name: float(stats[f"energy_{name}_full"]) for name, _, _ in QUBO_COMPONENT_SPECS}
    total = sum(abs(v) for v in values.values())
    text = "  ".join(
        f"{name}={value:.6g} ({abs(value) / total if total > 0 else 0.0:.2%})"
        for name, value in values.items()
    )
    print(f"{label} {text}")


def build_Q_cov_from_edges(
    edges_junc_for_cover: Sequence[Tuple[int, int]],
    N: int,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Build the MVC-type edge coverage objective."""
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive")

    edges = set()
    for u, v in edges_junc_for_cover:
        u, v = int(u), int(v)
        if not (0 <= u < N and 0 <= v < N):
            raise ValueError(f"edge ({u}, {v}) is outside [0, {N - 1}]")
        edges.add((u, v) if u <= v else (v, u))

    Q = np.zeros((N, N), dtype=float)
    if not edges:
        return _symmetrize_checked(Q, "build_Q_cov_from_edges"), 0.0

    weight = float(gamma) / float(len(edges))
    const = 0.0
    for i, j in edges:
        const += weight
        Q[i, i] -= weight
        if i == j:
            continue
        Q[j, j] -= weight
        Q[i, j] += 0.5 * weight
        Q[j, i] += 0.5 * weight
    return _symmetrize_checked(Q, "build_Q_cov_from_edges"), float(const)


def build_Q_budget(
    N: int,
    K_budget: int,
    beta: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Build the K-squared-normalized exact-K budget penalty."""
    N, K = int(N), int(K_budget)
    if N <= 0 or K <= 0 or K > N:
        raise ValueError(f"invalid N={N}, K_budget={K}")

    # Normalize the budget penalty by K squared.
    beta_eff = float(beta) / float(K * K)

    Q = np.full((N, N), beta_eff, dtype=float)
    Q[np.diag_indices(N)] = beta_eff * (1.0 - 2.0 * K)
    Q = _symmetrize_checked(Q, "build_Q_budget")
    return Q, float(beta_eff * K * K)


def build_Q_leak_from_H(
    H: np.ndarray,
    K_budget: int,
    alpha: float = 1.0,
    lambda_pair: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Build the response importance and redundancy QUBO component."""
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] == 0 or H.shape[1] == 0:
        raise ValueError(f"H must be a non-empty 2D matrix, got {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("H contains non-finite values")
    K = int(K_budget)
    if K <= 0:
        raise ValueError("K_budget must be positive")

    importance, redundancy = compute_leak_importance_and_redundancy_from_H(H)
    eps_density, eps_importance, eps_pair = 1e-9, 1e-6, 1e-6
    density = np.sum(redundancy, axis=1)
    task_uniqueness = importance / (density + eps_density)
    diag_cost = np.log1p(1.0 / (task_uniqueness + eps_importance))

    pair_denom = np.sqrt(np.maximum(np.outer(importance, importance), 0.0))
    pair_cost = np.log1p(redundancy / (pair_denom + eps_pair))
    pair_cost = np.maximum(0.0, 0.5 * (pair_cost + pair_cost.T))
    np.fill_diagonal(pair_cost, 0.0)

    Q_base = 0.5 * float(lambda_pair) * pair_cost
    Q_base[np.diag_indices(H.shape[0])] = diag_cost
    alpha_eff = float(alpha) / float(K)
    Q = alpha_eff * Q_base
    return _symmetrize_checked(Q, "build_Q_leak_from_H"), 0.0


def _normalize_H_for_qubo(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize response columns and rows before QUBO construction."""
    H = np.asarray(H, dtype=float)
    col_norm = np.linalg.norm(H, axis=0, keepdims=True)
    H_col = np.divide(H, col_norm + eps, out=np.zeros_like(H), where=col_norm > eps)
    row_norm = np.linalg.norm(H_col, axis=1, keepdims=True)
    H_norm = np.divide(H_col, row_norm + eps, out=np.zeros_like(H_col), where=row_norm > eps)
    if not np.isfinite(H_norm).all():
        raise ValueError("normalized H contains non-finite values")
    return H_norm


def build_Q_total(
    *,
    N: int,
    K_budget: int,
    H: Optional[np.ndarray] = None,
    edges_for_mvc: Optional[Sequence[Tuple[int, int]]] = None,
    alpha_bud: float = 1.0,
    alpha_leak_h: float = 0.0,
    alpha_mvc: float = 0.0,
    alpha_demand: float = 0.0,
    demand_junc: Optional[np.ndarray] = None,
    demand_clip_quantile: float = 0.95,
    lambda_pair: float = 2.25,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Combine budget, response, coverage, and optional demand components."""
    N, K = int(N), int(K_budget)
    if N <= 0 or K <= 0:
        raise ValueError(f"invalid N={N}, K_budget={K}")

    Q_budget = np.zeros((N, N), dtype=float)
    Q_cov = np.zeros((N, N), dtype=float)
    Q_leak = np.zeros((N, N), dtype=float)
    Q_demand = np.zeros((N, N), dtype=float)
    c_budget = c_cov = c_leak = 0.0

    if float(alpha_bud) != 0.0:
        Q_budget, c_budget = build_Q_budget(N=N, K_budget=K, beta=float(alpha_bud))
    if float(alpha_mvc) != 0.0:
        if edges_for_mvc is None:
            raise ValueError("edges_for_mvc is required when alpha_mvc is non-zero")
        Q_cov, c_cov = build_Q_cov_from_edges(edges_for_mvc, N=N, gamma=float(alpha_mvc))
    if float(alpha_leak_h) != 0.0:
        if H is None:
            raise ValueError("H is required when alpha_leak_h is non-zero")
        H_norm = _normalize_H_for_qubo(H)
        if H_norm.ndim != 2 or H_norm.shape[0] != N or H_norm.shape[1] == 0:
            raise ValueError(f"H must have {N} sensor rows and nonempty leak columns, got {H_norm.shape}")
        Q_leak, c_leak = build_Q_leak_from_H(
            H_norm,
            K_budget=K,
            alpha=float(alpha_leak_h),
            lambda_pair=float(lambda_pair),
        )

    if not np.isfinite(alpha_demand):
        raise ValueError("alpha_demand must be finite")
    if float(alpha_demand) != 0.0:
        if demand_junc is None:
            raise ValueError("demand_junc is required when alpha_demand is non-zero")
        demand = np.asarray(demand_junc, dtype=float).reshape(-1)
        if demand.size != N or not np.isfinite(demand).all():
            raise ValueError("demand_junc must be finite and aligned with the QUBO rows")
        if not 0 < float(demand_clip_quantile) <= 1:
            raise ValueError("demand_clip_quantile must be in (0, 1]")
        weights, _ = normalize_demand_clip(demand, clip_q=float(demand_clip_quantile))
        Q_demand, _ = build_Q_size_demand(N, weights, lambda_dem=float(alpha_demand))

    Q_total = _symmetrize_checked(Q_budget + Q_cov + Q_leak + Q_demand, "build_Q_total")
    c_total = float(c_budget + c_cov + c_leak)
    return {
        "Q_total": Q_total,
        "Q_budget": Q_budget,
        "Q_cov": Q_cov,
        "Q_leak": Q_leak,
        "Q_demand": Q_demand,
    }, {
        "c_total": c_total,
        "const": c_total,
        "c_budget": float(c_budget),
        "c_cov": float(c_cov),
        "c_leak": float(c_leak),
        "c_demand": 0.0,
        "const_bud": float(c_budget),
        "const_budget": float(c_budget),
        "const_cov": float(c_cov),
        "const_leak": float(c_leak),
    }
