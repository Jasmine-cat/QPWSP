"""Residual-dictionary evaluation; invalid simulations and abstentions differ."""
from __future__ import annotations

import numpy as np


def evaluate_localization(x, R_list, S_list, dist_junc=None, tol=1e-12,
                          return_errors=True, unreachable_penalty=None,
                          softmax_temp=0.1, demand_junc=None,
                          valid_scenarios=None, test_leak_indices=None):
    x = np.asarray(x, float).reshape(-1)
    R, S = np.asarray(R_list, float), np.asarray(S_list, float)
    if R.ndim != 3 or R.shape != S.shape or R.shape[1] != x.size or R.shape[0] == 0:
        raise ValueError("R/S must share shape (pairs, sensors, leak candidates), matching x.")
    if not np.isfinite(x).all() or softmax_temp <= 0:
        raise ValueError("Invalid deployment vector or softmax temperature.")
    pairs, _, candidates = R.shape
    valid = (np.ones((pairs, candidates), bool) if valid_scenarios is None
             else np.asarray(valid_scenarios, bool))
    if valid.shape != (pairs, candidates) or not valid.any():
        raise ValueError("No valid simulation scenarios, or invalid mask dimensions.")
    dictionary_valid = valid.copy()
    if test_leak_indices is not None:
        tested = np.zeros(candidates, bool)
        indices = np.asarray(test_leak_indices, int)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= candidates):
            raise ValueError("Invalid test-leak indices.")
        tested[indices] = True
        valid = valid & tested[None, :]
        if not valid.any():
            raise ValueError("No valid scenarios in the requested test subset.")
    # Failure columns may be NaN only when explicitly excluded by the common mask.
    for k in range(pairs):
        if not np.isfinite(R[k][:, valid[k]]).all() or not np.isfinite(S[k][:, dictionary_valid[k]]).all():
            raise ValueError("A scenario marked valid contains non-finite R/S data.")
    dist = None if dist_junc is None else np.asarray(dist_junc, float)
    if dist is not None and (dist.shape != (candidates, candidates) or not np.isfinite(dist).all()):
        raise ValueError("Distance matrix must match the leak-candidate axis.")
    penalty = (float(unreachable_penalty) if unreachable_penalty is not None
               else max(1.0, float(np.max(dist))) if dist is not None else float(candidates))
    selected = x > 0.5
    hits = np.zeros(valid.shape, float)
    reciprocal_ranks = np.zeros(valid.shape, float)
    topo = np.full(valid.shape, np.nan)
    expected = np.full(valid.shape, np.nan)
    if dist is not None:
        topo[valid] = penalty
        expected[valid] = penalty
    abstained = valid.copy()
    for k in range(pairs):
        if not selected.any():
            continue
        r, s = R[k][selected, :], S[k][selected, :]
        nr, ns = np.linalg.norm(r, axis=0), np.linalg.norm(s, axis=0)
        usable_r = valid[k] & np.isfinite(nr) & (nr > 0)
        usable_s = dictionary_valid[k] & np.isfinite(ns) & (ns > 0)
        if not usable_s.any():
            continue
        ids = np.flatnonzero(usable_s)
        # Normalize only nonzero vectors. Never add synthetic signal to a zero vector.
        dictionary = s[:, ids] / ns[ids]
        for d in np.flatnonzero(usable_r):
            scores = (r[:, d] / nr[d]) @ dictionary
            finite = np.isfinite(scores)
            if not finite.any():
                continue
            eligible, scores = ids[finite], scores[finite]
            tied = eligible[scores >= np.max(scores) - tol]
            # Preserve the original finite-tie success convention. Zero observations
            # never reach this branch and cannot become artificial tied successes.
            if d in tied:
                predicted = int(d)
            elif dist is not None:
                predicted = int(tied[np.argmin(dist[d, tied])])
            else:
                predicted = int(tied[0])
            abstained[k, d] = False
            hits[k, d] = float(predicted == d)
            truth = np.flatnonzero(eligible == d)
            if truth.size:
                rank = 1 + int(np.sum(scores > scores[truth[0]] + tol))
                reciprocal_ranks[k, d] = 1.0 / rank
            if dist is not None:
                topo[k, d] = dist[d, predicted]
                weights = np.exp((scores - np.max(scores)) / softmax_temp)
                weights /= weights.sum()
                expected[k, d] = weights @ dist[d, eligible]
    total_demand = selected_demand = coverage = None
    if demand_junc is not None:
        demand = np.asarray(demand_junc, float).reshape(-1)
        if demand.size != x.size or not np.isfinite(demand).all():
            raise ValueError("Demand vector must be finite and aligned with x.")
        demand = np.maximum(demand, 0)
        total_demand, selected_demand = float(demand.sum()), float(demand[selected].sum())
        denominator = selected.sum() * float(np.median(demand))
        coverage = selected_demand / denominator * 100 if denominator > 0 else 0.0
    success = float(hits[valid].mean())
    return {
        "success_rate": success, "error_rate": 1-success,
        "num_sensors": int(selected.sum()), "mrr": float(reciprocal_ranks[valid].mean()),
        "mean_topo_error": float(topo[valid].mean()) if dist is not None else None,
        "mean_expected_topo_error": float(expected[valid].mean()) if dist is not None else None,
        "topo_err_mat": topo if return_errors and dist is not None else None,
        "expected_topo_err_mat": expected if return_errors and dist is not None else None,
        "mrr_mat": np.where(valid, reciprocal_ranks, np.nan) if return_errors else None,
        "demand_total": total_demand, "demand_selected": selected_demand,
        "water_demand_coverage_rate": coverage,
        "valid_scenario_count": int(valid.sum()), "unlocalized_count": int(abstained.sum()),
    }
