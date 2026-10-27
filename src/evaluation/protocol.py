"""Compatibility entry for the shared residual-dictionary evaluator."""
from src.evaluation.metrics import evaluate_localization


def localization_error_rate_with_topo(
    x, R_list, S_list, dist_junc=None, tol=1e-12, return_errors=True,
    unreachable_penalty=None, softmax_temp=0.1, demand_junc=None,
    valid_scenarios=None, test_leak_indices=None,
):
    return evaluate_localization(
        x, R_list, S_list, dist_junc=dist_junc, tol=tol,
        return_errors=return_errors, unreachable_penalty=unreachable_penalty,
        softmax_temp=softmax_temp, demand_junc=demand_junc,
        valid_scenarios=valid_scenarios,
        test_leak_indices=test_leak_indices,
    )
