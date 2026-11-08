"""Optional nodal-demand preference retained from the historical implementation.

The adopted experiment profiles leave its coefficient at zero.
"""
import numpy as np
from typing import Tuple

def normalize_demand_clip(
    d_junction: np.ndarray,
    *,
    clip_q: float = 0.95,
    eps: float = 1e-12,
    use_positive_for_clip: bool = True,
    use_positive_for_scale: bool = True,
) -> Tuple[np.ndarray, float]:
    """Clip and normalize junction base demands to the unit interval."""
    d = np.asarray(d_junction, dtype=float).reshape(-1)  # (N_j,)
    if d.size == 0:
        return d.copy(), 0.0

    if use_positive_for_clip:
        base = d[d > 0]
    else:
        base = d
    if base.size == 0:

        return np.zeros_like(d), 0.0


    clip_value = float(np.quantile(base, clip_q))
    d_clip = np.clip(d, 0.0, clip_value)


    if use_positive_for_scale:
        scale_base = d_clip[d_clip > 0]
        mx = float(np.max(scale_base)) if scale_base.size > 0 else 0.0
    else:
        mx = float(np.max(d_clip))

    if mx <= eps:
        d_norm = np.zeros_like(d_clip)
    else:
        d_norm = d_clip / (mx + eps)

    return d_norm.astype(float), clip_value


def build_Q_size_demand(
    N: int,
    w: np.ndarray,
    lambda_dem: float = 1.0,
    *,
    lambda_norm: str = "N",
    clip_w: bool = True,
    return_const: bool = True,
) -> Tuple[np.ndarray, float]:
    """Build the optional demand-based diagonal QUBO component."""
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != N:
        raise ValueError(f"w.shape[0]={w.shape[0]} != N={N}")

    if clip_w:
        w = np.clip(w, 0.0, 1.0)

    lambda_dem = float(lambda_dem)

    if lambda_norm == "none":
        lambda_eff = lambda_dem
    elif lambda_norm.upper() == "N":
        lambda_eff = lambda_dem / float(N)
    else:
        raise ValueError(f"Unknown lambda_norm='{lambda_norm}', choose from: none, N")

    c = lambda_eff * (1.0 - w)  # (N,)

    Q = np.zeros((N, N), dtype=float)
    Q[np.diag_indices(N)] = c

    if return_const:
        return Q, 0.0
    else:
        return Q, 0.0
