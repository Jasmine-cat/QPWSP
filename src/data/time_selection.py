from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


SELECTED_PTIME_METHOD = "min_mean_then_max_std_within_pm2"
SELECTED_PTIME_WINDOW_STEPS = 2


def _to_int_if_close(value: Any) -> Any:
    try:
        fv = float(value)
    except Exception:
        return value
    iv = int(round(fv))
    if abs(fv - iv) <= 1e-9:
        return iv
    return value


def _choose_pressure_columns(
    base_pressure: pd.DataFrame,
    node_feats: Optional[pd.DataFrame],
) -> list[str]:
    cols = [str(c) for c in base_pressure.columns]
    if node_feats is None or ("type" not in node_feats.columns):
        return cols

    junc_nodes = node_feats[node_feats["type"] == "Junction"].index
    junc_cols = [str(n) for n in junc_nodes if str(n) in base_pressure.columns]
    return junc_cols if len(junc_cols) > 0 else cols


def compute_selected_ptime(
    base_pressure: pd.DataFrame,
    node_feats: Optional[pd.DataFrame] = None,
    *,
    window_steps: int = SELECTED_PTIME_WINDOW_STEPS,
    prefer_high_std: bool = True,
    verbose: bool = False,
    log_prefix: str = "[min_ptime]",
) -> Any:
    """
    Unified selected pressure time policy:
    1) pick t_min by minimum mean pressure over selected nodes;
    2) pick t_best by maximum std within +/- window_steps around t_min.
    """
    if base_pressure is None or base_pressure.empty:
        raise ValueError("base_pressure is empty; cannot select pressure time.")

    cols = _choose_pressure_columns(base_pressure, node_feats)
    if len(cols) == 0:
        raise ValueError("No valid pressure columns are available for time selection.")

    X = base_pressure[cols]
    avg_p = X.mean(axis=1)
    if avg_p.empty:
        raise ValueError("avg pressure series is empty; cannot select pressure time.")

    t_pos_min = int(np.argmin(avg_p.to_numpy(dtype=float)))
    t_min = avg_p.index[t_pos_min]

    best_pos = t_pos_min
    if prefer_high_std and int(window_steps) > 0 and len(avg_p) > 1:
        std_p = X.std(axis=1, ddof=0)
        lo = max(0, t_pos_min - int(window_steps))
        hi = min(len(avg_p) - 1, t_pos_min + int(window_steps))
        cand_pos = np.arange(lo, hi + 1, dtype=int)

        std_vals = std_p.iloc[cand_pos].to_numpy(dtype=float)
        std_vals = np.where(np.isfinite(std_vals), std_vals, -np.inf)
        best_std = np.max(std_vals)
        tie_pos = cand_pos[np.where(std_vals >= best_std - 1e-12)[0]]

        if tie_pos.size == 1:
            best_pos = int(tie_pos[0])
        else:
            tie_avg = avg_p.iloc[tie_pos].to_numpy(dtype=float)
            best_pos = int(tie_pos[int(np.argmin(tie_avg))])

    t_best = avg_p.index[best_pos]
    t_best_out = _to_int_if_close(t_best)

    if verbose:
        std_series = X.std(axis=1, ddof=0)
        try:
            avg_at_best = float(avg_p.iloc[best_pos])
        except Exception:
            avg_at_best = float("nan")
        try:
            std_at_best = float(std_series.iloc[best_pos])
        except Exception:
            std_at_best = float("nan")
        print(
            f"{log_prefix} min-mean at t={_to_int_if_close(t_min)}, "
            f"choose t_best={t_best_out} within +/-{int(window_steps)} by max-std. "
            f"(avg={avg_at_best:.4f}, std={std_at_best:.4f})"
        )

    return t_best_out
