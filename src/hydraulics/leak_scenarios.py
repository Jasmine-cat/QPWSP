"""Shared constant-demand leakage and basic EPANET result handling."""

from __future__ import annotations

import copy
import glob
import os
from contextlib import contextmanager
from pathlib import Path
import re
import tempfile
import time
import uuid

import numpy as np
from wntr.sim import EpanetSimulator


def _attach_constant_leak(wn, node_id: str, flow_m3s: float):
    flow = float(flow_m3s)
    multiplier = float(wn.options.hydraulic.demand_multiplier)
    if not np.isfinite(flow) or flow <= 0:
        raise ValueError("Leak flow must be finite and positive (m3/s).")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("A positive demand multiplier is required for demand-based leakage.")
    node = wn.get_node(str(node_id))
    if node.node_type != "Junction":
        raise ValueError(f"Leak node {node_id} is not a junction.")
    pattern = f"constant_leak_{uuid.uuid4().hex[:12]}"
    wn.add_pattern(pattern, [1.0])
    index = len(node.demand_timeseries_list)
    node.add_demand(flow / multiplier, pattern, category="synthetic_leak")
    actual = float(node.demand_timeseries_list[index].at(0)) * multiplier
    if not np.isclose(actual, flow, rtol=1e-10, atol=0):
        del node.demand_timeseries_list[index]
        wn._pattern_reg.remove_usage(pattern, (node.name, "Junction"))
        wn.remove_pattern(pattern)
        raise ValueError("Configured constant leak does not match the requested flow.")
    return node, index, pattern


@contextmanager
def constant_leak(wn, node_id: str, flow_m3s: float):
    """Add a separate all-ones demand; restore the model even after failure."""
    node, index, pattern = _attach_constant_leak(wn, node_id, flow_m3s)
    try:
        yield wn
    finally:
        del node.demand_timeseries_list[index]
        wn._pattern_reg.remove_usage(pattern, (node.name, "Junction"))
        wn.remove_pattern(pattern)


def clone_with_constant_leak(wn_base, node_id: str, flow_lps: float):
    wn = copy.deepcopy(wn_base)
    _attach_constant_leak(wn, node_id, float(flow_lps) / 1000.0)
    return wn


def checked_pressure_result(result, node_names, *, required_time_s=None):
    """Check only completeness/finiteness of the pressure window actually used."""
    frame = result.node["pressure"]
    names = list(map(str, node_names))
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Duplicate node IDs or report times in pressure results.")
    missing = set(names) - set(map(str, frame.columns))
    if missing:
        raise ValueError(f"Missing pressure nodes: {sorted(missing)[:5]}")
    times = np.asarray(frame.index, dtype=float)
    if times.size == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Pressure results have an empty or invalid time axis.")
    if required_time_s is not None:
        target = float(required_time_s)
        if not np.any(np.isclose(times, target, rtol=0, atol=1e-9)):
            raise ValueError(f"Simulation did not reach required report time {target:g}s.")
        frame = frame.loc[times <= target + 1e-9]
    elif getattr(result, "error_code", None) is not None:
        raise RuntimeError(f"Incomplete hydraulic result: {result.error_code}")
    values = frame.loc[:, names].to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite pressure in the evaluation window.")
    return values, np.asarray(frame.index, dtype=float)


def run_pressure_array(wn, node_names, *, tag="pressure", max_retries=0,
                       retry_sleep_sec=0.05, required_time_s=None):
    """Run/check one flow; the asset builder owns any group-wide flow fallback."""
    if required_time_s is None:
        required_time_s = getattr(wn, "_experiment_evaluation_time_s", None)
    last_error = None
    for attempt in range(max(0, int(max_retries)) + 1):
        with tempfile.TemporaryDirectory(prefix="gnn_qubo_") as folder:
            prefix = str(Path(folder) / "simulation")
            try:
                result = EpanetSimulator(wn).run_sim(
                    file_prefix=prefix, version=2.2, convergence_error=False,
                )
                # Ordinary warnings are retained by WNTR; explicit unbalanced
                # solutions are not accepted as valid pressure observations.
                report = Path(prefix + ".rpt")
                if report.exists():
                    text = report.read_text(encoding="utf-8", errors="replace")
                    for match in re.finditer(r"WARNING[^\n]*unbalanced[^\n]*", text, re.IGNORECASE):
                        clock = re.search(r"(\d+):(\d+):(\d+)", match.group())
                        at = (3600*int(clock[1]) + 60*int(clock[2]) + int(clock[3])) if clock else None
                        if required_time_s is None or at is None or at <= float(required_time_s):
                            raise RuntimeError("Unbalanced hydraulics in the evaluation window.")
                return checked_pressure_result(result, node_names,
                                               required_time_s=required_time_s)
            except Exception as error:
                last_error = error
        if attempt < int(max_retries):
            time.sleep(max(0.0, float(retry_sleep_sec)))
    raise RuntimeError(f"{tag}: hydraulic simulation failed: {last_error}") from last_error
