import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import wntr
from src.hydraulics.leak_scenarios import constant_leak

from src.data.loaders import unit_to_LPS_factor
from src.hydraulics.residual_generator import (
    _apply_time_settings_from_meta,
    _load_step1_graph_meta,
    _run_pressure_array_with_retry,
    _strict_find_time_index,
)


def _resolve_q0_json_path(inp_name: str, q0_json_path: Optional[str], node_features_dir: Optional[str]) -> str:
    if q0_json_path is not None:
        return os.path.abspath(q0_json_path)
    if node_features_dir is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        node_features_dir = os.path.join(repo_root, "data", "features")
    path = os.path.join(node_features_dir, f"{inp_name}q0.json")
    return os.path.abspath(path)


def _load_q0_info(q0_json_path: str) -> dict:
    if not os.path.exists(q0_json_path):
        raise FileNotFoundError(
            f"q0 JSON file not found: {q0_json_path}。"
            "Run network preparation or specify q0_json_path explicitly."
        )
    with open(q0_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_q_ref_lps(q0_info: dict) -> tuple[float, str]:
    beta_status = str(q0_info.get("beta_select_status", "ok")).strip().lower()
    if beta_status == "ok":
        if "q0_LPS" in q0_info:
            q_ref_lps = float(q0_info["q0_LPS"])
        else:
            q0_inp = float(q0_info["q0_inp"])
            units = str(q0_info.get("units", "LPS"))
            q_ref_lps = q0_inp * float(unit_to_LPS_factor(units))
        return q_ref_lps, "q0_LPS"

    if "LDD_lps" not in q0_info:
        raise KeyError(
            "The fallback q_ref requires LDD_lps in q0.json."
            f"Current keys={list(q0_info.keys())}"
        )
    return float(q0_info["LDD_lps"]), "LDD_lps_fallback"


def _save_npz_atomic(out_npz: str, **arrays: Any) -> None:
    out_npz = os.path.abspath(out_npz)
    out_dir = os.path.dirname(out_npz)
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=Path(out_npz).stem + "_", suffix=".tmp.npz", dir=out_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, out_npz)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def generate_H_from_inp(
    inp_path: str,
    out_npz: str,
    out_csv: Optional[str] = None,
    *,
    q0_json_path: Optional[str] = None,
    node_features_dir: Optional[str] = None,
    h_ref_factor: float = 0.2,
    h_high_failure_retry_factor: float = 0.1,
    _h_retry_depth: int = 0,
    overwrite: bool = False,
    h_recovery_factors: tuple[float, ...] = (0.5, 0.3, 0.1),
) -> Dict[str, Any]:
    if Path(out_npz).exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing H asset: {out_npz}")
    inp_path = os.path.abspath(inp_path)
    inp_name = Path(inp_path).stem

    print(f"[H] Loading network: {inp_path}")
    wn = wntr.network.WaterNetworkModel(inp_path)
    all_junction_names = list(wn.junction_name_list)
    all_junction_nodes = [wn.get_node(name) for name in all_junction_names]
    all_base_demands_m3s = np.array(
        [jn.demand_timeseries_list[0].base_value for jn in all_junction_nodes],
        dtype=float,
    )
    keep_junction = np.isfinite(all_base_demands_m3s) & (all_base_demands_m3s >= 0.0)
    skipped_negative = [
        str(name) for name, keep in zip(all_junction_names, keep_junction) if not bool(keep)
    ]
    if skipped_negative:
        print(
            f"[H] Excluded {len(skipped_negative)} junctions with negative demand from the candidate set: "
            f"{skipped_negative[:8]}"
            f"{'...' if len(skipped_negative) > 8 else ''}"
        )
    junction_names = [name for name, keep in zip(all_junction_names, keep_junction) if bool(keep)]
    if len(junction_names) == 0:
        raise ValueError(f"[H][{inp_name}] No eligible junctions with nonnegative demand remain.")

    junction_nodes = [wn.get_node(name) for name in junction_names]
    base_demands_m3s = np.array(
        [jn.demand_timeseries_list[0].base_value for jn in junction_nodes],
        dtype=float,
    )

    q0_json_path = _resolve_q0_json_path(inp_name, q0_json_path, node_features_dir)
    q0_info = _load_q0_info(q0_json_path)
    q_ref_base_lps, q_ref_base_source = _resolve_q_ref_lps(q0_info)
    # Use a small reference perturbation for the H design asset.
    q_ref_lps = float(q_ref_base_lps) * float(h_ref_factor)
    q_ref_source = f"{q_ref_base_source}*h_ref_factor"
    if q_ref_lps <= 0:
        raise ValueError(f"[H][{inp_name}] q_ref_lps must be positive; got {q_ref_lps}")
    q_ref_m3s = float(q_ref_lps) / 1000.0
    # Regularize the reference-flow denominator against numerical instability.
    q_ref_denominator_eps_lps = 1e-9
    q_ref_denominator_lps = float(q_ref_lps) + q_ref_denominator_eps_lps

    if node_features_dir is None:
        node_features_dir = os.path.dirname(q0_json_path)
    graph_meta = _load_step1_graph_meta(node_features_dir, inp_name)
    duration_cfg_s, report_step_cfg_s, hyd_step_cfg_s = _apply_time_settings_from_meta(wn, graph_meta)

    selected_ptime_method = str(
        q0_info.get(
            "selected_ptime_method",
            graph_meta.get("selected_ptime_method", "min-mean-max-std-window"),
        )
    )
    selected_ptime_s = q0_info.get("selected_ptime_s", graph_meta.get("selected_ptime_s", None))
    if selected_ptime_s is None:
        raise KeyError(
            f"[H][{inp_name}] selected_ptime_s is missing from q0 JSON and graph metadata."
        )
    selected_ptime_s = int(selected_ptime_s)

    print(
        f"[H] q_ref_base_lps={q_ref_base_lps:.6g} (source={q_ref_base_source}), "
        f"h_ref_factor={h_ref_factor:.6g}, q_ref_lps={q_ref_lps:.6g}, "
        f"selected_ptime_s={selected_ptime_s}"
    )

    wn._experiment_evaluation_time_s = selected_ptime_s
    np_origin, times = _run_pressure_array_with_retry(
        wn,
        junction_names,
        tag=f"H_baseline_{inp_name}",
        max_retries=0,
    )
    selected_idx = _strict_find_time_index(times, selected_ptime_s, inp_name)

    deltaP_ref = np.full((len(junction_names), len(junction_names)), np.nan)
    H = np.full_like(deltaP_ref, np.nan)
    failed_details: list[dict[str, Any]] = []
    failure_abort_ratio = 0.30

    q_by_node_lps = np.full(len(junction_names), q_ref_lps, dtype=float)
    retry_by_node = np.zeros(len(junction_names), dtype=int)

    def simulate_column(leak_idx, flow_lps):
        node_name = junction_names[leak_idx]
        with constant_leak(wn, node_name, flow_lps / 1000.0):
            pressure, leak_times = _run_pressure_array_with_retry(
                wn, junction_names, tag=f"H_leak_{inp_name}_{leak_idx}", max_retries=0,
            )
        ti = _strict_find_time_index(leak_times, selected_ptime_s, inp_name)
        if not np.isclose(leak_times[ti], times[selected_idx]):
            raise ValueError("Baseline and leak report times differ.")
        residual = np_origin[selected_idx, :] - pressure[ti, :]
        if not np.isfinite(residual).all():
            raise ValueError("Non-finite H response.")
        deltaP_ref[:, leak_idx] = residual
        H[:, leak_idx] = residual / (flow_lps + q_ref_denominator_eps_lps)

    failed_indices = []
    for leak_idx in range(len(junction_names)):
        try:
            simulate_column(leak_idx, q_by_node_lps[leak_idx])
        except Exception as error:
            failed_indices.append(leak_idx)
            failed_details.append({"leak_index": leak_idx,
                                   "leak_node": junction_names[leak_idx],
                                   "attempt": 0, "error_repr": repr(error)})
    if failed_indices:
        factors = tuple(float(value) for value in h_recovery_factors)
        if (not factors or len(factors) > 3 or
                any(not 0 < value < 1 for value in factors) or
                any(left <= right for left, right in zip(factors, factors[1:]))):
            raise ValueError("H recovery requires up to three decreasing factors between zero and one.")
        for leak_idx in failed_indices:
            for attempt, factor in enumerate(factors, start=1):
                q_by_node_lps[leak_idx] = q_ref_lps * factor
                retry_by_node[leak_idx] = attempt
                try:
                    simulate_column(leak_idx, q_by_node_lps[leak_idx])
                    break
                except Exception as error:
                    failed_details.append({"leak_index": leak_idx,
                                           "leak_node": junction_names[leak_idx],
                                           "attempt": attempt,
                                           "q_lps": float(q_by_node_lps[leak_idx]),
                                           "error_repr": repr(error)})
    valid_leaks = np.isfinite(H).all(axis=0)
    if not valid_leaks.any():
        raise RuntimeError("No valid H leak simulations after recovery.")
    if float((~valid_leaks).mean()) > failure_abort_ratio:
        raise RuntimeError("H leak failure ratio exceeds 30% after node-level recovery.")
    leak_names = np.asarray(junction_names)[valid_leaks]
    H = H[:, valid_leaks]
    deltaP_ref = deltaP_ref[:, valid_leaks]

    out_npz = os.path.abspath(out_npz)
    if out_csv is None:
        out_csv = os.path.splitext(out_npz)[0] + ".csv"
    out_csv = os.path.abspath(out_csv)

    _save_npz_atomic(
        out_npz,
        H=H,
        deltaP_ref=deltaP_ref,
        junction_names=np.array(junction_names),
        leak_node_names=leak_names,
        leak_model=np.array("constant_additional_demand"),
        q_ref_lps=np.array(float(q_ref_lps), dtype=float),
        q_ref_base_lps=np.array(float(q_ref_base_lps), dtype=float),
        q_ref_source=np.array(str(q_ref_source)),
        q_ref_base_source=np.array(str(q_ref_base_source)),
        h_ref_factor=np.array(float(h_ref_factor), dtype=float),
        h_high_failure_retry_factor=np.array(float(h_high_failure_retry_factor), dtype=float),
        h_retry_depth=np.array(int(retry_by_node.max()), dtype=int),
        recovery_mode=np.array("failed_nodes_only"),
        h_recovery_factors=np.asarray(h_recovery_factors, dtype=float),
        q_ref_lps_by_leak=q_by_node_lps[valid_leaks],
        q_ref_denominator_lps_by_leak=q_by_node_lps[valid_leaks] + q_ref_denominator_eps_lps,
        retry_count_by_leak=retry_by_node[valid_leaks],
        excluded_failed_leak_nodes=np.asarray(junction_names)[~valid_leaks],
        q_ref_denominator_lps=np.array(float(q_ref_denominator_lps), dtype=float),
        q_ref_denominator_eps_lps=np.array(float(q_ref_denominator_eps_lps), dtype=float),
        excluded_negative_demand_junctions=np.array(skipped_negative, dtype=str),
        node_level_failures=np.array([json.dumps(item, ensure_ascii=False) for item in failed_details], dtype=str),
        failed_fill_value=np.array(np.nan, dtype=float),
        failure_abort_ratio=np.array(0.30, dtype=float),
        selected_ptime_s=np.array(int(selected_ptime_s), dtype=int),
        selected_ptime_method=np.array(str(selected_ptime_method)),
        units=np.array(str(q0_info.get("units", "UNKNOWN"))),
        duration_cfg_s=np.array(-1 if duration_cfg_s is None else int(duration_cfg_s), dtype=int),
        report_step_cfg_s=np.array(int(report_step_cfg_s), dtype=int),
        hyd_step_cfg_s=np.array(int(hyd_step_cfg_s), dtype=int),
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(H, index=junction_names, columns=leak_names).to_csv(out_csv, encoding="utf-8")

    print(f"[H] Saved: {out_npz}, {out_csv}")
    return {
        "inp_name": inp_name,
        "junction_count": int(len(junction_names)),
        "q_ref_lps": float(q_ref_lps),
        "q_ref_base_lps": float(q_ref_base_lps),
        "h_ref_factor": float(h_ref_factor),
        "h_retry_depth": int(retry_by_node.max()),
        "q_ref_source": str(q_ref_source),
        "selected_ptime_s": int(selected_ptime_s),
        "selected_ptime_method": str(selected_ptime_method),
        "out_npz": out_npz,
        "out_csv": out_csv,
    }


def _iter_inp_paths(target: str) -> list[str]:
    target = os.path.abspath(target)
    if os.path.isdir(target):
        return sorted(glob.glob(os.path.join(target, "*.inp")))
    return [target]


def main() -> None:
    raise SystemExit("Use python -m script.run_experiments assets --help")
