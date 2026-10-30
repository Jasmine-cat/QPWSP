"""Cache multi-strength design pressures for sampled leak locations."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import wntr

from src.hydraulics.leak_scenarios import constant_leak, run_pressure_array


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _start_worker(settings, flows_lps):
    global _design_model, _design_sensors, _design_flows, _design_time
    _design_model = wntr.network.WaterNetworkModel(settings["inp_path"])
    for key, attr in [("duration_cfg_s", "duration"), ("report_step_cfg_s", "report_timestep"),
                      ("hyd_step_cfg_s", "hydraulic_timestep")]:
        if settings[key] >= 0:
            setattr(_design_model.options.time, attr, settings[key])
    _design_sensors = settings["sensor_ids"]
    _design_flows = flows_lps
    _design_time = settings["selected_ptime_s"]


def _simulate_node(node):
    pressures = np.full((len(_design_flows), len(_design_sensors)), np.nan)
    failures = []
    for level, flow in enumerate(_design_flows):
        try:
            with constant_leak(_design_model, node, float(flow) / 1000.0):
                values, times = run_pressure_array(
                    _design_model, _design_sensors, required_time_s=_design_time,
                    tag=f"information_{node}_{level}", max_retries=0)
            match = np.flatnonzero(np.isclose(times, _design_time, rtol=0, atol=1e-9))
            if len(match) != 1:
                raise ValueError("Missing or duplicated design pressure time.")
            pressures[level] = values[match[0]]
        except Exception as error:
            failures.append({"node": node, "level": level, "flow_lps": float(flow),
                             "reason": str(error)})
    return pressures, failures


def generate_information_design(*, inp_path, H_path, sensor_ids, seeds, out_path,
                                leak_fraction=1/3, strength_range=(0.25, 2.0),
                                strength_levels=3, workers=4, retry_factor=0.1):
    """Each seed sees only its sampled locations, with all pressure columns.

    Fixed strength strata use their midpoints. Only the seed-selected union is
    simulated; repeated node/strength cases are cached across seeds. Per-layout
    available cases and actual unique simulation calls remain separate costs.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids, seeds = list(map(str, sensor_ids)), list(map(int, seeds))
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Unique candidate IDs and seeds are required.")
    size = max(2, int(np.floor(len(ids) * float(leak_fraction) + 0.5)))
    if size > len(ids) or not 0 < float(leak_fraction) <= 1:
        raise ValueError("Invalid sampled leak fraction.")
    low, high = map(float, strength_range)
    levels = int(strength_levels)
    if levels < 2 or not 0 < low < high:
        raise ValueError("Multiple strengths in a positive interval are required.")
    factors = low + (np.arange(levels) + .5) * (high-low) / levels
    subsets = np.stack([np.sort(np.random.default_rng(seed).choice(len(ids), size, replace=False))
                        for seed in seeds])
    union = np.unique(subsets)
    with np.load(H_path, allow_pickle=False) as asset:
        q0_lps = float(asset["q_ref_base_lps"])
        timing = {key: int(asset[key]) for key in ["selected_ptime_s", "duration_cfg_s",
                                                   "report_step_cfg_s", "hyd_step_cfg_s"]}
    settings = {"inp_path": str(Path(inp_path).resolve()), "inp_sha256": _sha(inp_path),
                "H_metadata_sha256": _sha(H_path), "sensor_ids": ids, "seeds": seeds,
                "subset_indices": subsets.tolist(), "union_indices": union.tolist(),
                "q0_lps": q0_lps, "strength_factors": factors.tolist(),
                "retry_factor": float(retry_factor), **timing}
    if not np.isfinite(q0_lps) or q0_lps <= 0:
        raise ValueError("A finite positive nominal leak scale is required.")
    manifest_path = out.with_suffix(".json")
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError("Cached design settings differ; use a new design directory.")
    else:
        manifest_path.write_text(json.dumps({"settings": settings, "status": "preparing"}, indent=2), encoding="utf-8")
    if out.exists():
        if previous is None or previous.get("status") != "completed" or previous.get("asset_sha256") != _sha(out):
            raise ValueError("Cached design asset is incomplete or has changed.")
        return previous
    cache = out.with_suffix("")
    cache.mkdir(exist_ok=True)
    actual_calls = 0
    for attempt, scale in enumerate([1.0, float(retry_factor)]):
        attempt_dir = cache / f"attempt{attempt}"
        attempt_dir.mkdir(exist_ok=True)
        flows = (q0_lps * factors * scale).tolist()
        pending = [int(i) for i in union if not (attempt_dir / f"node{i}.npz").exists()]
        done = len(union) - len(pending)

        def save(index, result):
            nonlocal done
            pressures, failures = result
            temporary = attempt_dir / f"node{index}.tmp.npz"
            np.savez_compressed(temporary, pressures=pressures,
                                failures=json.dumps(failures), simulation_calls=levels)
            temporary.replace(attempt_dir / f"node{index}.npz")
            done += 1
            if done % 50 == 0 or done == len(union):
                print(f"[design] {Path(inp_path).stem}: {done}/{len(union)} leak locations, "
                      f"{levels} strengths, attempt={attempt}", flush=True)

        if pending:
            count = max(1, min(int(workers), len(pending))) if len(ids) >= 128 else 1
            if count == 1:
                _start_worker(settings, flows)
                for index in pending:
                    save(index, _simulate_node(ids[index]))
            else:
                with ProcessPoolExecutor(max_workers=count, initializer=_start_worker,
                                         initargs=(settings, flows)) as executor:
                    futures = {executor.submit(_simulate_node, ids[index]): index for index in pending}
                    for future in as_completed(futures):
                        save(futures[future], future.result())
        pressures = np.full((levels, len(union), len(ids)), np.nan)
        failures = []
        for column, index in enumerate(union):
            with np.load(attempt_dir / f"node{index}.npz", allow_pickle=False) as record:
                pressures[:, column] = record["pressures"]
                failures.extend(json.loads(str(record["failures"].item())))
                actual_calls += int(record["simulation_calls"])
        valid = np.isfinite(pressures).all(axis=2)
        failure_ratio = float((~valid.all(axis=0)).mean())
        if failure_ratio <= .30:
            break
        if attempt == 1:
            raise RuntimeError("Design leak-location failure ratio exceeds 30% after group recovery.")
        print("[design] >30% failed locations; retry the complete design at 0.1x flow as new scenarios.", flush=True)
    if not valid.all(axis=0).any():
        raise RuntimeError("No valid complete multi-strength leak location remains.")
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, pressures=pressures, valid=valid,
                        sensor_ids=np.asarray(ids), leak_ids=np.asarray(ids)[union],
                        union_indices=union, subset_indices=subsets, seeds=np.asarray(seeds),
                        flows_lps=np.asarray(flows), requested_factors=factors,
                        q0_lps=q0_lps, selected_ptime_s=timing["selected_ptime_s"])
    temporary.replace(out)
    result = {"settings": settings, "status": "completed", "asset_path": str(out.resolve()),
              "asset_sha256": _sha(out), "sampled_leak_locations_per_layout": size,
              "available_design_cases_per_layout": size * levels,
              "design_calls_per_layout_with_recovery": size * levels * (attempt+1),
              "unique_simulation_calls": actual_calls, "union_leak_locations": len(union),
              "flow_scale": scale, "failed_location_fraction": failure_ratio,
              "failed_simulation_records": failures}
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
