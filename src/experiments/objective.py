"""Information-theoretic placement using the existing nominal H asset."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data.information_design import load_nominal_H_design
from src.data.loaders import (
    _compute_rel_all1_metrics, _guess_edge_features_csv_from_nodes_csv,
    build_junction_hop_dist_from_edges, load_graph_data,
)
from src.data.residual_assets import load_residual_assets
from src.evaluation.protocol import localization_error_rate_with_topo
from src.optimization.information_theory import rank_sensors


def run_information_comparison(*, nodes_csv, inp_path, H_path, rs_asset_path,
                               K_budget, out_csv, bins=16):
    """Run once per fixed design asset; keep raw and all1-relative metrics."""
    out = Path(out_csv)
    layouts = out.with_name(out.stem + "_layouts")
    ranking_path = out.with_name(out.stem + "_ranking.json")
    if out.exists() or layouts.exists() or ranking_path.exists():
        raise FileExistsError(f"Use a fresh information-comparison output: {out}")
    graph = load_graph_data(
        nodes_csv=str(nodes_csv),
        edge_feat_csv=_guess_edge_features_csv_from_nodes_csv(str(nodes_csv)),
        inp_path=str(inp_path), drop_non_junction=False,
    )
    node_ids = list(map(str, graph[3]))
    junction_idx = np.asarray(graph[7], dtype=int)
    junction_ids = list(map(str, graph[8]))
    X, y, design = load_nominal_H_design(H_path, junction_ids)
    started = time.perf_counter()
    ranking = rank_sensors(X, y, K_budget, junction_ids, bins=bins)
    selection_time = time.perf_counter() - started
    x = np.zeros(len(junction_ids), dtype=int)
    x[ranking["selected_indices"]] = 1
    x_global = np.zeros(len(node_ids), dtype=int)
    x_global[junction_idx] = x
    assets = load_residual_assets(rs_asset_path).aligned(junction_ids)
    distance = build_junction_hop_dist_from_edges(
        N_all=len(node_ids), edges_idx=graph[5], junction_idx=junction_idx,
        include_reservoir_in_paths=True, unreachable_penalty=len(junction_ids),
    )
    common = dict(R_list=assets.R, S_list=assets.S, dist_junc=distance,
                  demand_junc=graph[6]["base_demand"].to_numpy(float)[junction_idx],
                  valid_scenarios=assets.valid_scenarios)
    metrics = localization_error_rate_with_topo(x=x, **common)
    all1 = localization_error_rate_with_topo(x=np.ones(len(x)), return_errors=False, **common)
    raw = {"_loc_success_raw": metrics["success_rate"],
           "_loc_mrr_raw": metrics["mrr"],
           "_loc_error_dist_raw": metrics["mean_topo_error"],
           "_loc_expected_topo_error_raw": metrics["mean_expected_topo_error"],
           "_water_demand_coverage_rate_raw": metrics["water_demand_coverage_rate"]}
    result = {"net": Path(inp_path).stem, "route": "information",
              "method": "information_rri", "seed": None,
              "repetition": "fixed_design", "K_budget": int(K_budget),
              "num_sensors": int(x.sum()), "bins": int(bins),
              "runtime_opt_sec": selection_time,
              "valid_scenario_count": metrics["valid_scenario_count"],
              "unlocalized_count": metrics["unlocalized_count"],
              **design, **raw, **_compute_rel_all1_metrics(raw, all1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    layouts.mkdir()
    np.savez_compressed(
        layouts / f"information_shared_K{int(K_budget)}.npz",
        x_best_junction=x, x_best_global=x_global,
        junction_node_ids=np.asarray(junction_ids), global_node_ids=np.asarray(node_ids),
        design_leak_node_ids=y, valid_scenarios=assets.valid_scenarios,
        mrr_mat=metrics["mrr_mat"], topo_err_mat=metrics["topo_err_mat"],
        expected_topo_err_mat=metrics["expected_topo_err_mat"],
    )
    ranking_path.write_text(json.dumps(
        {"reference": "https://doi.org/10.3390/s22020443", **design,
         "bins": int(bins), "trace": ranking["trace"],
         "selected_node_ids": ranking["selected_node_ids"]}, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[information] {result['net']}: K={int(x.sum())}, "
          f"SR={metrics['success_rate']:.4f}, no additional design simulations; saved {out}")
    return result
