"""Information-theoretic placement on nominal or sampled hydraulic designs."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data.information_design import load_nominal_H_design, load_sampled_design
from src.data.loaders import (
    _compute_rel_all1_metrics, _guess_edge_features_csv_from_nodes_csv,
    build_junction_hop_dist_from_edges, load_graph_data,
)
from src.data.residual_assets import load_residual_assets
from src.evaluation.protocol import localization_error_rate_with_topo
from src.optimization.information_theory import rank_sensors


def information_context(*, nodes_csv, inp_path, rs_asset_path):
    """Load one common ID-aligned evaluation and all-sensor reference."""
    graph = load_graph_data(
        nodes_csv=str(nodes_csv),
        edge_feat_csv=_guess_edge_features_csv_from_nodes_csv(str(nodes_csv)),
        inp_path=str(inp_path), drop_non_junction=False,
    )
    ids = list(map(str, graph[8]))
    junction_idx = np.asarray(graph[7], dtype=int)
    assets = load_residual_assets(rs_asset_path).aligned(ids)
    distance = build_junction_hop_dist_from_edges(
        N_all=len(graph[3]), edges_idx=graph[5], junction_idx=junction_idx,
        include_reservoir_in_paths=True, unreachable_penalty=len(ids),
    )
    common = dict(R_list=assets.R, S_list=assets.S, dist_junc=distance,
                  demand_junc=graph[6]["base_demand"].to_numpy(float)[junction_idx],
                  valid_scenarios=assets.valid_scenarios)
    all1 = localization_error_rate_with_topo(x=np.ones(len(ids)), return_errors=False, **common)
    return dict(ids=ids, node_ids=list(map(str, graph[3])), junction_idx=junction_idx,
                common=common, all1=all1, net=Path(inp_path).stem)


def _save_placement(X, y, design, *, context, K_budget, bins, out_csv,
                    route, seed=None):
    out = Path(out_csv)
    layouts = out.with_name(out.stem + "_layouts")
    ranking_path = out.with_name(out.stem + "_ranking.json")
    if out.exists() or layouts.exists() or ranking_path.exists():
        raise FileExistsError(f"Use a fresh information-comparison output: {out}")
    ids = context["ids"]
    started = time.perf_counter()
    ranking = rank_sensors(X, y, K_budget, ids, bins=bins)
    selection_time = time.perf_counter() - started
    x = np.zeros(len(ids), dtype=int)
    x[ranking["selected_indices"]] = 1
    x_global = np.zeros(len(context["node_ids"]), dtype=int)
    x_global[context["junction_idx"]] = x
    metrics = localization_error_rate_with_topo(x=x, **context["common"])
    raw = {"_loc_success_raw": metrics["success_rate"],
           "_loc_mrr_raw": metrics["mrr"],
           "_loc_error_dist_raw": metrics["mean_topo_error"],
           "_loc_expected_topo_error_raw": metrics["mean_expected_topo_error"],
           "_water_demand_coverage_rate_raw": metrics["water_demand_coverage_rate"]}
    result = {"net": context["net"], "route": route,
              "method": "information_rri", "seed": seed,
              "repetition": "fixed_design" if seed is None else "sampled_design",
              "K_budget": int(K_budget), "num_sensors": int(x.sum()), "bins": int(bins),
              "runtime_opt_sec": selection_time,
              "valid_scenario_count": metrics["valid_scenario_count"],
              "unlocalized_count": metrics["unlocalized_count"],
              **design, **raw, **_compute_rel_all1_metrics(raw, context["all1"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    layouts.mkdir()
    label = "shared" if seed is None else f"sampled_seed{seed}"
    np.savez_compressed(
        layouts / f"information_{label}_K{int(K_budget)}.npz",
        x_best_junction=x, x_best_global=x_global,
        junction_node_ids=np.asarray(ids), global_node_ids=np.asarray(context["node_ids"]),
        design_leak_node_ids=y, valid_scenarios=context["common"]["valid_scenarios"],
        mrr_mat=metrics["mrr_mat"], topo_err_mat=metrics["topo_err_mat"],
        expected_topo_err_mat=metrics["expected_topo_err_mat"],
        quantization_low=ranking["quantization"]["low"],
        quantization_high=ranking["quantization"]["high"],
        relevance_bits=ranking["relevance_bits"],
    )
    ranking_path.write_text(json.dumps(
        {"reference": "https://doi.org/10.3390/s22020443", **design,
         "bins": int(bins), "trace": ranking["trace"],
         "selected_node_ids": ranking["selected_node_ids"]}, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[{route}] {result['net']}, seed={seed}: K={int(x.sum())}, "
          f"SR={metrics['success_rate']:.4f}; saved {out}", flush=True)
    return result


def run_shared_H_information_comparison(*, nodes_csv, inp_path, H_path, rs_asset_path,
                               K_budget, out_csv, bins=16, context=None):
    """Run once per fixed H design; no new hydraulic simulations."""
    if context is None:
        context = information_context(nodes_csv=nodes_csv, inp_path=inp_path, rs_asset_path=rs_asset_path)
    X, y, design = load_nominal_H_design(H_path, context["ids"])
    return _save_placement(X, y, design, context=context, K_budget=K_budget,
                           bins=bins, out_csv=out_csv, route="information_h")


def run_information_comparison(*, nodes_csv, inp_path, H_path, rs_asset_path,
                                       K_budget, out_csv, seeds, bins=16,
                                       leak_fraction=1/3, strength_range=(0.25, 2.0),
                                       strength_levels=3, workers=4, context=None):
    """Generate/cache the seed-union, then expose only each seed's design data."""
    from src.hydraulics.information_design import generate_information_design
    out = Path(out_csv)
    if out.exists():
        raise FileExistsError(f"Use a fresh comparison output: {out}")
    seeds = list(map(int, seeds))
    if context is None:
        context = information_context(nodes_csv=nodes_csv, inp_path=inp_path, rs_asset_path=rs_asset_path)
    design_path = out.parent / "design" / f"{context['net']}_sampled.npz"
    manifest = generate_information_design(
        inp_path=inp_path, H_path=H_path, sensor_ids=context["ids"], seeds=seeds,
        out_path=design_path, leak_fraction=leak_fraction, strength_range=strength_range,
        strength_levels=strength_levels, workers=workers,
    )
    records = []
    for seed in seeds:
        seed_csv = out.parent / (out.stem + "_seeds") / f"seed{seed}.csv"
        if seed_csv.exists():
            saved = pd.read_csv(seed_csv)
            row = saved.iloc[0]
            if (len(saved) != 1 or int(row.seed) != seed or int(row.bins) != bins
                    or int(row.K_budget) != K_budget
                    or row.design_asset_sha256 != manifest["asset_sha256"]):
                raise ValueError("Existing seed output differs from the requested design/settings.")
            records.append(row.to_dict())
            continue
        X, y, design = load_sampled_design(design_path, seed, context["ids"])
        records.append(_save_placement(
            X, y, design, context=context, K_budget=K_budget, bins=bins,
            out_csv=seed_csv, route="information", seed=seed,
        ))
    pd.DataFrame(records).to_csv(out, index=False, encoding="utf-8-sig")
    return records
