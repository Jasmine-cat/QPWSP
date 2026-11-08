"""Run placement optimizers and evaluate localization."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.experiments.settings import config
from src.optimization.differential_evolution import solve_qubo_de_topk
from src.optimization.genetic import solve_qubo_ga_energy_soft_budget
from src.optimization.greedy_swap import solve_qubo_greedy_swap_refine
from src.optimization.local_search import solve_qubo_khot_swap_local_search
from src.data.loaders import (
    _compute_rel_all1_metrics,
    _get_method_cfg,
    _guess_edge_features_csv_from_nodes_csv,
    build_junction_hop_dist_from_edges,
    load_graph_data,
    load_H_for_nodes,
)
from src.evaluation.protocol import localization_error_rate_with_topo
from src.data.residual_assets import ResidualAssets, load_residual_assets
from src.qubo.topology import build_junction_edge_list_from_full_edges, undirected_unique_edges
from src.qubo.objective import build_Q_total, compute_qubo_component_stats, print_qubo_component_stats
from src.optimization.trainer import TrainConfig, project_to_binary, train_multi_start


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_assets(R, S, node_ids, valid, path, target_ids):
    if path is not None:
        return load_residual_assets(path).aligned(target_ids)
    if R is None and S is None:
        return None
    if R is None or S is None or node_ids is None:
        raise ValueError("Pass rs_node_ids with R/S arrays, or use rs_asset_path.")
    R, S = np.asarray(R), np.asarray(S)
    valid = np.ones((R.shape[0], R.shape[2]), bool) if valid is None else np.asarray(valid, bool)
    return ResidualAssets(R, S, list(map(str, node_ids)), list(map(str, node_ids)), valid).aligned(target_ids)


def _save_layout(result, out_csv):
    if not out_csv:
        return
    folder = Path(out_csv).with_suffix("").with_name(Path(out_csv).stem + "_layouts")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{result['method']}_seed{result['seed']}_K{result['K_budget']}.npz"
    path = folder / name
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, x_best_junction=result["x_best_junction"],
                        x_best_global=result["x_best_global"],
                        global_node_ids=np.asarray(result["global_node_ids"]),
                        junction_node_ids=np.asarray(result["junction_node_ids"]))




def _build_train_config(
    cfg: Dict[str, Any], *, net: str, method: str, seed: int, K_budget: int, route_tag: str
) -> TrainConfig:
    return TrainConfig(
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.0)),
        use_physics_injection=bool(cfg.get("use_physics_injection", True)),
        use_physics_gate=bool(cfg.get("use_physics_gate", True)),
        use_stagewise_ste=bool(cfg.get("use_stagewise_ste", True)),
        ste_stage1_ratio=float(cfg.get("ste_stage1_ratio", 0.10)),
        k_budget_hint=int(K_budget),
        ebin_patience=int(cfg.get("ebin_patience", 120)),
        ebin_min_delta=float(cfg.get("ebin_min_delta", 1e-8)),
        ebin_min_evals_stage2=int(cfg.get("ebin_min_evals_stage2", 80)),
        lr=float(cfg.get("lr", 1e-4)),
        max_epochs=int(cfg.get("max_epochs", 20000)),
        print_every=int(cfg.get("print_every", 200)),
        min_epochs=int(cfg.get("min_epochs", 600)),
        ema_window=int(cfg.get("ema_window", 50)),
        rel_tol=float(cfg.get("rel_tol", 1e-3)),
        plateau_patience=int(cfg.get("plateau_patience", 40)),
        lr_factor=float(cfg.get("lr_factor", 0.5)),
        min_lr=float(cfg.get("min_lr", 1e-5)),
        grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0)),
    )


def run_single_budget_experiment(
    *,
    nodes_csv: str,
    inp_path: str,
    K_budget: int,
    H_path: Optional[str] = None,
    alpha_bud: float = 1.0,
    alpha_leak_h: float = 0.0,
    alpha_mvc: float = 0.0,
    alpha_demand: float = 0.0,
    demand_clip_quantile: float = 0.95,
    lambda_pair: float = 2.25,
    method: str = "A0_gnn",
    x_mode: str = "full",
    cfg_list: Optional[List[Dict[str, Any]]] = None,
    seed: int = 0,
    route_tag: str = "response",
    R_list: Optional[List[np.ndarray]] = None,
    S_list: Optional[List[np.ndarray]] = None,
    rs_node_ids: Optional[List[str]] = None,
    valid_scenarios: Optional[np.ndarray] = None,
    rs_asset_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one network, sensor budget, seed, and optimizer combination."""
    inp_name = Path(inp_path).stem
    cfg = {} if method == "random" else _get_method_cfg(cfg_list, method)
    (
        X, A, A_norm, node_ids, _id2idx, edges_idx, df_nodes, junction_idx,
        junction_node_ids, graph_aux,
    ) = load_graph_data(
        nodes_csv=nodes_csv,
        edge_feat_csv=_guess_edge_features_csv_from_nodes_csv(nodes_csv),
        inp_path=inp_path,
        drop_non_junction=False,
        standardize=True,
        x_mode=x_mode,
        return_graph_struct=True,
    )
    junction_idx = np.asarray(junction_idx, dtype=int)
    N_all, N_junc = int(X.shape[0]), int(junction_idx.size)
    if not (0 < int(K_budget) <= N_junc):
        raise ValueError(f"K_budget={K_budget} is invalid for {N_junc} junctions")

    edge_list, _ignored, _full_to_junction = build_junction_edge_list_from_full_edges(
        edges_idx_full=undirected_unique_edges(edges_idx, remove_self=True),
        junction_idx_full=junction_idx,
        N_full=N_all,
    )
    H_junc = None
    if float(alpha_leak_h) != 0.0:
        resolved_H = H_path or str(REPO_ROOT / "outputs" / "assets" / f"{inp_name}_H.npz")
        H_junc = load_H_for_nodes(resolved_H, junction_node_ids)

    Q_dist, const = build_Q_total(
        N=N_junc,
        K_budget=int(K_budget),
        H=H_junc,
        edges_for_mvc=edge_list,
        alpha_bud=float(alpha_bud),
        alpha_leak_h=float(alpha_leak_h),
        alpha_mvc=float(alpha_mvc),
        alpha_demand=float(alpha_demand),
        demand_junc=df_nodes["base_demand"].to_numpy(dtype=float)[junction_idx],
        demand_clip_quantile=float(demand_clip_quantile),
        lambda_pair=float(lambda_pair),
    )

    histories = None
    solver_info: Dict[str, Any] = {}
    p_best_all = None
    p_j = None
    loss_best = float("nan")

    started = time.perf_counter()
    if method == "A0_gnn":
        train_cfg = _build_train_config(
            cfg, net=inp_name, method=method, seed=seed, K_budget=K_budget, route_tag=route_tag
        )
        p_best_all, loss_best, histories = train_multi_start(
            Q_dist_np=Q_dist,
            const_dist=const,
            X_np=X,
            A_norm_np=A_norm,
            junction_idx_np=junction_idx,
            config=train_cfg,
            graph_aux_np=graph_aux,
            num_restarts=int(cfg.get("num_restarts", 1)),
            seed=int(seed),
        )
        p_j = np.asarray(p_best_all, dtype=float)[junction_idx]
        x_j = project_to_binary(p_j, int(K_budget))
        solver_info["num_restarts"] = int(cfg.get("num_restarts", 1))
    elif method == "A1_qubo":
        max_iters = int(cfg.get("max_iters", 10**4))
        x_j, _best_E, histories, solver_info = solve_qubo_khot_swap_local_search(
            Q_dist_np=Q_dist,
            const_dist=const,
            K_budget=int(K_budget),
            num_restarts=int(cfg.get("num_restarts", 80)),
            max_iters=max_iters,
            max_iters_per_restart=int(cfg.get("max_iters_per_restart", max_iters)),
            min_restarts=int(cfg.get("min_restarts", 5)),
            max_restarts=int(cfg.get("max_restarts", cfg.get("num_restarts", 80))),
            restart_patience=int(cfg.get("restart_patience", 12)),
            restart_improve_tol=float(cfg.get("restart_improve_tol", 1e-10)),
            zero_sample=cfg.get("zero_sample", None),
            time_budget_sec=cfg.get("time_budget_sec", None),
            keep_histories=bool(cfg.get("keep_histories", True)),
            seed=int(seed),
        )
        p_j = np.asarray(x_j, dtype=float)
    elif method == "A2_ga":
        x_j, _best_E, histories, solver_info = solve_qubo_ga_energy_soft_budget(
            Q_dist_np=Q_dist,
            const_dist=const,
            K_budget=int(K_budget),
            pop_size=int(cfg.get("pop_size", 80)),
            max_iters=int(cfg.get("max_iters", 2500)),
            mut_rate=float(cfg.get("mut_rate", 0.02)),
            time_budget_sec=cfg.get("time_budget_sec", None),
            early_stop_patience=int(cfg.get("early_stop_patience", 200)),
            early_stop_min_iters=int(cfg.get("early_stop_min_iters", 200)),
            early_stop_abs_tol=float(cfg.get("early_stop_abs_tol", 1e-8)),
            keep_histories=bool(cfg.get("keep_histories", True)),
            seed=int(seed),
        )
        p_j = np.asarray(x_j, dtype=float)
    elif method == "A4_greedy_swap":
        x_j, _best_E, histories, solver_info = solve_qubo_greedy_swap_refine(
            Q_dist_np=Q_dist,
            const_dist=const,
            K_budget=int(K_budget),
            swap_max_rounds=int(cfg.get("swap_max_rounds", 30)),
            swap_zero_sample=cfg.get("swap_zero_sample", None),
            improve_tol=float(cfg.get("improve_tol", 1e-12)),
            random_tie_break=bool(cfg.get("random_tie_break", True)),
            time_budget_sec=cfg.get("time_budget_sec", None),
            keep_histories=bool(cfg.get("keep_histories", True)),
            seed=int(seed),
        )
        p_j = np.asarray(x_j, dtype=float)
    elif method == "A5_de":
        x_j, _best_E, histories, solver_info = solve_qubo_de_topk(
            Q_dist_np=Q_dist,
            const_dist=const,
            K_budget=int(K_budget),
            pop_size=int(cfg.get("pop_size", 80)),
            max_iters=int(cfg.get("max_iters", 1000)),
            F=float(cfg.get("F", 0.5)),
            CR=float(cfg.get("CR", 0.9)),
            time_budget_sec=cfg.get("time_budget_sec", None),
            early_stop_patience=int(cfg.get("early_stop_patience", 200)),
            early_stop_min_iters=int(cfg.get("early_stop_min_iters", 200)),
            early_stop_abs_tol=float(cfg.get("early_stop_abs_tol", 1e-8)),
            keep_histories=bool(cfg.get("keep_histories", True)),
            seed=int(seed),
        )
        p_j = np.asarray(x_j, dtype=float)
    elif method == "random":
        rng = np.random.default_rng(int(seed))
        x_j = np.zeros(N_junc, dtype=int)
        x_j[rng.choice(N_junc, size=int(K_budget), replace=False)] = 1
        p_j = x_j.astype(float)
        solver_info = {"restarts_done": 1, "iters_total": 1, "stop_reason": "random_khot"}
    else:
        raise ValueError(f"unsupported method: {method}")
    runtime_opt_sec = float(time.perf_counter() - started)

    x_j = np.asarray(x_j, dtype=int)
    x_global = np.zeros(N_all, dtype=int)
    x_global[junction_idx] = x_j
    soft_stats = compute_qubo_component_stats(p_j, Q_dist, const)
    binary_stats = compute_qubo_component_stats(x_j.astype(float), Q_dist, const)
    print_qubo_component_stats("[Energy-Full]", soft_stats)

    raw_metrics: Dict[str, float] = {}
    assets = _resolve_assets(R_list, S_list, rs_node_ids, valid_scenarios,
                             rs_asset_path, junction_node_ids)
    if assets is not None:
        dist_junc = build_junction_hop_dist_from_edges(
            N_all=N_all,
            edges_idx=edges_idx,
            junction_idx=junction_idx,
            include_reservoir_in_paths=True,
            unreachable_penalty=N_junc,
        )
        loc = localization_error_rate_with_topo(
            x=x_j.astype(float),
            R_list=assets.R,
            S_list=assets.S,
            valid_scenarios=assets.valid_scenarios,
            dist_junc=dist_junc,
            demand_junc=df_nodes["base_demand"].to_numpy(dtype=float)[junction_idx],
        )
        raw_metrics = {
            "_loc_success_raw": float(loc["success_rate"]),
            "_loc_error_dist_raw": float(loc["mean_topo_error"]),
            "_loc_mrr_raw": float(loc["mrr"]),
            "_loc_expected_topo_error_raw": float(loc["mean_expected_topo_error"]),
            "_water_demand_coverage_rate_raw": float(loc["water_demand_coverage_rate"]),
        }
        print(
            f"[budget={K_budget}] loc_success={raw_metrics['_loc_success_raw']:.4f}, "
            f"loc_error_dist={raw_metrics['_loc_error_dist_raw']:.4f}, #sensors={int(x_j.sum())}"
        )

    return {
        "net": inp_name,
        "route": route_tag,
        "method": method,
        "seed": int(seed),
        "K_budget": int(K_budget),
        "alpha_bud": float(alpha_bud),
        "alpha_leak_h": float(alpha_leak_h),
        "alpha_mvc": float(alpha_mvc),
        "alpha_demand": float(alpha_demand),
        "demand_clip_quantile": float(demand_clip_quantile),
        "lambda_pair": float(lambda_pair),
        "x_mode": str(x_mode),
        "loss": float(loss_best),
        "energy_soft_full": float(soft_stats["energy_total_full"]),
        "energy_binary_full": float(binary_stats["energy_total_full"]),
        "energy_cov_full": float(soft_stats["energy_cov_full"]),
        "energy_bud_full": float(soft_stats["energy_bud_full"]),
        "energy_leak_full": float(soft_stats["energy_leak_full"]),
        "energy_demand_full": float(soft_stats["energy_demand_full"]),
        "ratio_cov_full": float(soft_stats["ratio_cov_full"]),
        "ratio_bud_full": float(soft_stats["ratio_bud_full"]),
        "ratio_leak_full": float(soft_stats["ratio_leak_full"]),
        "ratio_demand_full": float(soft_stats["ratio_demand_full"]),
        "num_sensors": int(x_j.sum()),
        "p_best_all": p_best_all,
        "p_best_junction": p_j,
        "x_best_global": x_global,
        "x_best_junction": x_j,
        "junction_node_ids": list(map(str, junction_node_ids)),
        "global_node_ids": list(map(str, node_ids)),
        "histories": histories,
        "runtime_opt_sec": runtime_opt_sec,
        "solver_restarts_done": solver_info.get("restarts_done"),
        "solver_iters_total": solver_info.get("iters_total"),
        "solver_stop_reason": solver_info.get("stop_reason"),
        "solver_gens_done": solver_info.get("gens_done"),
        **raw_metrics,
    }


def run_multiple_budgets(
    *,
    budgets: List[int],
    nodes_csv: str,
    inp_path: str,
    H_path: Optional[str] = None,
    alpha_bud: float = 1.0,
    alpha_leak_h: float = 0.0,
    alpha_mvc: float = 0.0,
    alpha_demand: float = 0.0,
    demand_clip_quantile: float = 0.95,
    lambda_pair: float = 2.25,
    x_mode: str = "full",
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    route_tag: str = "response",
    R_list: Optional[List[np.ndarray]] = None,
    S_list: Optional[List[np.ndarray]] = None,
    rs_node_ids: Optional[List[str]] = None,
    valid_scenarios: Optional[np.ndarray] = None,
    rs_asset_path: Optional[str] = None,
    out_csv: Optional[str] = None,
    cfg_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run optimizer, objective, ablation, or budget comparisons."""
    if out_csv and Path(out_csv).exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {out_csv}")
    methods = list(methods or ["A0_gnn", "A1_qubo", "A2_ga", "A4_greedy_swap", "A5_de"])
    seeds = list(seeds or [0])
    inp_name = Path(inp_path).stem

    graph = load_graph_data(
        nodes_csv=nodes_csv, edge_feat_csv=_guess_edge_features_csv_from_nodes_csv(nodes_csv),
        inp_path=inp_path, drop_non_junction=False,
    )
    assets = _resolve_assets(R_list, S_list, rs_node_ids, valid_scenarios,
                             rs_asset_path, graph[8])
    if assets is None:
        raise ValueError("Batch localization experiments require labeled R/S assets.")
    dist = build_junction_hop_dist_from_edges(
        N_all=graph[0].shape[0], edges_idx=graph[5], junction_idx=graph[7],
        include_reservoir_in_paths=True, unreachable_penalty=len(graph[8]),
    )
    all1_ref = localization_error_rate_with_topo(
        np.ones(len(graph[8])), assets.R, assets.S, dist_junc=dist,
        demand_junc=graph[6]["base_demand"].to_numpy(float)[graph[7]],
        valid_scenarios=assets.valid_scenarios, return_errors=False,
    )
    R_list, S_list = assets.R, assets.S
    rs_node_ids, valid_scenarios = assets.sensor_ids, assets.valid_scenarios
    rs_asset_path = None

    results: List[Dict[str, Any]] = []
    normal_methods = [method for method in methods if method != "random"]
    for K_budget in budgets:
        for seed_index, seed in enumerate(seeds):
            for method in normal_methods:
                print("\n" + "=" * 70)
                print(f"Running method={method} | budget K={K_budget} | seed={seed}")
                result = run_single_budget_experiment(
                    nodes_csv=nodes_csv,
                    inp_path=inp_path,
                    K_budget=int(K_budget),
                    H_path=H_path,
                    alpha_bud=alpha_bud,
                    alpha_leak_h=alpha_leak_h,
                    alpha_mvc=alpha_mvc,
                    alpha_demand=alpha_demand,
                    demand_clip_quantile=demand_clip_quantile,
                    lambda_pair=lambda_pair,
                    method=method,
                    x_mode=x_mode,
                    cfg_list=cfg_list,
                    seed=int(seed),
                    route_tag=route_tag,
                    R_list=R_list,
                    S_list=S_list,
                    rs_node_ids=rs_node_ids, valid_scenarios=valid_scenarios,
                )
                result.update(_compute_rel_all1_metrics(result, all1_ref, eps=1e-8))
                results.append(result)
                _save_layout(result, out_csv)

            print("\n" + "=" * 70)
            print(f"Running method=random | budget K={K_budget} | seed={seed}")
            random_result = run_single_budget_experiment(
                nodes_csv=nodes_csv,
                inp_path=inp_path,
                K_budget=int(K_budget),
                H_path=H_path,
                alpha_bud=alpha_bud,
                alpha_leak_h=alpha_leak_h,
                alpha_mvc=alpha_mvc,
                alpha_demand=alpha_demand,
                demand_clip_quantile=demand_clip_quantile,
                lambda_pair=lambda_pair,
                method="random",
                x_mode=x_mode,
                cfg_list=cfg_list,
                seed=int(seed),
                route_tag=route_tag,
                R_list=R_list,
                S_list=S_list,
                rs_node_ids=rs_node_ids, valid_scenarios=valid_scenarios,
            )
            random_result.update(_compute_rel_all1_metrics(random_result, all1_ref, eps=1e-8))
            results.append(random_result)
            _save_layout(random_result, out_csv)

    records = []
    for result in results:
        records.append({
            key: result.get(key, np.nan)
            for key in (
                "net", "route", "method", "seed", "K_budget", "alpha_bud", "alpha_leak_h",
                "alpha_mvc", "lambda_pair", "x_mode", "loss",
                "alpha_demand", "demand_clip_quantile", "energy_demand_full", "ratio_demand_full",
                "energy_soft_full", "energy_binary_full", "energy_cov_full", "energy_bud_full",
                "energy_leak_full", "ratio_cov_full", "ratio_bud_full", "ratio_leak_full", "num_sensors",
                "loc_success_rel_all1", "loc_error_dist_rel_all1", "loc_mrr_rel_all1",
                "loc_expected_topo_error_rel_all1", "water_demand_coverage_rate_rel_all1", "runtime_opt_sec",
                "solver_restarts_done", "solver_iters_total", "solver_stop_reason", "solver_gens_done",
                "_loc_success_raw", "_loc_mrr_raw", "_loc_error_dist_raw", "_loc_expected_topo_error_raw",
            )
        })
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] Results saved to {out_csv}")
    return results
