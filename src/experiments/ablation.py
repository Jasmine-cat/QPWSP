"""Run component ablation experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.experiments.settings import config
from src.optimization.runner import run_multiple_budgets


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "experiments" / "ablation"
SEEDS = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
NETWORKS = ("CY_DMA", "Richmond_skeleton", "Hanoi", "EX")


RESPONSE_ANCHORS = config.ablation_response
MVC_ANCHORS = config.ablation_mvc

VARIANTS: List[Dict[str, Any]] = [
    {"variant": "V1", "objective": "H-response", "x_mode": "full", "edge_physics": True, "stagewise": True},
    {"variant": "V2", "objective": "MVC-type", "x_mode": "full", "edge_physics": True, "stagewise": True},
    {"variant": "V3", "objective": "H-response", "x_mode": "ones", "edge_physics": True, "stagewise": True},
    {"variant": "V4", "objective": "H-response", "x_mode": "degree", "edge_physics": True, "stagewise": True},
    {"variant": "V5", "objective": "H-response", "x_mode": "ones+degree", "edge_physics": True, "stagewise": True},
    {"variant": "V6", "objective": "H-response", "x_mode": "full", "edge_physics": False, "stagewise": True},
    {"variant": "V7", "objective": "H-response", "x_mode": "full", "edge_physics": True, "stagewise": False},
    {"variant": "V8", "objective": "MVC-type", "x_mode": "ones+degree", "edge_physics": False, "stagewise": True},
    {"variant": "V9", "objective": "H-response", "x_mode": "ones+degree", "edge_physics": False, "stagewise": True},
]


def _objective_params(net: str, objective: str) -> Dict[str, float]:
    if objective == "H-response":
        anchor = RESPONSE_ANCHORS[net]
        return {
            "alpha_bud": anchor["alpha_bud"],
            "alpha_leak_h": anchor["alpha_leak_h"],
            "alpha_mvc": 0.0,
            "lambda_pair": anchor["lambda_pair"],
        }
    anchor = MVC_ANCHORS[net]
    return {
        "alpha_bud": anchor["alpha_bud"],
        "alpha_leak_h": 0.0,
        "alpha_mvc": anchor["alpha_mvc"],
        "lambda_pair": 0.0,
    }


def _variant_cfg(variant: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg_list = copy.deepcopy(config.cfg_list)
    cfg = next(item["cfg"] for item in cfg_list if item["method"] == "A0_gnn")
    edge_on = bool(variant["edge_physics"])
    cfg["use_physics_injection"] = edge_on
    cfg["use_physics_gate"] = edge_on
    cfg["use_stagewise_ste"] = bool(variant["stagewise"])
    return cfg_list



def main(output_dir=None, assets_dir=None, networks=None, seeds=None, objective_options=None) -> None:
    global OUT_DIR
    OUT_DIR = Path(output_dir) if output_dir is not None else ROOT / "outputs" / "experiments" / "ablation"
    assets_dir = Path(assets_dir) if assets_dir is not None else ROOT / "outputs" / "assets"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files: List[Path] = []
    for net in (networks or NETWORKS):
        nodes_csv = ROOT / "data" / "features" / f"{net}_node_features.csv"
        inp_path = ROOT / "data" / "networks" / f"{net}.inp"
        H_path = assets_dir / f"{net}_H.npz"

        for variant in VARIANTS:
            params = _objective_params(net, str(variant["objective"]))
            params.update(objective_options or {})
            out_csv = OUT_DIR / f"exp_metrics_{net}_{variant['variant']}.csv"
            run_multiple_budgets(
                budgets=[int(RESPONSE_ANCHORS[net]["K"])],
                nodes_csv=str(nodes_csv),
                inp_path=str(inp_path),
                H_path=str(H_path),
                methods=["A0_gnn"],
                seeds=seeds or SEEDS,
                x_mode=str(variant["x_mode"]),
                route_tag=str(variant["variant"]),
                rs_asset_path=str(assets_dir / f"{net}_RS.npz"),
                out_csv=str(out_csv),
                cfg_list=_variant_cfg(variant),
                **params,
            )
            df = pd.read_csv(out_csv)
            df["variant"] = variant["variant"]
            df["objective"] = variant["objective"]
            df["node_input"] = variant["x_mode"]
            df["edge_physics"] = "On" if variant["edge_physics"] else "Off"
            df["training_selection"] = "Top-k STE" if variant["stagewise"] else "Soft only"
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            output_files.append(out_csv)

    detail = pd.concat([pd.read_csv(path) for path in output_files], ignore_index=True)
    detail.to_csv(OUT_DIR / "exp_metrics_abolition_detail.csv", index=False, encoding="utf-8-sig")
    metric_cols = [
        "loc_success_rel_all1", "loc_error_dist_rel_all1", "loc_mrr_rel_all1",
        "loc_expected_topo_error_rel_all1", "energy_binary_full", "loss", "runtime_opt_sec",
    ]
    summary = detail.groupby(["net", "variant", "method"], dropna=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(OUT_DIR / "exp_metrics_abolition_summary.csv", index=False, encoding="utf-8-sig")
