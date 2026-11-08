"""Run sensor-budget sensitivity experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.experiments.settings import config
from src.optimization.runner import run_multiple_budgets


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "experiments" / "budget"
RESPONSE_ANCHORS = config.budget_response
TOP_SEEDS = {
    "CY_DMA": [66, 99, 22, 77, 0, 55],
    "Richmond_skeleton": [11, 33, 44, 55, 66, 88],
    "Hanoi": [11, 0, 77, 22, 44, 66],
    "EX": [11, 99, 33, 22, 55, 77],
}



def main(output_dir=None, assets_dir=None, networks=None, seeds=None, objective_options=None) -> None:
    global OUT_DIR
    OUT_DIR = Path(output_dir) if output_dir is not None else ROOT / "outputs" / "experiments" / "budget"
    assets_dir = Path(assets_dir) if assets_dir is not None else ROOT / "outputs" / "assets"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_paths: List[Path] = []
    for net in (networks or RESPONSE_ANCHORS):
        anchor = dict(RESPONSE_ANCHORS[net])
        anchor.update(objective_options or {})
        from src.data.residual_assets import load_residual_assets
        n_junc = len(load_residual_assets(assets_dir / f"{net}_RS.npz").sensor_ids)
        budgets = sorted({max(1, min(n_junc, int(round(n_junc * ratio / 10.0)))) for ratio in range(1, 11)})
        out_csv = OUT_DIR / f"exp_metrics_{net}_V1_sensitivity_detail.csv"
        run_multiple_budgets(
            budgets=budgets,
            nodes_csv=str(ROOT / "data" / "features" / f"{net}_node_features.csv"),
            inp_path=str(ROOT / "data" / "networks" / f"{net}.inp"),
            H_path=str(assets_dir / f"{net}_H.npz"),
            methods=["A0_gnn"],
            seeds=seeds or TOP_SEEDS[net],
            route_tag="V1_sensitivity",
            rs_asset_path=str(assets_dir / f"{net}_RS.npz"),
            out_csv=str(out_csv),
            cfg_list=config.gate_enabled_cfg,
            **anchor,
        )
        df = pd.read_csv(out_csv)
        df["K_ratio"] = df["K_budget"] / float(n_junc)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        detail_paths.append(out_csv)

    detail = pd.concat([pd.read_csv(path) for path in detail_paths], ignore_index=True)
    detail.to_csv(OUT_DIR / "exp_metrics_V1_sensitivity_4nets_top6seed_detail.csv", index=False, encoding="utf-8-sig")
    metric_cols = [
        "loc_success_rel_all1", "loc_mrr_rel_all1", "loc_error_dist_rel_all1",
        "loc_expected_topo_error_rel_all1", "runtime_opt_sec",
    ]
    curve = detail.groupby(["net", "K_budget", "K_ratio"], dropna=False)[metric_cols].agg(["mean", "std"])
    curve.columns = ["_".join(col) for col in curve.columns]
    curve.reset_index().to_csv(
        OUT_DIR / "exp_metrics_V1_sensitivity_4nets_top6seed_curve.csv", index=False, encoding="utf-8-sig"
    )
