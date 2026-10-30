"""Entry points for preprocessing, assets and recorded experiment profiles."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.experiments.settings import ROOT, load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "assets", "compare", "objective", "ablation", "budget"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--nets", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--objectives", nargs="+", choices=["response", "mvc", "information"],
                        help="Objective stage only; information reuses the existing H without simulation.")
    parser.add_argument("--inp-dir", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physics-gate", choices=["on", "off"])
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()
    stage = args.stage
    if args.objectives is not None and stage != "objective":
        parser.error("--objectives is only supported by the objective stage.")
    profiles = {"prepare": "base.yaml", "assets": "base.yaml",
                "compare": "optimizer_comparison.yaml", "objective": "objective_comparison.yaml",
                "ablation": "ablation.yaml", "budget": "budget_sensitivity.yaml"}
    profile = load_profile(args.config or ROOT / "configs" / profiles[stage])
    if args.show_config:
        import yaml
        print(yaml.safe_dump(profile, sort_keys=False))
        return
    experiment, paths = profile["experiment"], profile["paths"]
    nets = args.nets or experiment["networks"]
    seeds = args.seeds if args.seeds is not None else experiment.get("seed_list")
    inp_dir = args.inp_dir or ROOT / paths["networks"]
    features_dir = args.features_dir or ROOT / paths["features"]
    assets_dir = args.assets_dir or ROOT / paths["assets"]
    output_dir = args.output_dir or ROOT / paths["results"]
    for folder in [assets_dir, output_dir]:
        target = folder.resolve()
        if target == ROOT or any(target.is_relative_to(ROOT / name) for name in ["data", "src", "configs"]):
            parser.error("Generated assets/results must be outside source and input-data directories.")
    if stage == "prepare":
        from src.data.preprocessing import get_base_features_fromInp
        for net in nets:
            if (features_dir / f"{net}_node_features.csv").exists() or (features_dir / f"{net}q0.json").exists():
                raise FileExistsError(f"Use a new features directory to recalibrate {net}.")
            get_base_features_fromInp(str(inp_dir / f"{net}.inp"), str(features_dir))
        return
    if stage == "assets":
        from src.hydraulics.h_response import generate_H_from_inp
        from src.hydraulics.residual_generator import generate_RS_from_inp
        hydraulic = profile["hydraulics"]
        for net in nets:
            common = dict(inp_path=str(inp_dir / f"{net}.inp"), node_features_dir=str(features_dir))
            generate_H_from_inp(out_npz=str(assets_dir / f"{net}_H.npz"),
                                h_ref_factor=hydraulic["h_ref_factor"],
                                h_recovery_factors=tuple(hydraulic["h_recovery_factors"]), **common)
            generate_RS_from_inp(out_npz=str(assets_dir / f"{net}_RS.npz"),
                                 n_leak_levels=hydraulic["rs_n_levels"],
                                 factor_min=hydraulic["rs_factor_min"], factor_max=hydraulic["rs_factor_max"],
                                 rs_high_failure_retry_factor=hydraulic["rs_group_retry_factor"], **common)
        return
    if stage in ["ablation", "budget"]:
        from src.experiments import ablation, budget
        module = ablation if stage == "ablation" else budget
        # The existing modules preserve their distinct variant/seed schedules.
        if inp_dir.resolve() != (ROOT / "data/networks").resolve() or features_dir.resolve() != (ROOT / "data/features").resolve():
            parser.error("Ablation/budget currently use data/networks and data/features; assets/results are configurable.")
        if stage == "ablation":
            module.RESPONSE_ANCHORS = profile["ablation_response"]
            module.MVC_ANCHORS = profile["ablation_mvc"]
        else:
            module.RESPONSE_ANCHORS = profile["budget_response"]
        module.main(output_dir=output_dir / stage, assets_dir=assets_dir, networks=nets, seeds=seeds,
                    objective_options=profile.get("optional_objective", {}))
        return
    cfg = copy.deepcopy(profile["optimizers"])
    if args.physics_gate is not None:
        next(item["cfg"] for item in cfg if item["method"] == "A0_gnn")["use_physics_gate"] = args.physics_gate == "on"
    for net in nets:
        objectives = (args.objectives or ["response", "mvc"]) if stage == "objective" else ["response"]
        if stage == "objective" and len(set(objectives)) > 1:
            if profile["response"][net]["K"] != profile["mvc"][net]["K"]:
                raise ValueError(f"Objective-comparison sensor budgets differ for {net}.")
        for objective in dict.fromkeys(objectives):
            if objective == "information":
                from src.experiments.objective import run_information_comparison
                options = profile.get("information", {})
                run_information_comparison(
                    nodes_csv=features_dir / f"{net}_node_features.csv",
                    inp_path=inp_dir / f"{net}.inp", H_path=assets_dir / f"{net}_H.npz",
                    rs_asset_path=assets_dir / f"{net}_RS.npz",
                    K_budget=profile["response"][net]["K"],
                    out_csv=output_dir / stage / f"{net}_information.csv",
                    bins=options.get("bins", 16),
                )
                continue
            from src.optimization.runner import run_multiple_budgets
            settings = copy.deepcopy(profile[objective][net])
            for key, value in profile.get("optional_objective", {}).items():
                settings.setdefault(key, value)
            k = settings.pop("K")
            out = output_dir / stage / f"{net}_{objective}.csv"
            run_multiple_budgets(
                budgets=[k], nodes_csv=str(features_dir / f"{net}_node_features.csv"),
                inp_path=str(inp_dir / f"{net}.inp"), H_path=str(assets_dir / f"{net}_H.npz"),
                rs_asset_path=str(assets_dir / f"{net}_RS.npz"), cfg_list=cfg, seeds=seeds,
                methods=["A0_gnn"] if stage == "objective" else (args.methods or experiment.get("methods")),
                route_tag=objective, out_csv=str(out), **settings,
            )
