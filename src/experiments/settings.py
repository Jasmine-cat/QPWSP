"""Load separate recorded experiment profiles."""
from pathlib import Path
from types import SimpleNamespace
import copy
import yaml
ROOT = Path(__file__).resolve().parents[2]
def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result

def load_profile(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError("Cyclic profile inheritance")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    return merge(load_profile(path.parent / parent, (*_stack, path)), data) if parent else data

base = load_profile(ROOT / "configs/base.yaml")
ablation = load_profile(ROOT / "configs/ablation.yaml")
budget = load_profile(ROOT / "configs/budget_sensitivity.yaml")
gate_cfg = copy.deepcopy(base["optimizers"])
next(item["cfg"] for item in gate_cfg if item["method"] == "A0_gnn")["use_physics_gate"] = True
config = SimpleNamespace(cfg_list=base["optimizers"], gate_enabled_cfg=gate_cfg,
                         ablation_response=ablation["ablation_response"], ablation_mvc=ablation["ablation_mvc"],
                         budget_response=budget["budget_response"])
