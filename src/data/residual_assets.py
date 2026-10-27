"""Load residual assets and align their sensor and leak axes by node ID."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _permutation(source, target):
    source, target = list(map(str, source)), list(map(str, target))
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("Node IDs must be unique before alignment.")
    if set(source) != set(target):
        raise ValueError(f"Candidate sets differ: missing={sorted(set(target)-set(source))[:5]}, "
                         f"extra={sorted(set(source)-set(target))[:5]}")
    position = {node: i for i, node in enumerate(source)}
    return np.asarray([position[node] for node in target], dtype=int)


@dataclass
class ResidualAssets:
    R: np.ndarray
    S: np.ndarray
    sensor_ids: list[str]
    leak_ids: list[str]
    valid_scenarios: np.ndarray

    def aligned(self, sensor_ids, leak_ids=None):
        leak_ids = list(sensor_ids) if leak_ids is None else list(leak_ids)
        rows = _permutation(self.sensor_ids, sensor_ids)
        cols = _permutation(self.leak_ids, leak_ids)
        return ResidualAssets(
            self.R[:, rows, :][:, :, cols], self.S[:, rows, :][:, :, cols],
            list(map(str, sensor_ids)), list(map(str, leak_ids)),
            self.valid_scenarios[:, cols],
        )


def load_residual_assets(path):
    with np.load(path, allow_pickle=False) as data:
        R, S = np.asarray(data["R"], float), np.asarray(data["S"], float)
        names = list(map(str, data["junction_names"]))
        leaks = list(map(str, data["leak_node_names"])) if "leak_node_names" in data else names
        valid = (np.asarray(data["valid_scenarios"], bool) if "valid_scenarios" in data
                 else np.ones((R.shape[0], R.shape[2]), dtype=bool))
    if R.ndim != 3 or R.shape != S.shape or R.shape[1:] != (len(names), len(leaks)):
        raise ValueError("R/S dimensions do not match their node labels.")
    if valid.shape != (R.shape[0], R.shape[2]):
        raise ValueError("Invalid scenario-mask dimensions.")
    if len(set(names)) != len(names) or len(set(leaks)) != len(leaks):
        raise ValueError("Duplicate node labels in R/S assets.")
    return ResidualAssets(R, S, names, leaks, valid)


def load_RS(
    R_path: str,
    S_path: str,
) -> Tuple[list, list]:
    """
    假设 R_path, S_path 是用 np.save(..., allow_pickle=True) 保存的 object 数组，
    这里统一转成 Python list[ndarray].
    """
    R_obj = np.load(R_path, allow_pickle=True)
    S_obj = np.load(S_path, allow_pickle=True)
    R_list = list(R_obj)
    S_list = list(S_obj)
    return R_list, S_list
