"""Reuse the H-response design asset for a matched-information comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.loaders import load_H_for_nodes


def load_nominal_H_design(path, node_ids):
    """Return one normalized nominal response per available leak location.

    This is a finite nominal-scenario adaptation, not the multi-magnitude design
    dataset used by Santos-Ruiz et al. No hydraulic simulation or augmentation is
    performed. All sensor rows remain when H has excluded failed leak columns.
    """
    path = Path(path)
    ids = list(map(str, node_ids))
    if path.suffix.lower() != ".npz":
        raise ValueError("The information comparison requires a labeled H NPZ asset.")
    with np.load(path, allow_pickle=False) as data:
        source_leaks = list(map(str, data.get("leak_node_names", data["junction_names"])))
    if not set(source_leaks).issubset(ids):
        raise ValueError("H contains leak classes outside the sensor candidate set.")
    H = load_H_for_nodes(str(path), ids)
    if not np.isfinite(H).all():
        raise ValueError("H contains non-finite responses; invalid columns must be excluded explicitly.")
    leak_set = set(source_leaks)
    leaks = [node for node in ids if node in leak_set]
    if len(leaks) < 2 or H.shape != (len(ids), len(leaks)):
        raise ValueError("At least two aligned nominal leak classes are required.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return H.T, np.asarray(leaks), {
        "design_representation": "normalized_H_nominal",
        "design_asset_sha256": digest.hexdigest(),
        "design_sample_count": len(leaks), "design_samples_per_class": 1,
        "design_leak_class_count": len(leaks),
        "additional_design_hydraulic_simulations": 0,
    }


def load_sampled_design(path, seed, node_ids):
    """Read only this seed's leak samples; retain every candidate sensor column."""
    ids = list(map(str, node_ids))
    with np.load(path, allow_pickle=False) as data:
        source_ids = data["sensor_ids"].astype(str).tolist()
        if source_ids != ids:
            raise ValueError("Sampled design sensor IDs must match the graph order.")
        selected_seed = np.flatnonzero(data["seeds"] == int(seed))
        if len(selected_seed) != 1:
            raise ValueError("Requested seed missing or duplicated in the design asset.")
        subset = data["subset_indices"][selected_seed[0]]
        position = {int(index): j for j, index in enumerate(data["union_indices"])}
        columns = np.asarray([position[int(index)] for index in subset])
        complete = data["valid"][:, columns].all(axis=0)
        if float((~complete).mean()) > .30:
            raise ValueError("This seed has more than 30% incomplete sampled leak locations.")
        columns, subset = columns[complete], subset[complete]
        if len(columns) < 2:
            raise ValueError("Fewer than two valid sampled leak classes remain.")
        pressures = data["pressures"][:, columns, :]
        X = pressures.reshape(-1, len(ids))
        y = np.tile(np.asarray(ids)[subset], pressures.shape[0])
        flows = data["flows_lps"].tolist()
    manifest = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return X, y, {"design_representation": "sampled_locations_multiple_strengths",
                  "design_asset_sha256": manifest["asset_sha256"],
                  "design_sample_count": len(y), "design_samples_per_class": len(flows),
                  "design_leak_class_count": len(columns),
                  "requested_leak_class_count": manifest["sampled_leak_locations_per_layout"],
                  "available_design_cases_per_layout": manifest["available_design_cases_per_layout"],
                  "design_calls_per_layout_with_recovery": manifest["design_calls_per_layout_with_recovery"],
                  "strengths_lps": flows, "sampled_leak_node_ids": np.asarray(ids)[subset].tolist()}
