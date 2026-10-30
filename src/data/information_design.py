"""Reuse the H-response design asset for a matched-information comparison."""
from __future__ import annotations

import hashlib
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
