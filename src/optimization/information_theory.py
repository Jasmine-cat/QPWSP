"""Information-theoretic sensor selection, without a QUBO reformulation.

Santos-Ruiz et al., Sensors 22 (2022), 443, Algorithm 1:
https://doi.org/10.3390/s22020443
The score is relevance divided by mean redundancy. Equal-width quantization
is fitted on design pressures only; 16 bins is the comparison setting here.
"""
from __future__ import annotations

import numpy as np


def quantize_design(pressures, bins=16):
    pressures = np.asarray(pressures, dtype=float)
    if pressures.ndim != 2 or not pressures.size or not np.isfinite(pressures).all():
        raise ValueError("Design pressures must be a finite sample-by-sensor matrix.")
    if isinstance(bins, bool) or int(bins) != bins or bins < 2:
        raise ValueError("The number of bins must be an integer of at least two.")
    bins = int(bins)
    low, high = pressures.min(axis=0), pressures.max(axis=0)
    span = high - low
    scaled = np.divide(pressures - low, span, out=np.zeros_like(pressures), where=span > 0)
    codes = np.minimum(np.floor(scaled * bins), bins - 1).astype(np.int32)
    return codes, {"low": low, "high": high, "constant": span == 0}


def _mutual_information(codes, target, alphabet_size, target_size, marginals):
    """Compute empirical MI in bounded-memory blocks of sensor columns."""
    samples, features = codes.shape
    target_marginal = np.bincount(target, minlength=target_size)
    output = np.empty(features)
    cells = alphabet_size * target_size
    block = max(1, min(128, 2_000_000 // samples, 2_000_000 // cells))
    for start in range(0, features, block):
        end = min(features, start + block)
        width = end - start
        joint_codes = (codes[:, start:end].T.astype(np.int64) * target_size
                       + target[None, :]
                       + np.arange(width, dtype=np.int64)[:, None] * cells)
        counts = np.bincount(joint_codes.ravel(), minlength=width * cells)
        counts = counts.reshape(width, alphabet_size, target_size)
        nonzero = counts > 0
        denominator = marginals[start:end, :, None] * target_marginal[None, None, :]
        ratios = np.ones(counts.shape, dtype=float)
        np.divide(counts.astype(float) * samples, denominator, out=ratios, where=nonzero)
        np.log2(ratios, out=ratios, where=nonzero)
        output[start:end] = np.maximum(0.0, np.sum((counts / samples) * ratios, axis=(1, 2)))
    return output


def rank_sensors(pressures, leak_labels, K, node_ids, *, bins=16):
    """Select exactly K sensors from labeled design pressures.

    Rows are design samples, columns follow node_ids, and labels identify leak
    locations. Ties follow the supplied column order. The method is deterministic
    for a fixed design pool; optimizer seeds do not create independent repeats.
    """
    codes, quantization = quantize_design(pressures, bins)
    bins = int(bins)
    labels = np.asarray(leak_labels)
    ids = list(map(str, node_ids))
    if labels.shape != (len(codes),):
        raise ValueError("One leak-class label is required for each design sample.")
    if (codes.shape[1] != len(ids) or len(set(ids)) != len(ids)
            or isinstance(K, bool) or int(K) != K or not 0 < K <= len(ids)):
        raise ValueError("Invalid node labels or sensor budget.")
    K = int(K)
    _, label_codes = np.unique(labels, return_inverse=True)
    classes = int(label_codes.max()) + 1
    if classes < 2:
        raise ValueError("Information-theoretic selection requires at least two leak classes.")
    marginals = np.stack([np.bincount(codes[:, j], minlength=bins) for j in range(len(ids))])
    relevance = _mutual_information(codes, label_codes, bins, classes, marginals)
    selected, trace = [], []
    available = np.ones(len(ids), dtype=bool)
    redundancy_sum = np.zeros(len(ids))
    tol = 1e-12  # Comparison tolerance; never added to the MI quotient.
    while len(selected) < K:
        candidates = np.flatnonzero(available)
        if not selected:
            chosen = int(candidates[np.argmax(relevance[candidates])])
            branch, redundancy, score = "max_relevance", 0.0, relevance[chosen]
        else:
            mean_redundancy = redundancy_sum / len(selected)
            positive = candidates[relevance[candidates] > tol]
            zero_red = positive[mean_redundancy[positive] <= tol]
            if len(zero_red):
                chosen = int(zero_red[np.argmax(relevance[zero_red])])
                branch = "positive_relevance_zero_redundancy"
                redundancy, score = mean_redundancy[chosen], None
            elif len(positive):
                scores = relevance[positive] / mean_redundancy[positive]
                chosen = int(positive[np.argmax(scores)])
                branch = "relevance_over_mean_redundancy"
                redundancy, score = mean_redundancy[chosen], float(scores.max())
            else:
                chosen = int(candidates[0])
                branch = "append_zero_relevance"
                redundancy, score = mean_redundancy[chosen], 0.0
        selected.append(chosen)
        available[chosen] = False
        trace.append({"step": len(selected), "node_id": ids[chosen], "branch": branch,
                      "relevance_bits": float(relevance[chosen]),
                      "mean_redundancy_bits": float(redundancy),
                      "RRI": None if score is None else float(score)})
        if len(selected) < K:
            remaining = np.flatnonzero(available)
            redundancy_sum[remaining] += _mutual_information(
                codes[:, remaining], codes[:, chosen], bins, bins, marginals[remaining])
    class_counts = np.bincount(label_codes)
    probabilities = class_counts / len(labels)
    return {"selected_indices": np.asarray(selected),
            "selected_node_ids": [ids[i] for i in selected], "trace": trace,
            "relevance_bits": relevance,
            "class_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
            "quantization": quantization, "bins": bins}
