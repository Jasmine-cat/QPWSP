"""Graph edge transformations for the MVC objective."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def undirected_unique_edges(
    edges_idx: Iterable[Sequence[int]], *, remove_self: bool = True
) -> List[Tuple[int, int]]:
    """Convert graph edges to a deduplicated undirected set."""
    edges = set()
    for edge in edges_idx:
        if len(edge) != 2:
            continue
        u, v = int(edge[0]), int(edge[1])
        if remove_self and u == v:
            continue
        edges.add((u, v) if u <= v else (v, u))
    return sorted(edges)


def build_junction_edge_list_from_full_edges(
    *,
    edges_idx_full: Iterable[Sequence[int]],
    junction_idx_full: Sequence[int],
    N_full: int,
    keep_unselectable_unselectable_edges: bool = False,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, int]]:
    """Extract junction-only graph edges for the MVC objective."""
    junction = np.asarray(junction_idx_full, dtype=int).reshape(-1)
    if np.any(junction < 0) or np.any(junction >= int(N_full)):
        raise ValueError("junction_idx_full contains an out-of-range index")

    full_to_junction = {int(full): int(local) for local, full in enumerate(junction.tolist())}
    kept, skipped = set(), set()
    for edge in edges_idx_full:
        if len(edge) != 2:
            continue
        u, v = int(edge[0]), int(edge[1])
        ju, jv = full_to_junction.get(u), full_to_junction.get(v)
        if ju is None or jv is None:
            if keep_unselectable_unselectable_edges:
                skipped.add((u, v) if u <= v else (v, u))
            continue
        kept.add((ju, jv) if ju <= jv else (jv, ju))
    return sorted(kept), sorted(skipped), full_to_junction
