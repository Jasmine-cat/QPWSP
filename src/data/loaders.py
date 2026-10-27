import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

NUMERIC_NODE_FEATURES = [
    "elevation",
    "base_demand",
    "dist_to_source_euclid_norm",
    "dist_to_source_euclid_log_norm",#到水源的欧氏距离（归一化二选一），替代x/y坐标信息
    "deg_c",
    "betw_c",
    "close_c",
    "hop_to_source_min",
]



def load_H_for_nodes(H_path: str, node_ids: List[str]) -> np.ndarray:
    """
    Align all sensor rows and available leak columns by node ID.

    Supported:
      1) .npz with keys:
         - H
         - junction_names (row labels)
         - leak_node_names (column labels, optional; default to junction_names)
      2) .csv with node ids as both index and columns
    """
    ext = os.path.splitext(H_path)[1].lower()

    if ext == ".npz":
        d = np.load(H_path, allow_pickle=True)
        if "H" not in d.files:
            raise KeyError(f"H missing in npz: {H_path}. Current keys={list(d.files)}")
        H = np.asarray(d["H"], dtype=float)
        row_names = [str(x) for x in d["junction_names"]] if "junction_names" in d.files else None
        col_names = [str(x) for x in d["leak_node_names"]] if "leak_node_names" in d.files else row_names
        if row_names is None or col_names is None:
            if H.shape != (len(node_ids), len(node_ids)):
                raise ValueError(
                    f"H npz shape {H.shape} cannot be aligned to node_ids length {len(node_ids)} "
                    "because row/col names are missing."
                )
            return H

        if H.ndim != 2 or H.shape != (len(row_names), len(col_names)):
            raise ValueError("H dimensions do not match its row/column labels.")
        if len(set(row_names)) != len(row_names) or len(set(col_names)) != len(col_names):
            raise ValueError("Duplicate H node labels.")

        row_map = {name: i for i, name in enumerate(row_names)}
        col_map = {name: i for i, name in enumerate(col_names)}
        missing_rows = [nid for nid in node_ids if nid not in row_map]
        missing_cols = [nid for nid in node_ids if nid not in col_map]
        partial_leak_axis = ("leak_model" in d.files and
                             str(d["leak_model"].item()) == "constant_additional_demand")
        if missing_rows or (missing_cols and not partial_leak_axis):
            raise ValueError(
                f"H npz alignment failed. missing_rows={missing_rows[:5]}, "
                f"missing_cols={missing_cols[:5]}"
            )
        row_idx = [row_map[nid] for nid in node_ids]
        col_idx = [col_map[nid] for nid in node_ids if nid in col_map]
        if not col_idx:
            raise ValueError("H contains no usable leak-response columns.")
        return H[np.ix_(row_idx, col_idx)].astype(float)

    if ext == ".csv":
        df = pd.read_csv(H_path, index_col=0)
        cols = [str(c) for c in df.columns.tolist()]
        idx = [str(i) for i in df.index.tolist()]
        df.index = idx
        df.columns = cols
        if len(set(idx)) != len(idx) or len(set(cols)) != len(cols):
            raise ValueError("Duplicate H CSV node labels.")
        if not set(node_ids).issubset(set(idx)):
            raise ValueError(
                f"H csv cannot align to node ids. "
                f"missing_rows={list(set(node_ids)-set(idx))[:5]}, "
                f"missing_cols={list(set(node_ids)-set(cols))[:5]}"
            )
        retained_cols = [nid for nid in node_ids if nid in cols]
        if not retained_cols:
            raise ValueError("H CSV contains no usable leak-response columns.")
        df = df.loc[node_ids, retained_cols]
        return df.to_numpy(dtype=float)

    raise ValueError(f"Unsupported H format: {H_path}")
















def _compute_rel_all1_metrics(
    raw_metrics: Dict[str, Any],
    all1_ref: Dict[str, float],
    eps: float = 1e-8,
) -> Dict[str, float]:
    def _to_float(v: Any) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return fv if np.isfinite(fv) else float("nan")

    succ = _to_float(raw_metrics.get("_loc_success_raw"))
    mrr = _to_float(raw_metrics.get("_loc_mrr_raw"))
    err = _to_float(raw_metrics.get("_loc_error_dist_raw"))
    exp_err = _to_float(raw_metrics.get("_loc_expected_topo_error_raw"))
    cov = _to_float(raw_metrics.get("_water_demand_coverage_rate_raw"))

    succ_all1 = _to_float(all1_ref.get("success_rate"))
    mrr_all1 = _to_float(all1_ref.get("mrr"))
    err_all1 = _to_float(all1_ref.get("mean_topo_error"))
    exp_err_all1 = _to_float(all1_ref.get("mean_expected_topo_error"))
    cov_all1 = _to_float(all1_ref.get("water_demand_coverage_rate"))

    def _ratio_pos(num: float, den: float) -> float:
        if not (np.isfinite(num) and np.isfinite(den)):
            return float("nan")
        return float(num / (den + eps))

    def _ratio_inv(ref: float, val: float) -> float:
        if not (np.isfinite(ref) and np.isfinite(val)):
            return float("nan")
        return float((ref + eps) / (val + eps))

    return {
        "loc_success_rel_all1": _ratio_pos(succ, succ_all1),
        "loc_mrr_rel_all1": _ratio_pos(mrr, mrr_all1),
        "loc_error_dist_rel_all1": _ratio_inv(err_all1, err),
        "loc_expected_topo_error_rel_all1": _ratio_inv(exp_err_all1, exp_err),
        "water_demand_coverage_rate_rel_all1": _ratio_pos(cov, cov_all1),
    }




def build_junction_hop_dist_from_edges(
    N_all: int,
    edges_idx: List[Tuple[int, int]],
    junction_idx: np.ndarray,
    include_reservoir_in_paths: bool = True,
    unreachable_penalty: Optional[int] = None,
) -> np.ndarray:
    """
    计算 Junction 子集之间的 hop 最短路距离矩阵 dist_junc (N_j, N_j)。
    参数
    ----
    N_all : 全图节点数（Junction + Reservoir）
    edges_idx : 全图边 (u,v) 的索引列表（基于全图索引）
    junction_idx : Junction 在全图中的索引数组（长度 N_j）
    include_reservoir_in_paths :
        True  -> BFS 在全图上跑（允许路径经过 Reservoir）
        False -> 只在 Junction 诱导子图上跑（Reservoir 不参与路径）
    unreachable_penalty :
        不可达时填充值；默认 N_j（一个较大但有限的惩罚）

    返回
    ----
    dist_junc : (N_j, N_j) int
        dist_junc[a,b] = dist( junction_idx[a], junction_idx[b] )
    """
    junction_idx = np.asarray(junction_idx, dtype=int).reshape(-1)
    N_j = len(junction_idx)
    if unreachable_penalty is None:
        unreachable_penalty = N_j

    # 全图邻接表
    full_adj = [[] for _ in range(N_all)]
    for u, v in edges_idx:
        u = int(u); v = int(v)
        full_adj[u].append(v)
        full_adj[v].append(u)

    # Junction 映射：global -> local
    g2j = {int(g): j for j, g in enumerate(junction_idx)}

    # 若只允许 junction 子图路径，构造 junction-only 邻接表（节点仍用 global 编号更省事）
    if not include_reservoir_in_paths:
        jset = set(int(g) for g in junction_idx)
        adj = [[] for _ in range(N_all)]
        for g in jset:
            adj[g] = [nb for nb in full_adj[g] if nb in jset]
    else:
        adj = full_adj

    # BFS 从每个 Junction 出发（在 adj 上跑），取到其它 Junction 的距离
    dist_junc = np.full((N_j, N_j), fill_value=unreachable_penalty, dtype=int)

    for a, g_src in enumerate(junction_idx):
        g_src = int(g_src)
        dist = np.full(N_all, fill_value=-1, dtype=int)
        dist[g_src] = 0
        q = deque([g_src])

        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if dist[nb] < 0:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)

        # 只摘出到 junction 的距离
        for b, g_dst in enumerate(junction_idx):
            d = dist[int(g_dst)]
            dist_junc[a, b] = d if d >= 0 else unreachable_penalty

    return dist_junc

def parse_inp_options_units_pressure(inp_path: str) -> tuple[str, str | None]:
    """
    解析得到inp文件中需求/流量的单位
    返回 (UNITS, PRESSURE)：
      - units: e.g. LPS/GPM/CMH/CMS...
      - pressure: e.g. METERS/PSI/KPA... 或 None（没写）
    """
    sec = None
    units = None
    pressure = None
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith(";"):
                continue
            if raw.startswith("[") and raw.endswith("]"):
                sec = raw[1:-1].strip().upper()
                continue
            if sec != "OPTIONS":
                continue

            pure = raw.split(";", 1)[0].strip()
            parts = re.split(r"\s+", pure)
            if len(parts) >= 2:
                key = parts[0].upper()
                val = parts[1].upper()
                if key == "UNITS":
                    units = val
                elif key == "PRESSURE":
                    pressure = val
    # 找不到就返回 None/默认
    return (units or "UNKNOWN"), pressure
def length_unit_to_m_factor(units: str) -> float:
    """
    转换elevation单位
    elevation/head 的长度单位：
      - US units (e.g. GPM) -> ft
      - SI units (LPS/CMH/CMS) -> m
    返回 1 (inp length unit) = ? meters
    """
    u = str(units).upper()
    # EPANET: US = CFS, GPM, MGD, IMGD, AFD etc；你现在遇到主要是 GPM
    if u in ("GPM", "CFS", "MGD", "IMGD", "AFD"):
        return 0.3048  # ft -> m
    # SI：LPS/CMH/CMS 等通常为 m
    if u in ("LPS", "L/S", "CMH", "CMS", "M3/H", "M^3/H", "M3/S", "M^3/S"):
        return 1.0
    # UNKNOWN 时保守：不转换
    return 1.0


def pressure_to_m_factor(pressure_unit: str | None, units: str) -> float:
    """
    把 pressure 数值转为 meters of water head 的系数：
      P_m = P_inp * factor
    如果 [OPTIONS] PRESSURE 给了单位则优先；否则根据 UNITS 推断（US->psi, SI->m）。
    """
    if pressure_unit is not None:
        pu = pressure_unit.upper()
        if pu in ("METERS", "M", "METER"):
            return 1.0
        if pu in ("FEET", "FT"):
            return 0.3048
        if pu == "PSI":
            # 1 psi = 0.7030696 mH2O (approx, water at ~4°C)
            return 0.7030696
        if pu in ("KPA",):
            # 1 kPa = 0.1019716 mH2O
            return 0.1019716
        # 其他单位你可再扩展
        return 1.0

    # 未显式给 PRESSURE：按 UNITS 体系推断
    u = str(units).upper()
    if u in ("GPM", "CFS", "MGD", "IMGD", "AFD"):
        # US 系统默认 pressure 常见是 psi（很多工具）
        return 0.7030696
    # SI 默认 meters
    return 1.0
def unit_to_LPS_factor(units: str) -> float:
    """
    返回: 1 (unit) 等于多少 L/s
    """
    u = str(units).upper()
    if u in ("LPS", "L/S", "LITERS/SEC", "LITRES/SEC"):
        return 1.0
    if u in ("GPM", "GALLONS/MIN"):
        return 0.0630902
    if u in ("CMH", "M3/H", "M^3/H"):
        return 1000.0 / 3600.0  # 0.277777...
    if u in ("CMS", "M3/S", "M^3/S"):
        return 1000.0           # 1 m^3/s = 1000 L/s
    if u in ("LPD",):
        return 1.0 / 86400.0
    if u in ("MLD",):
        return 1e6 / 86400.0
    # 其他单位你可以按需要补
    raise ValueError(f"Unsupported Units='{units}'. Please add conversion.")
def add_euclid_dist_to_source_features(
    df_nodes: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    type_col: str = "type",
    source_types: tuple = ("reservoir", "tank"),
    eps: float = 1e-12,
    use_log1p: bool = True,
) -> pd.DataFrame:
    """
    处理节点信息表计算得“到水源欧式距离”属性
    新增：
      - dist_to_source_euclid
      - dist_to_source_euclid_norm
      - dist_to_source_euclid_log_norm (若 use_log1p=True)
    """
    df = df_nodes.copy()

    # 坐标检查
    if not {x_col, y_col}.issubset(df.columns):
        df["dist_to_source_euclid"] = np.nan
        df["dist_to_source_euclid_norm"] = np.nan
        if use_log1p:
            df["dist_to_source_euclid_log_norm"] = np.nan
        return df

    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)

    # 处理坐标缺失：NaN 的点先记 NaN，后面统一填充
    finite_xy = np.isfinite(xs) & np.isfinite(ys)

    # 找水源集合（Reservoir/Tank）
    if type_col not in df.columns:
        raise ValueError(f"df_nodes 缺少列 {type_col}，无法识别水源节点。")

    t = df[type_col].astype(str).str.lower()
    is_source = np.zeros(len(df), dtype=bool)
    for st in source_types:
        is_source |= t.str.contains(st)

    src_idx = np.flatnonzero(is_source & finite_xy)
    if src_idx.size == 0:
        # 没有水源坐标：退化为 NaN（或全0）
        df["dist_to_source_euclid"] = np.nan
        df["dist_to_source_euclid_norm"] = np.nan
        if use_log1p:
            df["dist_to_source_euclid_log_norm"] = np.nan
        return df

    # 计算到最近水源的欧氏距离：对每个点取 min_j ||p_i - s_j||
    # 向量化实现： (N,1,2) - (1,S,2) -> (N,S,2)
    P = np.stack([xs, ys], axis=1)             # (N,2)
    S = P[src_idx, :]                          # (S,2)

    d = np.full(len(df), np.nan, dtype=float)

    # 只对 finite_xy 的点计算
    P_f = P[finite_xy]                         # (Nf,2)
    diff = P_f[:, None, :] - S[None, :, :]     # (Nf,S,2)
    dist = np.sqrt(np.sum(diff * diff, axis=2))# (Nf,S)
    d_min = np.min(dist, axis=1)               # (Nf,)
    d[finite_xy] = d_min

    # 如果有 NaN（坐标缺失）可用最大值填充（表示“很远/未知”）
    if np.any(~np.isfinite(d)):
        finite_d = d[np.isfinite(d)]
        fill = float(np.max(finite_d)) if finite_d.size > 0 else 0.0
        d = np.where(np.isfinite(d), d, fill)

    # max 归一化


    if use_log1p:
        dlog = np.log1p(d)
        denom2 = max(float(np.max(dlog)), eps)
        df["dist_to_source_euclid_log_norm"] = dlog / denom2
    else:
        denom = max(float(np.max(d)), eps)
        df["dist_to_source_euclid_norm"] = d / denom
    return df


def parse_inp_links_with_meta(inp_path: str) -> List[Dict[str, Any]]:
    """
    解析inp图结构边信息——>元变量
    Parse links from .inp with lightweight edge metadata.
    Stable fields only: endpoints + coarse type + pipe length(if available).
    """
    links: List[Dict[str, Any]] = []
    section = None

    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith(";"):
                continue

            if raw.startswith("[") and raw.endswith("]"):
                section = raw[1:-1].strip().upper()
                continue

            if section not in ("PIPES", "PUMPS", "VALVES"):
                continue

            pure = raw.split(";", 1)[0].strip()
            if not pure:
                continue

            parts = pure.split()
            if len(parts) < 3:
                continue

            link_id, n1, n2 = parts[:3]
            link_type = "pipe" if section == "PIPES" else ("pump" if section == "PUMPS" else "valve")
            length = np.nan
            diameter = np.nan
            roughness = np.nan
            if section == "PIPES" and len(parts) >= 4:
                try:
                    length = float(parts[3])
                except Exception:
                    length = np.nan
            if section == "PIPES" and len(parts) >= 5:
                try:
                    diameter = float(parts[4])
                except Exception:
                    diameter = np.nan
            if section == "PIPES" and len(parts) >= 6:
                try:
                    roughness = float(parts[5])
                except Exception:
                    roughness = np.nan

            links.append(
                {
                    "link_id": str(link_id),
                    "u": str(n1),
                    "v": str(n2),
                    "link_type": link_type,
                    "length": length,
                    "diameter": diameter,
                    "roughness": roughness,
                }
            )

    return links


def _log1p_zscore_positive(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    out = np.zeros_like(x, dtype=np.float32)
    mask = np.isfinite(x) & (x > 0)
    if not np.any(mask):
        return out
    y = np.log1p(x[mask])
    mu = float(np.mean(y))
    std = float(np.std(y)) + 1e-8
    out[mask] = ((y - mu) / std).astype(np.float32)
    return out


def _guess_edge_features_csv_from_nodes_csv(nodes_csv: str) -> Optional[str]:
    if not nodes_csv or (not os.path.isfile(nodes_csv)):
        return None

    folder = os.path.dirname(nodes_csv)
    base = os.path.basename(nodes_csv)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".csv":
        return None

    candidates = []
    if stem.endswith("_node_features"):
        name = stem[: -len("_node_features")]
        candidates.append(os.path.join(folder, f"{name}_edge_features.csv"))
    candidates.append(os.path.join(folder, stem.replace("_node_features", "_edge_features") + ".csv"))

    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _get_method_cfg(cfg_list: Optional[List[Dict[str, Any]]], method: str) -> Dict[str, Any]:
    """从 cfg_list 中取出对应 method 的 cfg（找不到则返回空 dict）。"""
    if not cfg_list:
        return {}
    for item in cfg_list:
        if item.get("method") == method:
            cfg = item.get("cfg", {})
            return cfg if isinstance(cfg, dict) else {}
    return {}
def load_graph_data(
    nodes_csv: str,
    edge_feat_csv: str,
    inp_path: str,
    drop_non_junction: bool = False,
    standardize: bool = True,#节点特征标准化，如坐标，中心距离等（受规模影响范围大）
    *,
    x_mode: str = "full",   # 新增：full / ones / degree / ones+degree
    return_graph_struct: bool = False,  # 新增：是否返回轻量物理注入所需图结构
    filter_negative_demand_junctions: bool = True,
) -> Tuple[
    np.ndarray,          # X_all
    np.ndarray,          # A_all
    np.ndarray,          # A_norm_all
    List[str],           # node_ids_all
    Dict[str, int],      # id2idx_all
    np.ndarray,          # edges_idx
    pd.DataFrame,        # df_nodes_all
    np.ndarray,          # junction_indices
    List[str],           # junction_node_ids
]:
    import pandas as pd
    import numpy as np

    # ---------- 1. 读全量节点 ----------
    df_all = pd.read_csv(nodes_csv)

    # --- 修正排序逻辑：数值 ID 在前，非数值 ID 在后 ---
    node_num = pd.to_numeric(df_all["node"], errors="coerce")
    df_all = df_all.assign(
        node_num=node_num,
        is_nan=node_num.isna(),
    ).sort_values(
        by=["is_nan", "node_num", "node"],
        ascending=[True, True, True],
    ).drop(columns=["node_num", "is_nan"])
    df_all.reset_index(drop=True, inplace=True)

    if drop_non_junction:
        df_nodes = df_all[df_all["type"] == "Junction"].copy()
        df_nodes.reset_index(drop=True, inplace=True)
    else:
        df_nodes = df_all.copy()

    # 节点 ID & 映射（全图）
    node_ids: List[str] = df_nodes["node"].astype(str).tolist()
    id2idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    # ---------- 1.4 Junction 子集 ----------
    if "type" not in df_nodes.columns:
        raise ValueError("nodes_feature.csv 中缺少 'type' 列。")
    type_series = df_nodes["type"].astype(str)
    junction_mask = (type_series == "Junction").to_numpy()
    if filter_negative_demand_junctions and "base_demand" in df_nodes.columns:
        # 负需求 junction 通常更像补给/边界等建模节点，不作为泄漏/传感器候选；
        # 节点仍保留在全图中，避免破坏拓扑路径和边结构。
        base_demand_all = pd.to_numeric(df_nodes["base_demand"], errors="coerce").to_numpy(dtype=float)
        neg_junction_mask = junction_mask & np.isfinite(base_demand_all) & (base_demand_all < 0)
        if np.any(neg_junction_mask):
            skipped = df_nodes.loc[neg_junction_mask, "node"].astype(str).tolist()
            print(
                f"[data_utils] 从候选集合剔除 {len(skipped)} 个负需求 junction: "
                f"{skipped[:8]}{'...' if len(skipped) > 8 else ''}"
            )
            junction_mask = junction_mask & (~neg_junction_mask)
    junction_indices = np.nonzero(junction_mask)[0]
    junction_node_ids = [node_ids[i] for i in junction_indices]
    print(f"[data_utils] Junction subset size = {len(junction_indices)}.")

    # ---------- 2. 边 ----------
    edge_feat_df = None

    if edge_feat_csv is not None:
        try:
            _df_edge = pd.read_csv(edge_feat_csv)
            if {"u", "v"}.issubset(_df_edge.columns):
                edge_feat_df = _df_edge
                print(f"[data_utils] Using precomputed edge features: {edge_feat_csv}")
            else:
                print(f"[data_utils] WARNING: edge csv missing 'u/v' columns, fallback to INP parse: {edge_feat_csv}")
        except Exception as e:
            print(f"[data_utils] WARNING: failed to read edge csv, fallback to INP parse: {repr(e)}")

    edges = []
    edge_types: List[str] = []
    edge_lengths: List[float] = []
    edge_diameters: List[float] = []
    edge_roughness: List[float] = []
    edge_attr_precomputed = None

    if edge_feat_df is not None:
        has_pre = {
            "len_log1p_z",
            "abs_elev_diff_log1p_z",
            "diam_log1p_z",
            "rough_log1p_z",
            "is_pipe",
            "is_pump",
            "is_valve",
        }.issubset(edge_feat_df.columns)
        if not has_pre:
            raise ValueError(
                "edge features schema mismatch: expected new columns "
                "[len_log1p_z, abs_elev_diff_log1p_z, diam_log1p_z, rough_log1p_z, is_pipe, is_pump, is_valve]. "
                f"Got columns: {list(edge_feat_df.columns)}"
            )
        pre_rows: List[List[float]] = []
        for _, row in edge_feat_df.iterrows():
            u_str = str(row.get("u", ""))
            v_str = str(row.get("v", ""))
            if u_str in id2idx and v_str in id2idx:
                iu = id2idx[u_str]
                iv = id2idx[v_str]
                edges.append((iu, iv))
                edge_types.append(str(row.get("link_type", "pipe")).lower())
                if "length_m" in edge_feat_df.columns:
                    edge_lengths.append(float(row.get("length_m", np.nan)))
                elif "length_raw" in edge_feat_df.columns:
                    edge_lengths.append(float(row.get("length_raw", np.nan)))
                else:
                    edge_lengths.append(float(row.get("length", np.nan)))
                edge_diameters.append(float(row.get("diameter_m", row.get("diameter_raw", np.nan))))
                edge_roughness.append(float(row.get("roughness_raw", np.nan)))
                pre_rows.append(
                    [
                        float(row.get("len_log1p_z", 0.0)),
                        float(row.get("abs_elev_diff_log1p_z", 0.0)),
                        float(row.get("diam_log1p_z", 0.0)),
                        float(row.get("rough_log1p_z", 0.0)),
                        float(row.get("is_pipe", 0.0)),
                        float(row.get("is_pump", 0.0)),
                        float(row.get("is_valve", 0.0)),
                    ]
                )
        if has_pre and len(pre_rows) == len(edges):
            edge_attr_precomputed = np.asarray(pre_rows, dtype=np.float32)
    else:
        link_items = parse_inp_links_with_meta(inp_path)
        for it in link_items:
            u_str = str(it["u"])
            v_str = str(it["v"])
            if u_str in id2idx and v_str in id2idx:
                iu = id2idx[u_str]
                iv = id2idx[v_str]
                edges.append((iu, iv))
                edge_types.append(str(it.get("link_type", "pipe")))
                edge_lengths.append(float(it.get("length", np.nan)))
                edge_diameters.append(float(it.get("diameter", np.nan)))
                edge_roughness.append(float(it.get("roughness", np.nan)))

    edges_idx = np.array(edges, dtype=np.int64)
    print(f"[data_utils] Loaded {len(edges_idx)} edges (between selected nodes).")

    # ---------- 3. 邻接矩阵（结构无 self-loop） ----------
    N = df_nodes.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    for iu, iv in edges_idx:
        A[iu, iv] = 1.0
        A[iv, iu] = 1.0
    np.fill_diagonal(A, 0.0)

    # 原始结构度（不含 self-loop）——给 degree 特征用
    deg_raw = A.sum(axis=1).astype(np.float32)

    # ---------- 3.1 GCN 归一化（训练用，可用 A+I） ----------
    A_tilde = A + np.eye(N, dtype=np.float32)
    deg_tilde = A_tilde.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg_tilde, dtype=np.float32)
    np.power(deg_tilde, -0.5, where=deg_tilde > 0, out=deg_inv_sqrt)
    D_inv_sqrt = np.diag(deg_inv_sqrt.astype(np.float32))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # ---------- 1.1~1.3 构造 X（根据 x_mode 切换） ----------
    x_mode = str(x_mode).lower()

    if x_mode == "full":
        # ---------- 1.1 数值特征 ----------
        feature_cols = [c for c in NUMERIC_NODE_FEATURES if c in df_nodes.columns]
        if not feature_cols:
            raise ValueError("nodes_feature.csv 中没有可用的数值特征列，请检查。")

        df_num = df_nodes[feature_cols].astype(float)

        nan_cols = df_num.columns[df_num.isna().any()].tolist()
        if nan_cols:
            print(f"[data_utils] WARNING: numeric features contain NaN in columns: {nan_cols}")
            for col in nan_cols:
                col_mean = df_num[col].mean()
                print(f"[data_utils]   -> filling NaN in '{col}' with mean = {col_mean:.6f}")
                df_num[col] = df_num[col].fillna(col_mean)

        X_num = df_num.to_numpy(dtype=np.float32)

        # ---------- 1.2 type one-hot ----------
        type_categories = sorted(type_series.unique())
        type_to_idx = {t: i for i, t in enumerate(type_categories)}
        num_types = len(type_categories)
        X_type = np.zeros((N, num_types), dtype=np.float32)
        for i, t in enumerate(type_series):
            X_type[i, type_to_idx[t]] = 1.0

        # ---------- 1.3 标准化数值特征 + 拼接 ----------
        if standardize:
            mean = X_num.mean(axis=0, keepdims=True)
            std = X_num.std(axis=0, keepdims=True) + 1e-8
            X_num = (X_num - mean) / std

        X = np.concatenate([X_num, X_type], axis=1).astype(np.float32)

        if not np.isfinite(X).all():
            raise ValueError("[data_utils] X contains NaN/Inf after preprocessing, please check.")

        print(
            f"[data_utils] X_mode=full, feature dim = {X.shape[1]} "
            f"(numeric {X_num.shape[1]} + type_one_hot {X_type.shape[1]})."
        )

    elif x_mode == "ones":
        # 纯结构：全 1 占位，不含任何节点属性
        X = np.ones((N, 1), dtype=np.float32)
        print("[data_utils] X_mode=ones, feature dim = 1 (no node attributes).")


    elif x_mode == "degree":

        # 纯结构：用原始度（不含 self-loop）

        X = deg_raw.reshape(-1, 1).astype(np.float32)

        if standardize:
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

        print("[data_utils] X_mode=degree, feature dim = 1 (structure-only degree_raw).")


    elif x_mode == "ones+degree":

        X1 = np.ones((N, 1), dtype=np.float32)

        Xd = deg_raw.reshape(-1, 1).astype(np.float32)

        if standardize:
            Xd = (Xd - Xd.mean(axis=0, keepdims=True)) / (Xd.std(axis=0, keepdims=True) + 1e-8)

        X = np.concatenate([X1, Xd], axis=1).astype(np.float32)

        print("[data_utils] X_mode=ones+degree, feature dim = 2 (structure-only, degree_raw).")

    else:
        raise ValueError(f"Unknown x_mode='{x_mode}'. Use: full / ones / degree / ones+degree.")

    if not return_graph_struct:
        return X, A, A_norm, node_ids, id2idx, edges_idx, df_nodes, junction_indices, junction_node_ids

    # ---------- 4. 轻量物理注入图结构 ----------
    E = int(edges_idx.shape[0])
    edge_index = edges_idx.T.astype(np.int64) if E > 0 else np.zeros((2, 0), dtype=np.int64)

    # B_incidence: node-edge signed incidence (+1 at source, -1 at target)
    B_incidence = np.zeros((N, E), dtype=np.float32)
    for e, (u, v) in enumerate(edges):
        B_incidence[int(u), e] = 1.0
        B_incidence[int(v), e] = -1.0

    # Edge features: [len_log1p_z, abs_elev_diff_log1p_z, diam_log1p_z, rough_log1p_z, is_pipe, is_pump, is_valve]
    if (edge_attr_precomputed is not None) and (edge_attr_precomputed.shape == (E, 7)):
        E_edge = edge_attr_precomputed.astype(np.float32)
    else:
        edge_len = np.asarray(edge_lengths, dtype=float) if len(edge_lengths) == E else np.full(E, np.nan, dtype=float)
        edge_len_feat = _log1p_zscore_positive(edge_len)

        edge_diam = np.asarray(edge_diameters, dtype=float) if len(edge_diameters) == E else np.full(E, np.nan, dtype=float)
        edge_diam_feat = _log1p_zscore_positive(edge_diam)

        edge_rough = np.asarray(edge_roughness, dtype=float) if len(edge_roughness) == E else np.full(E, np.nan, dtype=float)
        edge_rough_feat = _log1p_zscore_positive(edge_rough)

        elev = pd.to_numeric(df_nodes.get("elevation", pd.Series(np.zeros(N))), errors="coerce").to_numpy(dtype=float)
        elev_diff = np.zeros(E, dtype=float)
        for e, (u, v) in enumerate(edges):
            eu = elev[int(u)] if int(u) < elev.shape[0] else np.nan
            ev = elev[int(v)] if int(v) < elev.shape[0] else np.nan
            elev_diff[e] = abs(eu - ev) if (np.isfinite(eu) and np.isfinite(ev)) else np.nan
        elev_diff_feat = _log1p_zscore_positive(elev_diff)

        edge_types_arr = np.asarray(edge_types, dtype=object) if len(edge_types) == E else np.array(["pipe"] * E, dtype=object)
        is_pipe = (edge_types_arr == "pipe").astype(np.float32)
        is_pump = (edge_types_arr == "pump").astype(np.float32)
        is_valve = (edge_types_arr == "valve").astype(np.float32)

        E_edge = np.stack(
            [edge_len_feat, elev_diff_feat, edge_diam_feat, edge_rough_feat, is_pipe, is_pump, is_valve],
            axis=1,
        ).astype(np.float32)

    graph_struct: Dict[str, Any] = {
        "X_node": X.astype(np.float32),
        "E_edge": E_edge,
        "edge_attr": E_edge,  # alias
        "A_topology": A.astype(np.float32),
        "edge_index": edge_index,
        "B_incidence": B_incidence,
        "edge_uv": np.asarray(edges, dtype=np.int64) if E > 0 else np.zeros((0, 2), dtype=np.int64),
    }

    return X, A, A_norm, node_ids, id2idx, edges_idx, df_nodes, junction_indices, junction_node_ids, graph_struct



