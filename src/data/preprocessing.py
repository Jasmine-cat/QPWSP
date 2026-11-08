import copy
import glob
import os
import random
import math
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr
import networkx as nx
import pandas as pd
import numpy as np
import pathlib as _P

from src.data.loaders import parse_inp_options_units_pressure, unit_to_LPS_factor, length_unit_to_m_factor, \
    pressure_to_m_factor, add_euclid_dist_to_source_features, parse_inp_links_with_meta
from src.data.time_selection import (
    compute_selected_ptime,
    SELECTED_PTIME_METHOD,
    SELECTED_PTIME_WINDOW_STEPS,
)


# =========================
# EPYNET
# =========================
try:
    from epynet import Network
    from epynet import epanet2
except Exception as e:
    Network = None
    epanet2 = None
    _EPYNET_IMPORT_ERR = e


def _require_epynet():
    if Network is None or epanet2 is None:
        raise ImportError(
            "epynet is missing or could not be imported; the EPYNET backend is unavailable.\n"
            f"Import error: {repr(_EPYNET_IMPORT_ERR)}\n"
            "Install epynet with pip install epynet."
        )


def _safe_close_network(wn):
    if wn is None:
        return
    close_fn = getattr(wn, "close", None)
    if callable(close_fn):
        try:
            # epynet close() may print "closing" repeatedly; suppress it.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                close_fn()
        except Exception:
            pass


def _iter_nodes(wn) -> list:
    """Return node objects from an epynet network."""
    if hasattr(wn, "nodes"):
        nodes_obj = wn.nodes
        if isinstance(nodes_obj, dict):
            return list(nodes_obj.values())

        try:
            return list(nodes_obj)
        except Exception:
            pass
    raise RuntimeError("epynet Network has no iterable nodes container.")


def _iter_links(wn) -> list:
    """Return link objects from an epynet network."""
    if hasattr(wn, "links"):
        links_obj = wn.links
        if isinstance(links_obj, dict):
            return list(links_obj.values())
        try:
            return list(links_obj)
        except Exception:
            pass
    return []


def _node_id(node) -> str:
    """Return the node identifier across supported epynet versions."""
    if hasattr(node, "uid"):
        return str(node.uid)
    if hasattr(node, "id"):
        return str(node.id)

    return str(node)


def _node_type_str(node) -> str:
    """Normalize a node type to Junction, Reservoir, Tank, or Other."""
    t = node.__class__.__name__.lower()
    if "junction" in t:
        return "Junction"
    if "reservoir" in t:
        return "Reservoir"
    if "tank" in t:
        return "Tank"

    if hasattr(node, "type"):
        tt = str(getattr(node, "type")).lower()
        if "junction" in tt:
            return "Junction"
        if "reservoir" in tt:
            return "Reservoir"
        if "tank" in tt:
            return "Tank"
    return node.__class__.__name__


def _node_elevation_or_head(node) -> float:
    """Return a node elevation or reservoir head."""
    if hasattr(node, "elevation"):
        try:
            return float(node.elevation)
        except Exception:
            pass
    if hasattr(node, "head"):
        try:
            return float(node.head)
        except Exception:
            pass
    return float("nan")


def _node_base_demand(node) -> float:
    """Return a junction base demand."""
    for attr in ["basedemand", "base_demand", "demand"]:
        if hasattr(node, attr):
            try:
                v = float(getattr(node, attr))
                return v
            except Exception:
                continue
    return 0.0


def _node_xy(node) -> tuple[float, float]:
    """Return the node coordinates."""
    if hasattr(node, "coordinates") and node.coordinates is not None:
        try:
            x, y = node.coordinates
            return float(x), float(y)
        except Exception:
            pass
    if hasattr(node, "x") and hasattr(node, "y"):
        try:
            return float(node.x), float(node.y)
        except Exception:
            pass
    return float("nan"), float("nan")


def _link_endpoints(link) -> tuple[str, str] | None:
    """Return the identifiers of the endpoints of a link."""

    for a, b in [("start_node", "end_node"), ("startnode", "endnode"),
                 ("from_node", "to_node"), ("fromnode", "tonode")]:
        if hasattr(link, a) and hasattr(link, b):
            u = getattr(link, a)
            v = getattr(link, b)
            if u is None or v is None:
                continue
            return _node_id(u), _node_id(v)


    for a, b in [("node1", "node2")]:
        if hasattr(link, a) and hasattr(link, b):
            u = getattr(link, a)
            v = getattr(link, b)
            if u is None or v is None:
                continue
            return _node_id(u), _node_id(v)

    return None


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


def _set_time_params_epynet(wn, duration_s: int, step_s: int):
    """Set extended-period simulation time parameters via the EPANET toolkit."""
    if not hasattr(wn, "ep") or wn.ep is None:
        raise RuntimeError("epynet Network has no toolkit handle wn.ep")

    wn.ep.ENsettimeparam(epanet2.EN_DURATION, int(duration_s))
    wn.ep.ENsettimeparam(epanet2.EN_HYDSTEP, 3600)
    wn.ep.ENsettimeparam(epanet2.EN_REPORTSTEP, int(step_s))

def _extract_pressure_reporttimeseries_from_run(wn) -> pd.DataFrame:
    """Extract pressures at report times from a completed epynet run."""

    rep_step = wn.ep.ENgettimeparam(epanet2.EN_REPORTSTEP)
    rep_start = wn.ep.ENgettimeparam(epanet2.EN_REPORTSTART)


    times = np.asarray(list(map(int, wn.time)), dtype=int)
    if times.size == 0:
        raise RuntimeError("wn.time is empty. Did you call wn.run()?")


    mask = (times >= rep_start) & (((times - rep_start) % rep_step) == 0)
    rep_times = times[mask]
    if rep_times.size == 0:
        raise RuntimeError(
            f"No report times found in wn.time. "
            f"rep_step={rep_step}, rep_start={rep_start}, "
            f"time_span=[{times.min()},{times.max()}]"
        )


    nodes = _iter_nodes(wn)
    cols = []
    data_cols = []

    for node in nodes:
        nid = _node_id(node)
        if not hasattr(node, "results") or node.results is None:
            raise RuntimeError(f"node {nid} has no results.")
        if "pressure" not in node.results:
            raise RuntimeError(f"node {nid} missing results['pressure'], keys={list(node.results.keys())}")

        series = np.asarray(node.results["pressure"], dtype=float).reshape(-1)
        if series.shape[0] != times.shape[0]:
            raise RuntimeError(f"node {nid} pressure len={series.shape[0]} != time len={times.shape[0]}")

        cols.append(nid)
        data_cols.append(series[mask])

    M = np.vstack(data_cols).T  # (T_report, N)
    dfP = pd.DataFrame(M, index=pd.Index(rep_times, name="time"), columns=cols)
    return dfP
def _extract_pressure_timeseries_from_run2(wn) -> pd.DataFrame:
    """Extract the node pressure time series from a completed epynet run."""
    if not hasattr(wn, "time") or wn.time is None or len(wn.time) == 0:
        raise RuntimeError("epynet network has no time axis. Did you call wn.run()?")

    times = list(map(int, wn.time))
    nodes = _iter_nodes(wn)
    cols = []
    data_cols = []

    for node in nodes:
        nid = _node_id(node)
        if (not hasattr(node, "results")) or (node.results is None):
            raise RuntimeError(f"node {nid} has no results. Did wn.run() collect attributes?")

        if "pressure" not in node.results:


            raise RuntimeError(
                f"node {nid} has no results['pressure'] keys={list(node.results.keys())}"
            )

        series = node.results["pressure"]
        if len(series) != len(times):
            raise RuntimeError(
                f"node {nid} pressure length {len(series)} != len(times) {len(times)}"
            )

        cols.append(nid)
        data_cols.append(np.asarray(series, dtype=float).reshape(-1))

    M = np.vstack(data_cols).T  # (T, N)

    dfP = pd.DataFrame(M, index=pd.Index(times, name="time"), columns=cols)

    return dfP


def run_inp_to_features(
    inp_path: str,
    out_dir: str | None = None,
    *,
    duration_s: int = 24 * 3600,
    report_step_s: int = 3600,
    convert_to_SI: bool = True,
):
    """Generate graph features and baseline pressures from an EPANET input network."""
    _require_epynet()

    inp_path = _P.Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    out = _P.Path(out_dir) if out_dir else inp_path.parent / "node_features"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Processing network; output directory: {out.resolve()}")

    # --------------------------

    # --------------------------

    units, pressure_unit = parse_inp_options_units_pressure(str(inp_path))

    k_flow = float(unit_to_LPS_factor(units))   # 1 (inp flow unit) = ? L/s



    k_len = float(length_unit_to_m_factor(units))

    # pressure -> meters of water head
    k_p = float(pressure_to_m_factor(pressure_unit, units))

    # --------------------------

    # --------------------------
    wn = Network(str(inp_path))
    try:
        wn._inp_path = str(inp_path)
    except Exception:
        pass

    nodes_obj = _iter_nodes(wn)
    links_obj = _iter_links(wn)

    node_ids = [_node_id(n) for n in nodes_obj]
    print(f"[INFO] Loaded network with {len(node_ids)} nodes and {len(links_obj)} links")
    print(f"[INFO] INP Units={units}, PressureUnit={pressure_unit}, convert_to_SI={convert_to_SI}")

    # --------------------------

    # --------------------------
    types = []
    elev = []
    base_demand = []
    xcoord = []
    ycoord = []

    for n in nodes_obj:
        types.append(_node_type_str(n))
        elev.append(_node_elevation_or_head(n))
        base_demand.append(_node_base_demand(n))
        x, y = _node_xy(n)
        xcoord.append(x)
        ycoord.append(y)

    # --------------------------
    # (3) edges（links -> endpoints）
    # --------------------------
    edges = []
    for l in links_obj:
        ep = _link_endpoints(l)
        if ep is None:
            continue
        u, v = ep
        edges.append((u, v))

    # --------------------------

    # --------------------------
    G_u = nx.Graph()
    G_u.add_nodes_from(node_ids)
    G_u.add_edges_from(edges)

    deg = nx.degree_centrality(G_u)
    btw = nx.betweenness_centrality(G_u)
    clo = nx.closeness_centrality(G_u)

    node_feats = pd.DataFrame({
        "node": node_ids,
        "type": types,
        "elevation": elev,
        "base_demand": base_demand,
        "x": xcoord,
        "y": ycoord,
        "deg_c": [deg.get(n, 0.0) for n in node_ids],
        "betw_c": [btw.get(n, 0.0) for n in node_ids],
        "close_c": [clo.get(n, 0.0) for n in node_ids],
    }).set_index("node")


    sources = [nid for nid, tp in zip(node_ids, types) if str(tp).lower() in ("reservoir", "tank")]
    if sources:
        sp = {}
        for n in node_ids:
            dists = []
            for s in sources:
                try:
                    d = nx.shortest_path_length(G_u, n, s)
                    dists.append(d)
                except nx.NetworkXNoPath:
                    continue
            sp[n] = min(dists) if dists else np.nan
        node_feats["hop_to_source_min"] = pd.Series(sp)
    else:
        node_feats["hop_to_source_min"] = np.nan

    # --------------------------

    # --------------------------
    try:
        _set_time_params_epynet(wn, duration_s=int(duration_s), step_s=int(report_step_s))
        wn.run()
        pressure = _extract_pressure_reporttimeseries_from_run(wn)  # raw pressure in INP's pressure unit system
    finally:
        _safe_close_network(wn)

    # --------------------------

    # --------------------------
    conv_info = {
        "orig_units": units,
        "orig_pressure_unit": pressure_unit,
        "target_units": "\u539f\u59cbraw_inp_units",
        "flow_to_LPS_factor": float(k_flow),
        "len_to_m_factor": float(k_len),
        "pressure_to_m_factor": float(k_p),
    }

    if convert_to_SI:
        # base_demand -> L/s
        node_feats["base_demand"] = node_feats["base_demand"].astype(float) * k_flow

        # elevation/head -> m
        node_feats["elevation"] = node_feats["elevation"].astype(float) * k_len

        # pressure -> mH2O
        pressure = pressure.astype(float) * k_p

        conv_info["target_units"] = "SI(base_demand=L/s, elevation=m, pressure=mH2O)"
    node_feats = add_euclid_dist_to_source_features(node_feats, use_log1p=False)
    # --------------------------

    # --------------------------
    # --------------------------
    # edge static features (E_edge)
    # --------------------------
    link_items = parse_inp_links_with_meta(str(inp_path))
    edge_rows = []
    idx_set = set(node_feats.index.astype(str).tolist())

    if convert_to_SI:
        elev_m_map = pd.to_numeric(node_feats["elevation"], errors="coerce").to_dict()
    else:
        elev_m_map = (pd.to_numeric(node_feats["elevation"], errors="coerce") * float(k_len)).to_dict()

    for i, it in enumerate(link_items):
        u = str(it.get("u", "")).strip()
        v = str(it.get("v", "")).strip()
        if (u not in idx_set) or (v not in idx_set):
            continue

        link_type = str(it.get("link_type", "pipe")).strip().lower()
        if link_type not in ("pipe", "pump", "valve"):
            link_type = "pipe"

        try:
            length_raw = float(it.get("length", np.nan))
        except Exception:
            length_raw = np.nan
        length_m = float(length_raw * k_len) if np.isfinite(length_raw) else np.nan

        try:
            diameter_raw = float(it.get("diameter", np.nan))
        except Exception:
            diameter_raw = np.nan
        diameter_m = float(diameter_raw * k_len) if np.isfinite(diameter_raw) else np.nan

        try:
            roughness_raw = float(it.get("roughness", np.nan))
        except Exception:
            roughness_raw = np.nan

        eu = elev_m_map.get(u, np.nan)
        ev = elev_m_map.get(v, np.nan)
        elev_diff_m = abs(float(eu) - float(ev)) if (np.isfinite(eu) and np.isfinite(ev)) else np.nan

        edge_id = str(it.get("link_id", "")).strip() or f"E{i:05d}"
        edge_rows.append(
            {
                "edge": edge_id,
                "u": u,
                "v": v,
                "link_type": link_type,
                "length_raw": length_raw,
                "length_m": length_m,
                "diameter_raw": diameter_raw,
                "diameter_m": diameter_m,
                "roughness_raw": roughness_raw,
                "abs_elev_diff_m": elev_diff_m,
            }
        )

    if not edge_rows:
        for i, (u, v) in enumerate(edges):
            eu = elev_m_map.get(str(u), np.nan)
            ev = elev_m_map.get(str(v), np.nan)
            elev_diff_m = abs(float(eu) - float(ev)) if (np.isfinite(eu) and np.isfinite(ev)) else np.nan
            edge_rows.append(
                {
                    "edge": f"E{i:05d}",
                    "u": str(u),
                    "v": str(v),
                    "link_type": "pipe",
                    "length_raw": np.nan,
                    "length_m": np.nan,
                    "diameter_raw": np.nan,
                    "diameter_m": np.nan,
                    "roughness_raw": np.nan,
                    "abs_elev_diff_m": elev_diff_m,
                }
            )

    edge_feats = pd.DataFrame(edge_rows)
    if not edge_feats.empty:
        len_feat = _log1p_zscore_positive(edge_feats["length_m"].to_numpy(dtype=float))
        elev_feat = _log1p_zscore_positive(edge_feats["abs_elev_diff_m"].to_numpy(dtype=float))
        diam_feat = _log1p_zscore_positive(edge_feats["diameter_m"].to_numpy(dtype=float))
        rough_feat = _log1p_zscore_positive(edge_feats["roughness_raw"].to_numpy(dtype=float))
        link_type_arr = edge_feats["link_type"].astype(str).str.lower().to_numpy()
        edge_feats["len_log1p_z"] = len_feat
        edge_feats["abs_elev_diff_log1p_z"] = elev_feat
        edge_feats["diam_log1p_z"] = diam_feat
        edge_feats["rough_log1p_z"] = rough_feat
        edge_feats["is_pipe"] = (link_type_arr == "pipe").astype(np.float32)
        edge_feats["is_pump"] = (link_type_arr == "pump").astype(np.float32)
        edge_feats["is_valve"] = (link_type_arr == "valve").astype(np.float32)
        edge_feats = edge_feats.set_index("edge")
    else:
        edge_feats = pd.DataFrame(
            columns=[
                "u", "v", "link_type", "length_raw", "length_m", "diameter_raw", "diameter_m", "roughness_raw", "abs_elev_diff_m",
                "len_log1p_z", "abs_elev_diff_log1p_z", "diam_log1p_z", "rough_log1p_z", "is_pipe", "is_pump", "is_valve",
            ]
        )
        edge_feats.index.name = "edge"

    selected_ptime_s = int(
        compute_selected_ptime(
            pressure,
            node_feats,
            window_steps=SELECTED_PTIME_WINDOW_STEPS,
            prefer_high_std=True,
            verbose=False,
        )
    )

    name = os.path.splitext(os.path.basename(str(inp_path)))[0]
    node_feats_path = name + "_node_features.csv"
    edge_feats_path = name + "_edge_features.csv"
    pressure_path = name + "_pressure_timeseries.parquet"
    graph_meta_path = name + "_graph_meta.json"

    node_feats.to_csv(out / node_feats_path)
    edge_feats.to_csv(out / edge_feats_path)
    pressure.to_parquet(out / pressure_path)

    meta = {
        "inp_file": str(inp_path.resolve()),
        "n_nodes": int(len(node_ids)),
        "n_links": int(len(links_obj)),
        "n_edges_saved": int(len(edge_feats)),
        "n_junctions": int((np.array(types) == "Junction").sum()),
        "n_tanks": int((np.array(types) == "Tank").sum()),
        "n_reservoirs": int((np.array(types) == "Reservoir").sum()),
        "t_steps": int(len(pressure)),
        "report_step_s": int(report_step_s),
        "hydraulic_step_s": 3600,
        "duration_s": int(duration_s),
        "selected_ptime_s": int(selected_ptime_s),
        "selected_ptime_method": SELECTED_PTIME_METHOD,
        "selected_ptime_window_steps": int(SELECTED_PTIME_WINDOW_STEPS),

        # units/meta
        "units": units,
        "pressure_unit": pressure_unit,
        "convert_to_SI": bool(convert_to_SI),
        "unit_conversion": conv_info,
    }
    pd.Series(meta).to_json(out / graph_meta_path, force_ascii=False, indent=2)

    print(f"[INFO] Generated files: {name}_node_features.csv, {name}_pressure_timeseries.parquet, {name}_graph_meta.json")

    print(f"[INFO] edge static features file: {name}_edge_features.csv")

    return {
        "node_features": node_feats,
        "edge_features": edge_feats,
        "pressure_timeseries": pressure,
        "graph_meta": meta
    }


def compute_LDD(node_feats: pd.DataFrame) -> float:
    junc_nodes = node_feats[node_feats["type"] == "Junction"]
    base_demands = junc_nodes["base_demand"].values
    base_demands = base_demands[base_demands > 0]
    if len(base_demands) == 0:
        raise ValueError("All junction base demands are zero; LDD is undefined.")
    return float(base_demands.mean())
def relative_pressure_drop(
    P_base_t: pd.Series,
    P_leak_t: pd.Series,
    eps_p: float = 1e-10,
    min_base_p: float = 1e-6,
) -> pd.Series:
    """Compute dimensionless relative pressure drop with a clamped denominator."""
    P0 = P_base_t.astype(float).abs()
    denom = np.maximum(P0.to_numpy(), float(min_base_p)) + float(eps_p)
    rel = (P_base_t.astype(float) - P_leak_t.astype(float)).abs().to_numpy() / denom
    return pd.Series(rel, index=P_base_t.index)

def β_pick_pilot_leak_nodes(node_feats: pd.DataFrame, n_pilot_nodes=3, seed=0):
    rng = random.Random(seed)
    junc_nodes = list(node_feats[node_feats["type"] == "Junction"].index.astype(str))
    if len(junc_nodes) <= n_pilot_nodes:
        return junc_nodes
    return rng.sample(junc_nodes, n_pilot_nodes)


def _deepcopy_or_reload_epynet(wn_base):
    """Copy the simulation network, or reload the INP file if copying fails."""
    try:
        wn = copy.deepcopy(wn_base)
        return wn
    except Exception:
        inp_path = getattr(wn_base, "_inp_path", None)
        if inp_path is None:
            raise RuntimeError("deepcopy(epynet Network) failed and no wn_base._inp_path to reload.")
        wn = Network(str(inp_path))
        try:
            wn._inp_path = str(inp_path)
        except Exception:
            pass
        return wn


def β_simulate_leak(wn_base, leak_node, q_leak_inp, P_base_inp, meta):
    """Calibrate the same constant-demand leak used by H and R/S generation."""
    import wntr
    from src.hydraulics.leak_scenarios import constant_leak, run_pressure_array
    path = meta.get("inp_file", getattr(wn_base, "_inp_path", None))
    if path is None:
        raise ValueError("INP path is required for constant-leak calibration.")
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.options.time.duration = int(P_base_inp.index[-1])
    wn.options.time.report_timestep = int(meta.get("report_step_s", 3600))
    wn.options.time.hydraulic_timestep = int(meta.get("hydraulic_step_s", 3600))
    conversion = meta.get("unit_conversion", {})
    flow_m3s = float(q_leak_inp) * float(conversion.get("flow_to_LPS_factor", 1.0)) / 1000.0
    with constant_leak(wn, str(leak_node), flow_m3s):
        values, times = run_pressure_array(
            wn, list(map(str, P_base_inp.columns)), tag="beta_calibration",
            required_time_s=float(P_base_inp.index[-1]),
        )
    pressure = pd.DataFrame(values, index=times, columns=P_base_inp.columns)
    pressure /= float(conversion.get("pressure_to_m_factor", 1.0))
    pressure = pressure.reindex(index=P_base_inp.index, columns=P_base_inp.columns)
    if not np.isfinite(pressure.to_numpy()).all():
        raise ValueError("Incomplete pressure output during leakage calibration.")
    return P_base_inp - pressure



def β_evaluate_beta(wn_base, beta, LDD_inp, pilot_nodes, P_base_inp, node_feats, meta, tmp_threshold_rel=0.01,):
    """Evaluate candidate leakage magnitudes at pilot nodes."""
    stats = []

    P_junc = P_base_inp[node_feats[node_feats["type"] == "Junction"].index.astype(str)]
    min_ptime = compute_selected_ptime(
        P_base_inp,
        node_feats,
        window_steps=SELECTED_PTIME_WINDOW_STEPS,
        prefer_high_std=True,
        verbose=False,
    )
    sensor_nodes=list(P_junc.columns)
    k_flow = float(meta.get("unit_conversion", None).get("flow_to_LPS_factor", 1.0))
    for leak_node in pilot_nodes:
        q_leak_inp = float(beta) * float(LDD_inp)
        q_leak_lps = q_leak_inp * max(k_flow, 1e-12)
        warn_msgs = []
        run_error = None

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dP = β_simulate_leak(wn_base, leak_node, q_leak_inp, P_base_inp, meta)
            warn_msgs = [str(w.message) for w in caught]
        except Exception as e:
            dP = None
            run_error = repr(e)

        hyd_warn = any(("unbalanc" in m.lower()) or ("unstable" in m.lower()) for m in warn_msgs)
        if dP is None:
            stats.append({
                "leak_node": str(leak_node),
                "q_leak_lps": float(q_leak_lps),
                "p95_dp": float("nan"),
                "max_dp": float("nan"),
                "det_rate": float("nan"),
                "valid_case": 0,
                "hyd_warn": int(hyd_warn),
                "warn_count": int(len(warn_msgs)),
                "run_error": run_error,
            })
            continue

        #quick_diagnose_one_pilot_case(P_base, dP, node_feats, min_ptime, threshold_rel=0.01)

        P0 = P_base_inp.loc[min_ptime, sensor_nodes]
        Pleak = (P_base_inp.loc[min_ptime, sensor_nodes] - dP.loc[min_ptime, sensor_nodes])
        rel = relative_pressure_drop(P0, Pleak)
        rel_arr = np.asarray(rel, dtype=float)
        rel_arr = rel_arr[np.isfinite(rel_arr)]

        if rel_arr.size == 0:
            p95_rel_dp = float("nan")
            max_rel_dp = float("nan")
            det_rate = float("nan")
            valid_case = 0
        else:
            p95_rel_dp = float(np.percentile(rel_arr, 95))
            max_rel_dp = float(np.max(rel_arr))
            det_rate = float((rel_arr >= float(tmp_threshold_rel)).mean())
            valid_case = 1

        stats.append({
            "leak_node": str(leak_node),
            "q_leak_lps": float(q_leak_lps),
            "p95_dp": p95_rel_dp,
            "max_dp": max_rel_dp,
            "det_rate": det_rate,
            "valid_case": int(valid_case),
            "hyd_warn": int(hyd_warn),
            "warn_count": int(len(warn_msgs)),
            "run_error": run_error,
        })

    return pd.DataFrame(stats)


def β_auto_pick_beta(
    wn_base,
    beta_candidates,
    LDD_inp,
    node_feats,
    P_base_inp,
    meta,
    n_pilot_nodes=3,
    dp_low=0.001,
    dp_high=0.05,
    det_low=0.2,
    det_high=0.7,
    max_invalid_rate=0.4,
    max_hyd_warn_rate=0.6,
):
    """Select a reference leakage magnitude using pilot simulations."""
    pilot_nodes = β_pick_pilot_leak_nodes(node_feats, n_pilot_nodes=n_pilot_nodes)

    records = []
    for beta in beta_candidates:
        df_stats = β_evaluate_beta(wn_base, beta, LDD_inp, pilot_nodes, P_base_inp, node_feats, meta, tmp_threshold_rel=0.001)
        valid_mask = (df_stats["valid_case"] == 1) if len(df_stats) > 0 else pd.Series(dtype=bool)
        valid_rate = float(valid_mask.mean()) if len(df_stats) > 0 else 0.0
        invalid_rate = 1.0 - valid_rate
        hyd_warn_rate = float(df_stats["hyd_warn"].mean()) if len(df_stats) > 0 else 0.0

        if len(df_stats) > 0 and valid_mask.any():
            med_p95_dp = float(df_stats.loc[valid_mask, "p95_dp"].median())
            med_max_dp = float(df_stats.loc[valid_mask, "max_dp"].median())
            med_det_rate = float(df_stats.loc[valid_mask, "det_rate"].median())
        else:
            med_p95_dp = float("nan")
            med_max_dp = float("nan")
            med_det_rate = float("nan")

        records.append({
            "beta": beta,
            "med_p95_dp": med_p95_dp,
            "med_max_dp": med_max_dp,
            "med_det_rate": med_det_rate,
            "valid_rate": valid_rate,
            "invalid_rate": invalid_rate,
            "hyd_warn_rate": hyd_warn_rate,
        })

    def _range_penalty(v, low, high):
        if not np.isfinite(v):
            return 1e6
        scale = max(float(high) - float(low), 1e-12)
        if v < low:
            return float(low - v) / scale
        if v > high:
            return float(v - high) / scale
        return 0.0

    def _score_near_window(r):
        dp_pen = _range_penalty(float(r["med_p95_dp"]), float(dp_low), float(dp_high))
        det_pen = _range_penalty(float(r["med_det_rate"]), float(det_low), float(det_high))
        invalid_pen = max(0.0, float(r["invalid_rate"]) - float(max_invalid_rate)) / max(float(max_invalid_rate), 1e-12)
        hyd_warn_pen = max(0.0, float(r["hyd_warn_rate"]) - float(max_hyd_warn_rate)) / max(float(max_hyd_warn_rate), 1e-12)
        # hyd_warn is a softer constraint than invalid_rate
        return dp_pen + det_pen + 2.0 * invalid_pen + 0.35 * hyd_warn_pen

    good = [
        r for r in records
        if np.isfinite(r["med_p95_dp"])
        and np.isfinite(r["med_det_rate"])
        and (r["invalid_rate"] <= float(max_invalid_rate))
        and (r["hyd_warn_rate"] <= float(max_hyd_warn_rate))
        and (float(dp_low) <= r["med_p95_dp"] <= float(dp_high))
        and (float(det_low) <= r["med_det_rate"] <= float(det_high))
    ]

    if good:
        best = sorted(good, key=lambda r: r["beta"])[0]
        status = "ok"
        message = (
            f"Hit fixed windows: p95_dp in [{dp_low},{dp_high}], "
            f"det_rate in [{det_low},{det_high}], "
            f"hyd_warn_rate <= {max_hyd_warn_rate}"
        )
    else:
        usable = [
            r for r in records
            if np.isfinite(r["med_p95_dp"]) and np.isfinite(r["med_det_rate"]) and r["valid_rate"] > 0
        ]
        if usable:
            # fallback-1: nearest to fixed windows; tie-break by smaller beta (conservative)
            best = sorted(usable, key=lambda r: (_score_near_window(r), r["beta"]))[0]
            if float(best["hyd_warn_rate"]) > float(max_hyd_warn_rate):
                status = "fallback_hyd_warn"
                message = (
                    "Selected candidate has high hydraulic warning ratio; "
                    "using conservative fallback. Please inspect network hydraulics before formal experiments."
                )
            else:
                status = "0fallback_nearest"
                message = (
                    "No candidate hit the fixed windows; selected nearest stable candidate "
                    "with conservative tie-break (smaller beta). Please use leak intensity carefully."
                )
        else:
            # fallback-2: all pilot runs invalid/unstable -> choose minimum beta
            best = sorted(records, key=lambda r: r["beta"])[0]
            status = "0fallback_min_beta"
            message = (
                "Pilot simulations are mostly invalid or unstable; using minimum beta as conservative fallback. "
                "Please inspect network hydraulics and scenario settings before formal experiments."
            )

    best = dict(best)
    best["select_status"] = status
    best["select_message"] = message
    best["thresholds"] = {
        "dp_low": float(dp_low),
        "dp_high": float(dp_high),
        "det_low": float(det_low),
        "det_high": float(det_high),
        "max_invalid_rate": float(max_invalid_rate),
        "max_hyd_warn_rate": float(max_hyd_warn_rate),
    }
    best["n_pilot_nodes"] = int(len(pilot_nodes))

    return best, records


def get_base_features_fromInp(inp_path: str, out_dir: str | None = None):
    """Generate graph features and reference leakage calibration for an input network."""
    if out_dir is None:
        out_dir = str((_P.Path(inp_path).resolve().parent / "node_features"))
    step1_result = run_inp_to_features(inp_path, out_dir)


    wn_base = Network(str(inp_path))
    try:
        wn_base._inp_path = str(inp_path)
    except Exception:
        pass

    node_feats = step1_result["node_features"]
    P_base = step1_result["pressure_timeseries"]
    meta = step1_result["graph_meta"]
    units = meta.get("units", "UNKNOWN")
    k_flow = float(meta.get("unit_conversion",None).get("flow_to_LPS_factor", 1.0))
    k_p = float(meta.get("unit_conversion",None).get("pressure_to_m_factor", 1.0))
    LDD_LPS = compute_LDD(node_feats)
    LDD_inp = LDD_LPS / max(k_flow, 1e-12)
    P_base_inp = P_base / max(k_p, 1e-12)
    selected_ptime_s = meta.get("selected_ptime_s", None)
    if selected_ptime_s is None:
        selected_ptime_s = int(
            compute_selected_ptime(
                P_base,
                node_feats,
                window_steps=SELECTED_PTIME_WINDOW_STEPS,
                prefer_high_std=True,
                verbose=False,
            )
        )
        meta["selected_ptime_s"] = int(selected_ptime_s)
    selected_ptime_s = int(selected_ptime_s)

    beta_candidates = [0.05, 0.1, 0.5, 1, 2, 5,10,20]
    dp_low, dp_high = 0.001, 0.05
    det_low, det_high = 0.2, 0.7
    max_invalid_rate = 0.4
    # Soft hydraulic warning constraint: allow up to 60% warned pilot cases.
    max_hyd_warn_rate = 0.6

    try:
        best_beta_info, all_beta_info = β_auto_pick_beta(
            wn_base, beta_candidates, LDD_inp, node_feats, P_base_inp, meta,
            n_pilot_nodes=10,
            dp_low=dp_low,
            dp_high=dp_high,
            det_low=det_low,
            det_high=det_high,
            max_invalid_rate=max_invalid_rate,
            max_hyd_warn_rate=max_hyd_warn_rate,
        )
    finally:
        _safe_close_network(wn_base)

    best_beta = best_beta_info["beta"]
    q0_LPS = float(best_beta) * float(LDD_LPS)
    q0_inp = float(best_beta) * float(LDD_inp)
    beta_select_status = best_beta_info.get("select_status", "ok")
    beta_select_message = best_beta_info.get("select_message", "")

    q0_path = os.path.splitext(os.path.basename(inp_path))[0] + "q0.json"
    pd.Series({
        "LDD_lps": LDD_LPS,  # backward-compatible key, unit=LPS
        "LDD_inp": LDD_inp,
        "chosen_beta": best_beta,
        "beta_select_status": beta_select_status,
        "beta_select_message": beta_select_message,
        "beta_thresholds": best_beta_info.get("thresholds", {}),
        "beta_info": all_beta_info,
        "q0_LPS": q0_LPS,
        "q0_inp": q0_inp,
        "units": units,
        "flow_unit_out": "LPS",
        "pressure_unit_out": "mH2O",
        "selected_ptime_s": int(selected_ptime_s),
        "selected_ptime_method": SELECTED_PTIME_METHOD,
        "selected_ptime_window_steps": int(SELECTED_PTIME_WINDOW_STEPS),
    }).to_json(_P.Path(out_dir) / q0_path, force_ascii=False, indent=2)

    print("LDD_LPS (mean base_demand over junctions) =", LDD_LPS)
    print("Chosen beta =", best_beta, " => q0_LPS =", q0_LPS)
    if beta_select_status != "ok":
        print("[WARN] beta auto-pick fallback:", beta_select_message)

    return {
        "node_features": node_feats,
        "P_base_inp": P_base_inp,
        "P_base": P_base,
        "graph_meta": meta,
        "q0_LPS": q0_LPS,
        "q0_inp": q0_inp,
    }
