"""PIGNN训练与分阶段Top-k STE。"""

from __future__ import annotations

import random
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.gnn import LightPhysicsGNN, SimpleGCN
from src.qubo.objective import compute_qubo_component_stats, extract_qubo_total, print_qubo_component_stats


def qubo_loss_subset(
    p_all: torch.Tensor,
    qubo_const: torch.Tensor,
    Q: torch.Tensor,
    junction_idx: torch.Tensor,
) -> torch.Tensor:
    """计算 junction 子集上的 relaxed 或 hard QUBO 完整能量。"""
    p = p_all[junction_idx]
    return torch.dot(p, Q @ p) + qubo_const


def _ema(values: List[float], window: int) -> float:
    if not values:
        return float("inf")
    return float(np.mean(values[-min(int(window), len(values)) :]))


def _hard_topk(prob: torch.Tensor, k: int) -> torch.Tensor:
    """生成无梯度的 exact-K 二值向量。"""
    n, kk = int(prob.numel()), int(k)
    if kk <= 0:
        return torch.zeros_like(prob)
    if kk >= n:
        return torch.ones_like(prob)
    result = torch.zeros_like(prob)
    result.scatter_(0, torch.topk(prob, k=kk, largest=True, sorted=False).indices, 1.0)
    return result


def _ste_topk(prob: torch.Tensor, k: int) -> torch.Tensor:
    """前向 exact-K、反向恒等近似的 Top-k STE。"""
    hard = _hard_topk(prob, k)
    return hard + (prob - prob.detach())




@dataclass
class TrainConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    use_physics_injection: bool = True
    use_physics_gate: bool = True
    lr: float = 1e-4
    max_epochs: int = 20000
    print_every: int = 200
    min_epochs: int = 600
    ema_window: int = 50
    rel_tol: float = 1e-3
    plateau_patience: int = 40
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    use_stagewise_ste: bool = True
    ste_stage1_ratio: float = 0.10
    k_budget_hint: int = -1
    ebin_patience: int = 120
    ebin_min_delta: float = 1e-8
    ebin_min_evals_stage2: int = 80


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机状态。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _prepare_physics_tensors(
    graph_aux_np: Optional[Dict[str, Any]], n_all: int, device: torch.device
) -> Tuple[bool, str, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """校验并转化边物理编码、端点索引和关联矩阵。"""
    if not isinstance(graph_aux_np, dict):
        return False, "graph_aux_np is missing", None, None, None
    missing = [key for key in ("edge_attr", "edge_index", "B_incidence") if key not in graph_aux_np]
    if missing:
        return False, f"missing keys: {missing}", None, None, None
    try:
        edge_attr_np = np.asarray(graph_aux_np["edge_attr"], dtype=np.float32)
        edge_index_np = np.asarray(graph_aux_np["edge_index"], dtype=np.int64)
        incidence_np = np.asarray(graph_aux_np["B_incidence"], dtype=np.float32)
    except Exception as exc:
        return False, repr(exc), None, None, None

    if edge_attr_np.ndim != 2 or edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        return False, "invalid edge_attr or edge_index shape", None, None, None
    if incidence_np.ndim != 2 or incidence_np.shape[0] != int(n_all):
        return False, "invalid B_incidence shape", None, None, None
    if not (edge_attr_np.shape[0] == edge_index_np.shape[1] == incidence_np.shape[1]):
        return False, "edge count mismatch", None, None, None
    if edge_index_np.size and (edge_index_np.min() < 0 or edge_index_np.max() >= int(n_all)):
        return False, "edge_index is out of range", None, None, None
    return (
        True,
        "ok",
        torch.tensor(edge_attr_np, dtype=torch.float32, device=device),
        torch.tensor(edge_index_np, dtype=torch.long, device=device),
        torch.tensor(incidence_np, dtype=torch.float32, device=device),
    )




def _model_forward(
    model: torch.nn.Module,
    use_physics: bool,
    X: torch.Tensor,
    A_norm: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    edge_attr: Optional[torch.Tensor],
    incidence: Optional[torch.Tensor],
) -> torch.Tensor:
    if use_physics:
        return model(X, A_norm, edge_index, edge_attr, incidence)
    return model(X, A_norm)


def train_single_run(
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Dict[str, Any],
    X_np: np.ndarray,
    A_norm_np: np.ndarray,
    junction_idx_np: np.ndarray,
    config: TrainConfig,
    graph_aux_np: Optional[Dict[str, Any]] = None,
    *,
    min_epochs: Optional[int] = None,
    ema_window: Optional[int] = None,
    rel_tol: Optional[float] = None,
    plateau_patience: Optional[int] = None,
    lr_factor: Optional[float] = None,
    min_lr: Optional[float] = None,
    grad_clip_norm: Optional[float] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """执行一次 PIGNN/GCN 训练，stage2 固定采用 exact-K Top-k STE。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_all, feature_dim = X_np.shape
    Q_np, const_total = extract_qubo_total(Q_dist_np, const_dist)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    A_norm = torch.tensor(A_norm_np, dtype=torch.float32, device=device)
    Q = torch.tensor(Q_np, dtype=torch.float32, device=device)
    c_total = torch.tensor(float(const_total), dtype=torch.float32, device=device)
    junction_idx = torch.tensor(junction_idx_np, dtype=torch.long, device=device)

    use_physics = bool(config.use_physics_injection)
    edge_attr = edge_index = incidence = None
    if use_physics:
        ok, reason, edge_attr, edge_index, incidence = _prepare_physics_tensors(graph_aux_np, n_all, device)
        if not ok:
            raise ValueError(f"physics injection requested but unavailable: {reason}")
        model: torch.nn.Module = LightPhysicsGNN(
            in_dim=feature_dim,
            edge_dim=int(edge_attr.shape[1]),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
            use_physics_gate=bool(config.use_physics_gate),
        ).to(device)
        print(
            f"[train] model=LightPhysicsGNN (physics enabled, E={edge_attr.shape[0]}, "
            f"edge_dim={edge_attr.shape[1]}, physics_gate={bool(config.use_physics_gate)})"
        )
    else:
        model = SimpleGCN(
            in_dim=feature_dim,
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
        ).to(device)
        print("[train] model=SimpleGCN (edge physics disabled)")

    min_epochs = int(config.min_epochs if min_epochs is None else min_epochs)
    ema_window = int(config.ema_window if ema_window is None else ema_window)
    rel_tol = float(config.rel_tol if rel_tol is None else rel_tol)
    plateau_patience = int(config.plateau_patience if plateau_patience is None else plateau_patience)
    lr_factor = float(config.lr_factor if lr_factor is None else lr_factor)
    min_lr = float(config.min_lr if min_lr is None else min_lr)
    grad_clip_norm = float(config.grad_clip_norm if grad_clip_norm is None else grad_clip_norm)
    k_budget = int(config.k_budget_hint)
    if k_budget <= 0 or k_budget > len(junction_idx_np):
        raise ValueError("k_budget_hint must be within the junction subset")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=plateau_patience,
        threshold=0.0, cooldown=0, min_lr=min_lr,
    )
    use_stagewise = bool(config.use_stagewise_ste)
    stage2_epoch = int(max(1, np.floor(float(config.max_epochs) * np.clip(config.ste_stage1_ratio, 0.0, 1.0))))
    stage2_epoch_runtime = stage2_epoch
    if use_stagewise:
        print(
            f"[train] stagewise-STE enabled: start_epoch={stage2_epoch_runtime}, "
            f"use_topk=True, k_hint={k_budget}, selection=stage2-hard-metric, js_mode=identity"
        )

    model.eval()
    with torch.no_grad():
        p_init = _model_forward(model, use_physics, X, A_norm, edge_index, edge_attr, incidence)
    print_qubo_component_stats(
        "[Init-Full][A0-soft]",
        compute_qubo_component_stats(p_init[junction_idx].cpu().numpy(), Q_dist_np, const_dist),
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_ebin = float("inf")
    loss_history: List[float] = []
    last_ema = float("inf")
    stage2_entered = False
    ebin_no_improve = 0
    ebin_eval_count = 0

    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        p_soft = _model_forward(model, use_physics, X, A_norm, edge_index, edge_attr, incidence)
        p_forward = p_soft

        if use_stagewise and epoch >= stage2_epoch_runtime:
            if not stage2_entered:
                p_stage1 = p_soft[junction_idx].detach().cpu().numpy()
                print(f"[train] entering stage2-STE at epoch={epoch}")
                print_qubo_component_stats(
                    "[Stage1-Soft-Full]", compute_qubo_component_stats(p_stage1, Q_dist_np, const_dist)
                )
                sum_p = float(np.sum(p_stage1))
                print(
                    "[Stage1-budget-mass] "
                    f"sum_p={sum_p:.6g}, sum_p/K={sum_p / float(k_budget):.6g}, mean_p={np.mean(p_stage1):.6g}"
                )
                stage2_entered = True

            p_forward = p_soft.clone()
            p_j_soft = p_soft[junction_idx]
            p_forward[junction_idx] = _ste_topk(p_j_soft, k_budget)

        loss = qubo_loss_subset(p_forward, c_total, Q, junction_idx)
        loss_value = float(loss.item())
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step(loss_value)
        loss_history.append(loss_value)

        if use_stagewise:
            if epoch < stage2_epoch_runtime:
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if epoch >= min_epochs:
                    current_ema = _ema(loss_history, ema_window)
                    improvement = (
                        (last_ema - current_ema) / (abs(last_ema) + 1e-12)
                        if np.isfinite(last_ema) else float("inf")
                    )
                    last_ema = current_ema
                    if optimizer.param_groups[0]["lr"] <= min_lr + 1e-12 and improvement < rel_tol:
                        stage2_epoch_runtime = epoch + 1
                        print(
                            "[train] stage1 converged by EMA/LR criterion; "
                            f"advance stage2 start to epoch={stage2_epoch_runtime}."
                        )
            else:
                ebin_now = loss_value
                ebin_eval_count += 1
                if ebin_now < best_ebin - float(config.ebin_min_delta):
                    best_ebin = ebin_now
                    best_loss = ebin_now
                    ebin_no_improve = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    ebin_no_improve += 1
                if (
                    ebin_eval_count >= int(config.ebin_min_evals_stage2)
                    and ebin_no_improve >= int(config.ebin_patience)
                ):
                    break
        else:
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch >= min_epochs:
                current_ema = _ema(loss_history, ema_window)
                improvement = (
                    (last_ema - current_ema) / (abs(last_ema) + 1e-12)
                    if np.isfinite(last_ema) else float("inf")
                )
                last_ema = current_ema
                if optimizer.param_groups[0]["lr"] <= min_lr + 1e-12 and improvement < rel_tol:
                    break

        if epoch % int(config.print_every) == 0:
            print(f"[train] epoch={epoch:5d}  loss={loss_value:.6f}  best={best_loss:.6f}")

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_loss = float(loss_history[-1]) if loss_history else float("nan")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_best = _model_forward(model, use_physics, X, A_norm, edge_index, edge_attr, incidence).cpu().numpy()

    return p_best, float(best_loss), loss_history


def train_multi_start(
    Q_dist_np: Dict[str, np.ndarray],
    const_dist: Dict[str, Any],
    X_np: np.ndarray,
    A_norm_np: np.ndarray,
    junction_idx_np: np.ndarray,
    config: TrainConfig,
    graph_aux_np: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    deterministic_torch: bool = False,
    num_restarts: int = 1,
) -> Tuple[np.ndarray, float, List[List[float]]]:
    """多重启训练，并按 stage2 true Ebin 或 soft loss 选择最佳重启。"""
    best_p: Optional[np.ndarray] = None
    best_loss = float("inf")
    histories: List[List[float]] = []
    for restart in range(max(1, int(num_restarts))):
        run_seed = int(seed) + restart
        set_global_seed(run_seed, deterministic_torch=deterministic_torch)
        print(f"\n[train_multi_start] Restart {restart + 1}/{num_restarts} (seed={run_seed})")
        p_run, loss_run, history = train_single_run(
            Q_dist_np, const_dist, X_np, A_norm_np, junction_idx_np, config, graph_aux_np=graph_aux_np
        )
        histories.append(history)
        print(f"[train_multi_start] Restart {restart + 1} best loss = {loss_run:.6f}")
        if loss_run < best_loss:
            best_p, best_loss = p_run, float(loss_run)
    if best_p is None:
        raise RuntimeError("training produced no restart result")
    return best_p, best_loss, histories


def project_to_binary(p: np.ndarray, K_budget: int) -> np.ndarray:
    """按 soft 分数排序得到最终 exact-K 二值部署。"""
    p = np.asarray(p, dtype=float).reshape(-1)
    K = int(K_budget)
    if K < 0 or K > p.size:
        raise ValueError(f"invalid K_budget={K} for vector length={p.size}")
    x = np.zeros(p.size, dtype=int)
    x[np.argsort(-p)[:K]] = 1
    return x
