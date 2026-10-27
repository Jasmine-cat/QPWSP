"""Optional nodal-demand preference retained from the historical implementation.

The adopted experiment profiles leave its coefficient at zero.
"""
import numpy as np
from typing import Tuple

def normalize_demand_clip(
    d_junction: np.ndarray,
    *,
    clip_q: float = 0.95,
    eps: float = 1e-12,
    use_positive_for_clip: bool = True,
    use_positive_for_scale: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    对 junction 空间的 demand 做：长尾裁剪 + [0,1] 归一化（维度不变）。
    Parameters
    ----------
    d_junction : np.ndarray
        (N_j,) junction 节点的 base_demand（已经对齐索引空间）
    clip_q : float
        分位数裁剪阈值，比如 0.95 表示把大于 95% 分位的值截断到该分位值
    eps : float
        防止除零
    use_positive_for_clip : bool
        裁剪分位数是否只在 d>0 的子集上计算（避免大量 0 干扰）
    use_positive_for_scale : bool
        归一化 max 是否只在 d>0 的子集上计算（更稳健）

    Returns
    -------
    d_norm : np.ndarray
        (N_j,) 归一化后的 demand（范围约 [0,1]，维度不变）
    clip_value : float
        实际使用的裁剪阈值（便于日志记录）
    """
    d = np.asarray(d_junction, dtype=float).reshape(-1)  # (N_j,)
    if d.size == 0:
        return d.copy(), 0.0
    # 1) 选用于计算分位数的集合
    if use_positive_for_clip:
        base = d[d > 0]
    else:
        base = d
    if base.size == 0:
        # 全是 0 或非正：直接返回全 0
        return np.zeros_like(d), 0.0

    # 2) 分位数裁剪（不改变长度）
    clip_value = float(np.quantile(base, clip_q))
    d_clip = np.clip(d, 0.0, clip_value)

    # 3) 归一化（不改变长度）
    if use_positive_for_scale:
        scale_base = d_clip[d_clip > 0]
        mx = float(np.max(scale_base)) if scale_base.size > 0 else 0.0
    else:
        mx = float(np.max(d_clip))

    if mx <= eps:
        d_norm = np.zeros_like(d_clip)
    else:
        d_norm = d_clip / (mx + eps)

    return d_norm.astype(float), clip_value


def build_Q_size_demand(
    N: int,
    w: np.ndarray, #节点需水量（决定偏好）
    lambda_dem: float = 1.0,
    *,
    lambda_norm: str = "N",       # "none" | "N"#归一化开关
    clip_w: bool = True,
    return_const: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    线性“成本/偏好”项（对角Q）：
        E_size_dem(x) = lambda_eff * sum_i (1 - w_i) * x_i
    其中 w_i in [0,1] 表示“需求高低”（越大需求越高，(1-w) 越小越偏好被选）

    归一化:
      - lambda_norm="N": lambda_eff = lambda_dem / N   => 变成“平均意义”尺度，跨图更稳
      - "none": lambda_eff = lambda_dem

    注意：
      - 这是纯线性项，没有额外常数项（const=0）
      - 若你要把它和 Q_budget/Q_cov/Q_det 合并，直接 Q_total += Q_size_dem 即可
    """
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != N:
        raise ValueError(f"w.shape[0]={w.shape[0]} != N={N}")

    if clip_w:
        w = np.clip(w, 0.0, 1.0)

    lambda_dem = float(lambda_dem)

    if lambda_norm == "none":
        lambda_eff = lambda_dem
    elif lambda_norm.upper() == "N":
        lambda_eff = lambda_dem / float(N)
    else:
        raise ValueError(f"Unknown lambda_norm='{lambda_norm}', choose from: none, N")

    c = lambda_eff * (1.0 - w)  # (N,)

    Q = np.zeros((N, N), dtype=float)
    Q[np.diag_indices(N)] = c

    if return_const:
        return Q, 0.0
    else:
        return Q, 0.0
