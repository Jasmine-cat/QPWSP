"""Response statistics used by the H-response QUBO."""
import numpy as np
from typing import Optional, Tuple

def _column_l2_normalize(H: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    col_norm = np.linalg.norm(H, axis=0, keepdims=True)
    return np.divide(H, col_norm + float(eps), out=np.zeros_like(H), where=col_norm > float(eps))


def _randomized_svd_for_response(
    A: np.ndarray,
    rank: int,
    *,
    oversample: int = 8,
    n_iter: int = 2,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"[_randomized_svd_for_response] A must be 2D, got shape={A.shape}")
    m, n = A.shape
    max_rank = min(m, n)
    r = max(1, min(int(rank), max_rank))
    q = max(r, min(max_rank, r + max(int(oversample), 0)))

    rng = np.random.default_rng(int(random_state))
    omega = rng.standard_normal(size=(n, q))
    Y = A @ omega
    for _ in range(max(int(n_iter), 0)):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ A
    U_hat, s, _ = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    return U[:, :r], s[:r]


def _leak_pair_redundancy_from_H(
    H: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    基于列归一化后的 H 构造任务区分谱统计量。

    Returns:
      Red_g: cosine redundancy between node contribution profiles g_i.
    """
    Hn = _column_l2_normalize(H, eps=eps)
    N, P = Hn.shape
    iu, ju = np.triu_indices(P, k=1)
    if iu.size == 0:
        return np.zeros((N, N), dtype=float)

    max_chunk_entries = 2_000_000
    chunk_size = max(1, min(iu.size, max(16, max_chunk_entries // max(N, 1))))
    pair_d2 = np.empty(iu.size, dtype=float)
    for start in range(0, iu.size, chunk_size):
        end = min(start + chunk_size, iu.size)
        diff = Hn[:, iu[start:end]] - Hn[:, ju[start:end]]
        pair_d2[start:end] = np.sum(diff * diff, axis=0)

    eps = max(float(eps), np.finfo(float).eps)
    positive_d2 = pair_d2[pair_d2 > 0.0]
    eps_d2 = max(float(np.median(positive_d2)) if positive_d2.size else 1.0, 1.0) * eps
    w_raw = 1.0 / (pair_d2 + eps_d2)
    positive_w = w_raw[w_raw > 0.0]
    w_ref = float(np.median(positive_w)) if positive_w.size else 1.0
    w_scale = max(w_ref, eps)

    norm_sq = np.zeros(N, dtype=float)
    gram = np.zeros((N, N), dtype=float)
    for start in range(0, iu.size, chunk_size):
        end = min(start + chunk_size, iu.size)
        diff = Hn[:, iu[start:end]] - Hn[:, ju[start:end]]
        g = (diff * diff) * (w_raw[start:end] / w_scale)[None, :]
        norm_sq += np.sum(g * g, axis=1)
        gram += g @ g.T

    denom = np.sqrt(np.maximum(norm_sq, 0.0))
    denom = np.outer(denom, denom)
    Red = np.divide(gram, np.maximum(denom, eps), out=np.zeros_like(gram), where=denom > eps)
    Red = np.clip(Red, 0.0, 1.0)
    Red = np.maximum(0.0, 0.5 * (Red + Red.T))
    np.fill_diagonal(Red, 0.0)
    return Red


def _leak_leverage_importance_from_H(
    H: np.ndarray,
    *,
    pca_rank: Optional[int] = None,
    pca_energy: float = 0.95,
    max_auto_rank: int = 32,
    randomized_threshold: int = 512,
    randomized_oversample: int = 8,
    randomized_n_iter: int = 2,
    randomized_seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    Hn = _column_l2_normalize(H, eps=eps)
    N, P = Hn.shape
    if N <= 0 or P <= 0:
        return np.zeros(N, dtype=float)

    max_rank = min(Hn.shape)
    if max_rank <= 0:
        return np.zeros(N, dtype=float)

    eps = max(float(eps), np.finfo(float).eps)
    U_r = None
    s_r = None
    if pca_rank is None:
        rank_cap = max(1, min(int(max_auto_rank), max_rank))
        if max_rank <= int(randomized_threshold):
            energy = float(pca_energy)
            if not (0.0 < energy <= 1.0):
                raise ValueError(f"[_leak_leverage_importance_from_H] pca_energy must be in (0,1], got {pca_energy}")
            U, s, _ = np.linalg.svd(Hn, full_matrices=False)
            s2 = s * s
            if float(np.sum(s2)) <= eps:
                return np.zeros(N, dtype=float)
            cum_ratio = np.cumsum(s2) / (float(np.sum(s2)) + eps)
            energy_rank = int(np.searchsorted(cum_ratio, energy, side="left") + 1)
            r = max(1, min(energy_rank, rank_cap))
            U_r = U[:, :r]
            s_r = s[:r]
        else:
            r = rank_cap
            U_r, s_r = _randomized_svd_for_response(
                Hn,
                rank=r,
                oversample=randomized_oversample,
                n_iter=randomized_n_iter,
                random_state=randomized_seed,
            )
    else:
        r = max(1, min(int(pca_rank), max_rank))
        if max_rank > int(randomized_threshold):
            U_r, s_r = _randomized_svd_for_response(
                Hn,
                rank=r,
                oversample=randomized_oversample,
                n_iter=randomized_n_iter,
                random_state=randomized_seed,
            )
        else:
            U, s, _ = np.linalg.svd(Hn, full_matrices=False)
            U_r = U[:, :r]
            s_r = s[:r]

    if U_r is None or s_r is None or float(np.sum(s_r * s_r)) <= eps:
        return np.zeros(N, dtype=float)
    leverage = np.sum(U_r * U_r, axis=1)
    I_leverage = safe_minmax_1d(leverage)

    # Residual response energy highlights nodes that are informative in modes
    # 残差项用于补充主导 PCA/leverage 子空间未覆盖的小模式。
    H_main = U_r @ (U_r.T @ Hn)
    H_res = Hn - H_main
    residual_energy = np.sum(H_res * H_res, axis=1)
    I_res = safe_minmax_1d(residual_energy)

    return 0.7 * I_leverage + 0.3 * I_res


def compute_leak_importance_and_redundancy_from_H(
    H: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract H-space background variables for QUBO construction.

    Returns:
      I: fixed leverage-residual mixed node importance, normalized to [0,1].
      Red_H: task-profile redundancy between nodes, based on cosine(g_i, g_j).

    The importance is fixed to 0.7 PCA/leverage plus 0.3 residual-response
    energy. Redundancy remains the cosine similarity of leak-pair
    discrimination profiles.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError(f"[compute_leak_importance_and_redundancy_from_H] H must be 2D, got shape={H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("[compute_leak_importance_and_redundancy_from_H] H contains non-finite values")
    Red_g = _leak_pair_redundancy_from_H(H, eps=eps)
    return _leak_leverage_importance_from_H(H, eps=eps), Red_g


def safe_minmax_1d(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    '''minmax归一化'''
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return x.copy()
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    x_work = x.copy()
    fill_val = float(np.nanmin(x_work[finite]))
    x_work[~finite] = fill_val
    xmin = float(np.min(x_work))
    xmax = float(np.max(x_work))
    scale = max(xmax - xmin, eps)
    return (x_work - xmin) / scale
