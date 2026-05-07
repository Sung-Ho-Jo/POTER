"""
POTER – OT Reweighting Core
============================
Computes per-sample importance weights from class-conditioned optimal transport.

Two public functions
--------------------
compute_ot_potentials  : run Sinkhorn per class, return f* array (shape N_train)
compute_sample_weights : convert f* to training weights (label-balanced, clipped)
"""

import warnings
import numpy as np
import torch


def compute_ot_potentials(
    X_train,
    y_train,
    X_val,
    y_val,
    reg_scale: float = 0.03,
    device: str = None,
    max_iter: int = 300,
    stop_thr: float = 1e-6,
) -> np.ndarray:
    """Compute OT dual potentials f* via class-conditioned entropic OT.

    For each class c, solves the regularized OT problem between empirical
    feature distributions P_c (train) and Q_c (val) using Sinkhorn iterations
    in log-domain.  The source dual potential f* is then median-MAD normalized
    across classes.

    Args:
        X_train   : (N, d) training features
        y_train   : (N,)   training labels
        X_val     : (M, d) validation features
        y_val     : (M,)   validation labels
        reg_scale : epsilon = reg_scale * mean(C_c) where C_c is the per-class cost matrix
        device    : torch device string; default 'cuda' if available else 'cpu'
        max_iter  : maximum Sinkhorn iterations
        stop_thr  : convergence threshold on log-u change

    Returns:
        f_star : (N,) float64 array of normalized dual potentials
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val   = np.asarray(X_val,   dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)
    y_val   = np.asarray(y_val).reshape(-1)

    n_train = len(X_train)
    f_star  = np.zeros(n_train, dtype=np.float64)
    raw_by_class: dict = {}

    for cls in np.unique(y_train):
        idx_t = np.where(y_train == cls)[0]
        idx_v = np.where(y_val   == cls)[0]

        if len(idx_t) == 0:
            continue
        if len(idx_v) == 0:
            warnings.warn(
                f"[POTER] Class {cls} has no validation samples; f* set to 0 for this class."
            )
            continue

        Xt = torch.as_tensor(X_train[idx_t], dtype=torch.float32, device=device)
        Xv = torch.as_tensor(X_val[idx_v],   dtype=torch.float32, device=device)

        C   = torch.cdist(Xt, Xv, p=2).pow(2)   # squared Euclidean cost
        reg = float(C.mean().item() * reg_scale)

        n, m = Xt.shape[0], Xv.shape[0]
        tiny = torch.finfo(torch.float32).tiny
        eps  = 1e-12   # denominator stabiliser
        K = torch.exp(-C / reg)
        a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
        b = torch.full((m,), 1.0 / m, dtype=torch.float32, device=device)
        u = torch.ones(n, dtype=torch.float32, device=device)
        v = torch.ones(m, dtype=torch.float32, device=device)

        for it in range(max_iter):
            u_prev = u
            Kv  = K.mv(v) + eps
            u   = a / Kv
            KTu = K.t().mv(u) + eps
            v   = b / KTu
            if it % 20 == 0 and (u - u_prev).abs().max().item() < stop_thr:
                break

        f_cls = (reg * torch.log(torch.clamp(u, min=tiny))).cpu().numpy().astype(np.float64)
        raw_by_class[int(cls)] = (idx_t, f_cls)

    if not raw_by_class:
        return f_star

    # Per-class median-MAD normalization: each class is centered and scaled
    # independently so within-class rank is preserved and inter-class scales
    # are aligned.
    for cls, (idx_t, f_cls) in raw_by_class.items():
        center = np.median(f_cls)
        scale  = np.median(np.abs(f_cls - center))
        if scale < 1e-12:
            scale = 1.0
        f_star[idx_t] = (f_cls - center) / scale

    # Global shift so minimum is 0.
    f_star -= f_star.min()

    return f_star


def compute_sample_weights(
    f_star: np.ndarray,
    y_train: np.ndarray,
    f_power: float = 0.5,
    clip_cap: float = 10.0,
) -> np.ndarray:
    """Convert OT dual potentials f* to normalized sample importance weights.

    Samples with high f* (high transport cost → likely spurious/noisy) receive
    low weight; bias-conflicting samples with low f* receive high weight.
    Weights are label-balanced so that each class contributes equal total mass.

    Args:
        f_star   : (N,) array of OT dual potentials from compute_ot_potentials
        y_train  : (N,) array of training labels
        f_power  : α parameter controlling reweighting sharpness
                   (τ = median(f* - min(f*)) * α)
        clip_cap : maximum weight value; None to disable clipping

    Returns:
        weights : (N,) float64 array with mean ≈ 1.0
    """
    f_star = np.asarray(f_star, dtype=np.float64).reshape(-1)
    y      = np.asarray(y_train).reshape(-1)
    eps    = 1e-12

    f0  = f_star - np.min(f_star)
    tau = np.median(f0) * f_power
    w   = np.exp(-f0 / (tau + eps))

    labels = np.unique(y)

    # 1. Within-class normalization to avoid label-imbalance dominance
    for cls in labels:
        m = (y == cls)
        w[m] = w[m] / (w[m].mean() + eps)

    # 2. Enforce uniform class mass (each label gets 1/K of total weight)
    total  = w.sum()
    target = total / max(len(labels), 1)
    for cls in labels:
        m = (y == cls)
        w[m] *= target / (w[m].sum() + eps)

    # 3. Global normalization + optional clipping
    w = w / (w.mean() + eps)
    if clip_cap is not None:
        w = np.clip(w, 0.0, float(clip_cap))
        w = w / (w.mean() + eps)

    return w
