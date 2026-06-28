"""Debiased entropic OT drifting loss (W-Flow + CWG-E cluster-wise branch).

Drop-in alternative to ``drift_loss`` that replaces softmax-based affinities
with Sinkhorn optimal-transport barycentric maps.

Calling convention mirrors ``drift_loss``:
    loss, info = drift_loss_ot(gen, fixed_pos, fixed_neg, ...)

CWG-E extension: when ``cluster_mode != "none"``, the Sinkhorn coupling is
solved within feature-space clusters (hard / soft block-mask on the cost
matrix) instead of across the whole mini-batch. ``cluster_mode = "none"``
(default) preserves W-Flow's original behaviour exactly, including the
batched-multi-R / new-CFG paths.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from clustering import batched_kmeans, assign_to_centroids, expand_centroids

_COMPILE = os.environ.get("DRIFT_COMPILE", "1") != "0"


# ---------------------------------------------------------------------------
# Sinkhorn OT plan (balanced, log-domain)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sinkhorn_plan_log_impl(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: torch.Tensor,
    diag_mask: bool = False,
    num_iter: int = 50,
    stop_thr: float = 1e-4,
    target_weights: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Balanced entropic OT plan in log domain, batched over dim-0.

    Args:
        X: source points   [B, N, D]
        Y: target points   [B, M, D]
        reg: scalar tensor, entropic regularisation (> 0)
        diag_mask: if True, mask the first N diagonal entries of the cost
        num_iter: fixed number of Sinkhorn iterations
        stop_thr: unused (kept for API compat)
        target_weights: optional non-uniform target marginal [B, M]
        use_quadratic_cost: if True, use c(x,y) = ||x-y||^2 / 2 instead of ||x-y||

    Returns:
        Pi: transport plan  [B, N, M]
    """
    device, dtype = X.device, X.dtype
    B, N, _ = X.shape
    M = Y.shape[1]

    C = torch.cdist(X, Y)  # [B, N, M]
    if use_quadratic_cost:
        C = 0.5 * C * C

    if diag_mask:
        diag = torch.arange(min(N, M), device=device)
        C[:, diag, diag] = C[:, diag, diag] + 1e6

    logK = -C / reg  # [B, N, M]

    log_a = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
    if target_weights is None:
        log_b = torch.full((B, M), -math.log(M), device=device, dtype=dtype)
    else:
        b = target_weights.to(device=device, dtype=dtype)
        b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    n = max(int(num_iter), 1)
    # MMD: n = 0, i.e., unnormalized kernel
    # KL: n = 0.5, i.e., only compute: log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
    for _ in range(n):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)

    return torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])


if _COMPILE:
    _sinkhorn_plan_log = torch.compile(_sinkhorn_plan_log_impl, dynamic=True)
else:
    _sinkhorn_plan_log = _sinkhorn_plan_log_impl


# ---------------------------------------------------------------------------
# Sinkhorn OT plan — batched: per-element reg [B] and diag_mask [B]
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sinkhorn_batched_impl(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: torch.Tensor,
    diag_mask: torch.Tensor,
    num_iter: int,
    target_weights: torch.Tensor,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Sinkhorn OT plan with per-element regularisation and diagonal mask.

    Batches multiple independent transport problems (different reg values,
    different masking) into a single kernel launch.

    Args:
        X: source points          [B, N, D]
        Y: target points          [B, M, D]
        reg: per-element reg      [B]
        diag_mask: per-element    [B] bool
        num_iter: fixed Sinkhorn iterations
        target_weights:           [B, M]  (always required — no None branch)
        use_quadratic_cost: if True, use c(x,y) = ||x-y||^2 / 2 instead of ||x-y||

    Returns:
        Pi: transport plan  [B, N, M]
    """
    device, dtype = X.device, X.dtype
    B, N, _ = X.shape
    M = Y.shape[1]

    C = torch.cdist(X, Y)  # [B, N, M]
    if use_quadratic_cost:
        C = 0.5 * C * C

    eye = torch.eye(N, M, device=device, dtype=dtype)
    C = C + eye[None] * diag_mask.to(dtype=dtype)[:, None, None] * 1e6

    logK = -C / reg[:, None, None]  # [B, N, M]

    log_a = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
    b = target_weights.to(dtype=dtype)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    n = max(int(num_iter), 1)
    for _ in range(n):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)

    return torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])


if _COMPILE:
    _sinkhorn_batched = torch.compile(_sinkhorn_batched_impl, dynamic=True)
else:
    _sinkhorn_batched = _sinkhorn_batched_impl


# ---------------------------------------------------------------------------
# Barycentric map (non-batched, kept for backward compat / standalone use)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _barycentric_map(
    z: torch.Tensor,
    support: torch.Tensor,
    eps: torch.Tensor,
    num_iter: int = 50,
    stop_thr: float = 1e-4,
    diag_mask: bool = False,
    target_weights: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """OT barycentric projection: T(z) = (Pi @ support) / row_mass.

    All inputs are detached internally -- no gradients flow through OT.
    """
    Pi = _sinkhorn_plan_log(
        z.detach(),
        support.detach(),
        reg=eps,
        diag_mask=diag_mask,
        num_iter=int(num_iter),
        stop_thr=float(stop_thr),
        target_weights=target_weights,
        use_quadratic_cost=use_quadratic_cost,
    )
    row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.bmm(Pi, support.detach()) / row_mass


# ---------------------------------------------------------------------------
# Debiased OT velocity field
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_V_debiased(
    z: torch.Tensor,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    eps: torch.Tensor,
    num_iter: int = 50,
    stop_thr: float = 1e-4,
    neg_diag_mask: bool = True,
    neg_target_weights: torch.Tensor | None = None,
    cfg_weight: torch.Tensor | None = None,
    w_uncond: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Debiased OT velocity: V = T_pq - T_qneg  (+ optional CFG term).

    Args:
        z:       generated features  [B, N, D]
        w_pos:   positive features   [B, P, D]
        w_neg:   negative features   [B, Q, D]
        eps:     scalar tensor, entropic regularisation
        neg_diag_mask: mask self-transport in z->neg map
        neg_target_weights: optional weights for neg marginal [B, Q]
        cfg_weight: per-batch CFG strength [B] for new-CFG mode
        w_uncond:   unconditional features [B, U, D] for new-CFG mode
        use_quadratic_cost: if True, use quadratic OT cost

    Returns:
        V: velocity field  [B, N, D]
    """
    bary_kw = dict(eps=eps, num_iter=num_iter, stop_thr=stop_thr,
                   use_quadratic_cost=use_quadratic_cost)

    T_pq = _barycentric_map(z, w_pos, diag_mask=False, **bary_kw)
    T_qneg = _barycentric_map(
        z, w_neg,
        diag_mask=neg_diag_mask,
        target_weights=neg_target_weights,
        **bary_kw,
    )

    V = T_pq - T_qneg

    if cfg_weight is not None and w_uncond is not None:
        T_quncond = _barycentric_map(z, w_uncond, diag_mask=False, **bary_kw)
        V = V + cfg_weight.view(-1, 1, 1) * (T_pq - T_quncond)

    return V


# ---------------------------------------------------------------------------
# Batched velocity computation across all R values (optimised path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_V_all_R(
    z: torch.Tensor,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    R_list: Tuple[float, ...],
    reg_scale: float,
    num_iter: int,
    neg_diag_mask: bool,
    neg_target_weights: torch.Tensor | None,
    cfg_weight: torch.Tensor | None,
    w_uncond: torch.Tensor | None,
    use_quadratic_cost: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute aggregated velocity field for all R values in one Sinkhorn call.

    Batches every (R_value, map_type) combination along dim-0 with per-element
    ``reg`` and ``diag_mask``, replacing 6-9 sequential Sinkhorn invocations
    with a single one.

    Args:
        reg_scale: sqrt(S) for L2 cost, S for quadratic cost.

    Returns:
        V_agg:   [B, N, D]  aggregated (normalised) velocity
        f_norms: [num_R]    per-R velocity norms for logging
    """
    device, dtype = z.device, z.dtype
    B, N, D = z.shape
    num_R = len(R_list)
    use_cfg = cfg_weight is not None and w_uncond is not None
    maps_per_R = 3 if use_cfg else 2
    total_maps = num_R * maps_per_R

    P, Q = w_pos.shape[1], w_neg.shape[1]
    U = w_uncond.shape[1] if use_cfg else 0
    max_M = max(P, Q, U) if use_cfg else max(P, Q)

    # --- Pad targets and weights to uniform max_M ---
    # Padded target *positions* are filled with a large sentinel (not zeros) so
    # that cdist produces a huge cost and Sinkhorn routes no mass there, even
    # with a very sharp kernel (small reg).  Padded *weights* remain zero.
    _PAD_VAL = 1e4

    def _pad(t: torch.Tensor, tw: torch.Tensor | None, M_orig: int):
        if tw is None:
            tw = torch.ones(B, M_orig, device=device, dtype=dtype)
        if M_orig == max_M:
            return t, tw
        pad_size = max_M - M_orig
        t_padded = torch.cat([t, t.new_full((B, pad_size, D), _PAD_VAL)], dim=1)
        tw_padded = F.pad(tw, (0, pad_size))
        return t_padded, tw_padded

    pos_p, pos_tw = _pad(w_pos, None, P)
    neg_p, neg_tw = _pad(w_neg, neg_target_weights, Q)

    if use_cfg:
        unc_p, unc_tw = _pad(w_uncond, None, U)
        Y_per_R = torch.cat([pos_p, neg_p, unc_p], dim=0)
        W_per_R = torch.cat([pos_tw, neg_tw, unc_tw], dim=0)
    else:
        Y_per_R = torch.cat([pos_p, neg_p], dim=0)
        W_per_R = torch.cat([pos_tw, neg_tw], dim=0)

    # --- Stack across all R values ---
    X_batch = z.repeat(total_maps, 1, 1)        # [total*B, N, D]
    Y_batch = Y_per_R.repeat(num_R, 1, 1)       # [total*B, max_M, D]
    W_batch = W_per_R.repeat(num_R, 1)           # [total*B, max_M]

    # Per-element diag_mask: [pos=F, neg=?, uncond=F] repeated per R
    dm_false = torch.zeros(B, device=device, dtype=torch.bool)
    dm_neg = torch.full((B,), neg_diag_mask, device=device, dtype=torch.bool)
    if use_cfg:
        dm_per_R = torch.cat([dm_false, dm_neg, dm_false])
    else:
        dm_per_R = torch.cat([dm_false, dm_neg])
    DM_batch = dm_per_R.repeat(num_R)

    # Per-element reg: constant within one R, different across R
    eps_per_R = torch.tensor(
        [float(R) * reg_scale for R in R_list], device=device, dtype=dtype,
    )
    REG_batch = eps_per_R.repeat_interleave(maps_per_R * B)

    # --- Single Sinkhorn call ---
    Pi = _sinkhorn_batched(
        X_batch, Y_batch, REG_batch, DM_batch, num_iter, W_batch,
        use_quadratic_cost=use_quadratic_cost,
    )

    # --- Barycentric maps: T = (Pi @ Y) / row_mass ---
    row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    T_all = torch.bmm(Pi, Y_batch) / row_mass   # [total*B, N, D]
    T_all = T_all.view(num_R, maps_per_R, B, N, D)

    T_pq = T_all[:, 0]                           # [num_R, B, N, D]
    T_qneg = T_all[:, 1]                         # [num_R, B, N, D]

    V_raw = T_pq - T_qneg
    if use_cfg:
        T_quncond = T_all[:, 2]
        V_raw = V_raw + cfg_weight[None, :, None, None] * (T_pq - T_quncond)

    # Per-R normalisation and sum
    f_norms = (V_raw ** 2).mean(dim=(1, 2, 3))               # [num_R]
    force_scales = torch.sqrt(f_norms.clamp(min=1e-8))       # [num_R]
    V_agg = (V_raw / force_scales[:, None, None, None]).sum(dim=0)

    return V_agg, f_norms


# ---------------------------------------------------------------------------
# CWG-E: cluster-wise Sinkhorn (block-mask on cost) + debiased velocity
# ---------------------------------------------------------------------------

@torch.no_grad()
def _block_penalty(
    labels_x: torch.Tensor,   # [B, N]
    labels_y: torch.Tensor,   # [B, M]
    centroids: torch.Tensor,  # [B, K, D]
    cluster_mode: str,
    mask_lambda: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive cost penalty implementing the cluster block-mask.

    "hard" -> 1e6 on off-block pairs (later zeroed in Π for exact within-cluster).
    "soft" -> mask_lambda * ||centroid_i - centroid_j||^2 (smooth cross-routing).
    """
    B, N = labels_x.shape
    M = labels_y.shape[1]
    K = centroids.shape[1]

    if cluster_mode == "hard":
        off = (labels_x[:, :, None] != labels_y[:, None, :]).to(dtype)
        return off * 1e6

    if cluster_mode == "soft":
        cc = torch.cdist(centroids, centroids) ** 2                   # [B, K, K]
        rows = cc.gather(1, labels_x[:, :, None].expand(B, N, K))      # [B, N, K]
        pen = rows.gather(2, labels_y[:, None, :].expand(B, N, M))     # [B, N, M]
        return mask_lambda * pen.to(dtype)

    raise ValueError(f"unknown cluster_mode: {cluster_mode}")


@torch.no_grad()
def _sinkhorn_block_impl(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: torch.Tensor,
    labels_x: torch.Tensor,
    labels_y: torch.Tensor,
    centroids: torch.Tensor,
    cluster_mode: str,
    mask_lambda: float,
    num_iter: int = 50,
    target_weights: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Block-masked Sinkhorn plan in log domain (batched over dim-0).

    For ``"hard"`` mode, the residual ~1% off-block mass left by log-domain
    Sinkhorn (dual-potential compensation against the 1e6 penalty) is zeroed
    explicitly before return.
    """
    device, dtype = X.device, X.dtype
    B, N, _ = X.shape
    M = Y.shape[1]

    C = torch.cdist(X, Y)
    if use_quadratic_cost:
        C = 0.5 * C * C

    C = C + _block_penalty(
        labels_x, labels_y, centroids, cluster_mode, mask_lambda, dtype,
    )

    logK = -C / reg

    log_a = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
    if target_weights is None:
        log_b = torch.full((B, M), -math.log(M), device=device, dtype=dtype)
    else:
        b = target_weights.to(device=device, dtype=dtype)
        b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max(int(num_iter), 1)):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)

    Pi = torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])

    if cluster_mode == "hard":
        on_block = (labels_x[:, :, None] == labels_y[:, None, :]).to(dtype)
        Pi = Pi * on_block

    return Pi


if _COMPILE:
    _sinkhorn_block = torch.compile(_sinkhorn_block_impl, dynamic=True)
else:
    _sinkhorn_block = _sinkhorn_block_impl


@torch.no_grad()
def _barycentric_map_clustered(
    z: torch.Tensor,
    support: torch.Tensor,
    eps: torch.Tensor,
    labels_z: torch.Tensor,
    labels_s: torch.Tensor,
    centroids: torch.Tensor,
    cluster_mode: str,
    mask_lambda: float,
    num_iter: int = 50,
    target_weights: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    Pi = _sinkhorn_block(
        z.detach(),
        support.detach(),
        reg=eps,
        labels_x=labels_z,
        labels_y=labels_s,
        centroids=centroids,
        cluster_mode=cluster_mode,
        mask_lambda=mask_lambda,
        num_iter=int(num_iter),
        target_weights=target_weights,
        use_quadratic_cost=use_quadratic_cost,
    )
    row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.bmm(Pi, support.detach()) / row_mass


@torch.no_grad()
def _sinkhorn_small(
    X: torch.Tensor,                     # [n, D]
    Y: torch.Tensor,                     # [m, D]
    eps: torch.Tensor,
    num_iter: int = 50,
    target_weights: torch.Tensor | None = None,  # [m]
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """Single-problem log-domain Sinkhorn (no batch dim). Returns Pi [n, m]."""
    device, dtype = X.device, X.dtype
    n, _ = X.shape
    m = Y.shape[0]

    C = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)
    if use_quadratic_cost:
        C = 0.5 * C * C
    logK = -C / eps

    log_a = torch.full((n,), -math.log(n), device=device, dtype=dtype)
    if target_weights is None:
        log_b = torch.full((m,), -math.log(m), device=device, dtype=dtype)
    else:
        b = target_weights.to(device=device, dtype=dtype)
        b = b / b.sum().clamp_min(1e-12)
        log_b = torch.log(b.clamp_min(1e-30))

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(num_iter), 1)):
        log_u = log_a - torch.logsumexp(logK + log_v[None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.t() + log_u[None, :], dim=-1)
    return torch.exp(log_u[:, None] + logK + log_v[None, :])


@torch.no_grad()
def _per_cluster_positions(labels: torch.Tensor, K: int) -> torch.Tensor:
    """For each point i, its rank within its cluster.

    Returns [N] int64 with values in [0, max_cluster_size).
    """
    N = labels.shape[0]
    device = labels.device
    pos = torch.zeros(N, device=device, dtype=torch.long)
    for k in range(int(K)):
        mask = labels == k
        nk = int(mask.sum().item())
        if nk:
            pos[mask] = torch.arange(nk, device=device)
    return pos


@torch.no_grad()
def _per_cluster_bary(
    z: torch.Tensor,           # [B, N, D]
    support: torch.Tensor,     # [B, M, D]
    labels_z: torch.Tensor,    # [B, N]
    labels_s: torch.Tensor,    # [B, M]
    K: int,
    eps: torch.Tensor,
    num_iter: int,
    target_weights: torch.Tensor | None = None,   # [B, M]
    use_quadratic_cost: bool = False,
) -> torch.Tensor:
    """K independent Sinkhorn problems within each cluster — batched.

    Builds padded [K, max_nz, D] / [K, max_ms, D] tensors and routes a SINGLE
    batched Sinkhorn call (B'=K) instead of K Python iterations. Total
    Sinkhorn cost ≈ K * max_nz * max_ms (≈ N^2 / K in the balanced case),
    vs N^2 for the block-mask path.

    Empty-cluster handling: clusters with empty z stay at identity in T (so
    V contribution = 0). Clusters with empty support same — masked out via
    a dummy weight then their rows of T are not used.
    """
    B, N, D = z.shape
    M = support.shape[1]
    device, dtype = z.device, z.dtype
    T = z.clone()
    K = int(K)

    for b in range(B):
        lz = labels_z[b]                            # [N]
        ls = labels_s[b]                            # [M]
        nz_per_k = torch.bincount(lz, minlength=K)
        ns_per_k = torch.bincount(ls, minlength=K)
        nonempty = (nz_per_k > 0) & (ns_per_k > 0)
        if not bool(nonempty.any().item()):
            continue
        max_nz = int(nz_per_k.max().item())
        max_ms = int(ns_per_k.max().item())

        pos_z = _per_cluster_positions(lz, K)
        pos_s = _per_cluster_positions(ls, K)

        _PAD_VAL = 1e4
        z_pad = z.new_full((K, max_nz, D), _PAD_VAL)
        s_pad = z.new_full((K, max_ms, D), _PAD_VAL)
        s_w = z.new_zeros((K, max_ms))
        z_pad[lz, pos_z] = z[b]
        s_pad[ls, pos_s] = support[b]
        if target_weights is None:
            s_w[ls, pos_s] = 1.0
        else:
            s_w[ls, pos_s] = target_weights[b]

        # For clusters with empty support, plant a dummy weight at position 0
        # so the Sinkhorn b-marginal isn't zero (would log(0) → -inf).
        # Those clusters' T rows are discarded later via the nonempty mask.
        empty_s_mask = ~(ns_per_k > 0)
        if bool(empty_s_mask.any().item()):
            s_w[empty_s_mask, 0] = 1.0

        reg_per_k = torch.full((K,), float(eps), device=device, dtype=dtype)
        dm_per_k = torch.zeros(K, device=device, dtype=torch.bool)

        Pi = _sinkhorn_batched(
            z_pad, s_pad, reg_per_k, dm_per_k, num_iter, s_w,
            use_quadratic_cost=use_quadratic_cost,
        )                                            # [K, max_nz, max_ms]
        row_mass = Pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        T_pad = torch.bmm(Pi, s_pad) / row_mass      # [K, max_nz, D]

        # Scatter back; only for points whose cluster is non-empty on BOTH sides.
        # Other points keep T = z (identity, contributing 0 to V_pos - V_neg).
        valid_at_point = nonempty[lz]                # [N]
        T_b = z[b].clone()
        idx_k = lz[valid_at_point]
        idx_p = pos_z[valid_at_point]
        T_b[valid_at_point] = T_pad[idx_k, idx_p]
        T[b] = T_b

    return T


@torch.no_grad()
def _sinkhorn_AB(
    C: torch.Tensor,             # [B, K, K] cost
    a: torch.Tensor,             # [B, K] source marginal (may have zeros)
    b: torch.Tensor,             # [B, K] target marginal (may have zeros)
    reg: float,
    num_iter: int = 50,
) -> torch.Tensor:
    """Log-domain Sinkhorn with both marginals; safe for zero entries."""
    log_a = torch.log(a.clamp_min(1e-30))
    log_b = torch.log(b.clamp_min(1e-30))
    logK = -C / float(reg)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(num_iter), 1)):
        log_u = log_a - torch.logsumexp(logK + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u[:, None, :], dim=-1)
    return torch.exp(log_u[:, :, None] + logK + log_v[:, None, :])


@torch.no_grad()
def _outer_gamma_targets(
    centroids_p: torch.Tensor,     # [B, K, D]   target-side centroids
    centroids_q: torch.Tensor,     # [B, K, D]   source-side centroids
    marg_q: torch.Tensor,          # [B, K]      q cluster sizes (normalised)
    marg_p: torch.Tensor,          # [B, K]      p cluster sizes (normalised)
    eps_gamma: float,
    num_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outer centroid-to-centroid Sinkhorn coupling Γ.

    Returns:
        beta:        [B, K]    diagonal weight of Γ per source cluster
                              (β_k = 1 → all mass stays within cluster).
        off_target:  [B, K, D] weighted average of OFF-diagonal target
                              centroids, per source cluster.
    """
    # Cost between centroid pairs, scaled by their typical spread.
    C = torch.cdist(centroids_q, centroids_p) ** 2
    scale = C.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    C = C / scale
    Gamma = _sinkhorn_AB(C, marg_q, marg_p, reg=eps_gamma, num_iter=num_iter)

    Gamma_row = Gamma / Gamma.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    K = Gamma.shape[-1]
    eye = torch.eye(K, dtype=torch.bool, device=Gamma.device).unsqueeze(0)
    beta = Gamma_row.diagonal(dim1=-2, dim2=-1)                       # [B, K]

    Gamma_off = Gamma_row.masked_fill(eye, 0.0)                       # [B, K, K]
    off_sum = Gamma_off.sum(dim=-1, keepdim=True).clamp_min(1e-12)    # [B, K, 1]
    off_target = torch.bmm(Gamma_off, centroids_p) / off_sum          # [B, K, D]

    return beta, off_target


@torch.no_grad()
def _apply_outer_gamma(
    T_within: torch.Tensor,   # [B, N, D]   within-cluster barycentric
    labels_z: torch.Tensor,   # [B, N]
    beta: torch.Tensor,       # [B, K]
    off_target: torch.Tensor, # [B, K, D]
    z: torch.Tensor,          # [B, N, D]   source positions
) -> torch.Tensor:
    """Additive outer-Γ correction:

        T̃(x_i) = T_within(x_i) + (1-β_{k_i}) · (off_target_{k_i} - x_i)

    At q=p the marginals match → Γ ≈ I → β=1 → correction=0 → T̃ = T_within
    so V vanishes correctly (Thm 2). At α<1 with mass missing from some p
    cluster, β<1 and off_target pulls x_i toward those missing centroids.

    Earlier multiplicative form (β·T_within + (1-β)·off_target) shrank T̃
    toward off_target=0 even when off-coupling was tiny, producing a
    constant residual V ≈ (β-1)·x_i ≠ 0 at q=p.
    """
    beta_pp = beta.gather(1, labels_z)                                # [B, N]
    D = T_within.shape[-1]
    off_pp = off_target.gather(
        1, labels_z.unsqueeze(-1).expand(-1, -1, D),
    )                                                                  # [B, N, D]
    one_minus = (1.0 - beta_pp).unsqueeze(-1)
    return T_within + one_minus * (off_pp - z)


@torch.no_grad()
def _cluster_marginals(labels: torch.Tensor, K: int) -> torch.Tensor:
    """Per-batch normalised cluster mass. Returns [B, K], rows sum to 1."""
    B, N = labels.shape
    m = torch.zeros(B, K, device=labels.device, dtype=torch.float32)
    m.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
    return m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _compute_V_clustered(
    z: torch.Tensor,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    eps: torch.Tensor,
    cluster_mode: str,
    n_clusters: int,
    mask_lambda: float,
    num_iter: int = 50,
    neg_target_weights: torch.Tensor | None = None,
    cfg_weight: torch.Tensor | None = None,
    w_uncond: torch.Tensor | None = None,
    use_quadratic_cost: bool = False,
    cluster_centroids: torch.Tensor | None = None,
    use_per_cluster_sinkhorn: bool = True,
    use_outer_gamma: bool = False,
    outer_gamma_eps: float = 0.01,
    outer_gamma_iter: int = 200,
    cluster_marg_p_pool: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cluster-wise debiased OT velocity V = T_pq - T_qneg (+ optional CFG).

    Two clustering paths:
        * cluster_centroids is None: batched k-means on (z ⊕ pos ⊕ neg [⊕ uncond])
          — re-clustering per call (high variance across calls).
        * cluster_centroids provided ([K,D] or [B,K,D]): "sticky" — just assign,
          no k-means. Use this to share a partition across mini-batches.

    Two Sinkhorn paths:
        * use_per_cluster_sinkhorn=True (default, only for "hard"): K small
          Sinkhorns within each cluster. Real ~K-fold speedup, no off-block leak.
        * False / "soft": block-mask Sinkhorn (single batched call).
    """
    B = z.shape[0]
    has_uncond = cfg_weight is not None and w_uncond is not None

    if cluster_centroids is not None:
        centroids = expand_centroids(cluster_centroids.to(z), B)
        labels_z = assign_to_centroids(z, centroids)
        labels_p = assign_to_centroids(w_pos, centroids)
        labels_n = assign_to_centroids(w_neg, centroids)
        labels_u_ = assign_to_centroids(w_uncond, centroids) if has_uncond else None
    else:
        parts = [z, w_pos, w_neg]
        if has_uncond:
            parts.append(w_uncond)
        union = torch.cat(parts, dim=1)
        labels_u, centroids = batched_kmeans(union, K=n_clusters)
        N = z.shape[1]
        P = w_pos.shape[1]
        Q = w_neg.shape[1]
        labels_z = labels_u[:, :N]
        labels_p = labels_u[:, N:N + P]
        labels_n = labels_u[:, N + P:N + P + Q]
        labels_u_ = labels_u[:, N + P + Q:] if has_uncond else None

    # --- per-cluster Sinkhorn (real speedup) for hard mode ---
    if cluster_mode == "hard" and use_per_cluster_sinkhorn:
        T_pq = _per_cluster_bary(
            z, w_pos, labels_z, labels_p, n_clusters,
            eps=eps, num_iter=num_iter,
            use_quadratic_cost=use_quadratic_cost,
        )
        T_qneg = _per_cluster_bary(
            z, w_neg, labels_z, labels_n, n_clusters,
            eps=eps, num_iter=num_iter,
            target_weights=neg_target_weights,
            use_quadratic_cost=use_quadratic_cost,
        )

        # Outer Γ correction: cluster-to-cluster Sinkhorn over centroids
        # restores the "vanishes iff q=p" property by routing mass between
        # source clusters whose marginal is non-zero and target clusters
        # whose marginal is non-zero, even when one side is empty.
        if use_outer_gamma:
            cz = _cluster_marginals(labels_z, n_clusters).to(z.dtype)
            cn = _cluster_marginals(labels_n, n_clusters).to(z.dtype)
            # Sticky marg_p from pool removes mini-batch noise in Γ → preserves
            # the variance win (Prop 1). marg_q still varies per call so the
            # missing-mode signal for Thm 2 is unaffected.
            if cluster_marg_p_pool is not None:
                cp = cluster_marg_p_pool.to(z)
                if cp.dim() == 1:
                    cp = cp.unsqueeze(0).expand(z.shape[0], -1)
            else:
                cp = _cluster_marginals(labels_p, n_clusters).to(z.dtype)
            beta_p, off_p = _outer_gamma_targets(
                centroids, centroids, cz, cp,
                eps_gamma=outer_gamma_eps, num_iter=outer_gamma_iter,
            )
            T_pq = _apply_outer_gamma(T_pq, labels_z, beta_p, off_p, z=z)
            beta_n, off_n = _outer_gamma_targets(
                centroids, centroids, cz, cn,
                eps_gamma=outer_gamma_eps, num_iter=outer_gamma_iter,
            )
            T_qneg = _apply_outer_gamma(T_qneg, labels_z, beta_n, off_n, z=z)

        V = T_pq - T_qneg
        if labels_u_ is not None:
            T_quncond = _per_cluster_bary(
                z, w_uncond, labels_z, labels_u_, n_clusters,
                eps=eps, num_iter=num_iter,
                use_quadratic_cost=use_quadratic_cost,
            )
            if use_outer_gamma:
                cu = _cluster_marginals(labels_u_, n_clusters).to(z.dtype)
                beta_u, off_u = _outer_gamma_targets(
                    centroids, centroids, cz, cu,
                    eps_gamma=outer_gamma_eps, num_iter=outer_gamma_iter,
                )
                T_quncond = _apply_outer_gamma(T_quncond, labels_z, beta_u, off_u, z=z)
            V = V + cfg_weight.view(-1, 1, 1) * (T_pq - T_quncond)
        return V

    # --- block-mask path (mathematically cluster-wise, no speedup) ---
    bary_kw = dict(eps=eps, num_iter=num_iter, mask_lambda=mask_lambda,
                   use_quadratic_cost=use_quadratic_cost,
                   cluster_mode=cluster_mode, centroids=centroids)

    T_pq = _barycentric_map_clustered(
        z, w_pos, labels_z=labels_z, labels_s=labels_p, **bary_kw,
    )
    T_qneg = _barycentric_map_clustered(
        z, w_neg, labels_z=labels_z, labels_s=labels_n,
        target_weights=neg_target_weights, **bary_kw,
    )
    V = T_pq - T_qneg
    if labels_u_ is not None:
        T_quncond = _barycentric_map_clustered(
            z, w_uncond, labels_z=labels_z, labels_s=labels_u_, **bary_kw,
        )
        V = V + cfg_weight.view(-1, 1, 1) * (T_pq - T_quncond)
    return V


# ---------------------------------------------------------------------------
# Public entry point  (mirrors drift_loss signature)
# ---------------------------------------------------------------------------

def drift_loss_ot(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Iterable[float] = (0.02, 0.05, 0.2),
    sinkhorn_num_iter: int = 20,
    sinkhorn_stop_thr: float = 1e-4,
    use_new_cfg: bool = False,
    fixed_uncond: torch.Tensor | None = None,
    weight_uncond: torch.Tensor | None = None,
    disable_diag_mask: bool = False,
    batch_sinkhorn: bool = False,
    use_quadratic_cost: bool = False,
    # ---- CWG-E additions ----
    cluster_mode: str = "none",
    n_clusters: int = 8,
    mask_lambda: float = 1.0,
    cluster_centroids: torch.Tensor | None = None,
    use_per_cluster_sinkhorn: bool = True,
    use_outer_gamma: bool = False,
    outer_gamma_eps: float = 0.01,
    outer_gamma_iter: int = 200,
    cluster_marg_p_pool: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Debiased entropic-OT drifting loss.

    Same return contract as ``drift_loss``:
        loss  [B]  per-sample MSE
        info  dict of scalar metrics

    When *use_new_cfg=False* (default), unconditional negatives are expected to
    already be part of ``fixed_neg`` with appropriate ``weight_neg`` -- exactly
    the same as the original ``drift_loss``.

    When *use_new_cfg=True*, unconditional samples are passed separately via
    ``fixed_uncond`` / ``weight_uncond`` and a third barycentric map is used.

    When *batch_sinkhorn=True*, all R values and map types are batched into a
    single Sinkhorn call (6-9x fewer kernel launches).  Default *False*
    preserves the original sequential behaviour.

    CWG-E options
    -------------
    *cluster_mode*: "none" (default, exact W-Flow) | "hard" | "soft".
    *n_clusters*: K for the in-loss batched k-means.
    *mask_lambda*: cross-cluster penalty weight (only used by "soft").

    When ``cluster_mode != "none"`` the multi-R loop falls back to a sequential
    per-R Sinkhorn (``batch_sinkhorn`` is bypassed) so the per-R cluster
    partition can be reused without reshuffling Sinkhorn batches.
    """
    B, C_g, S = gen.shape

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()

    # -- feature-scale normalisation (same logic as drift_loss) --
    with torch.no_grad():
        info: Dict[str, torch.Tensor] = {}

        targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

        dist = torch.cdist(old_gen, targets)
        if use_quadratic_cost:
            weighted_dist_sq = (dist * dist) * targets_w[:, None, :]
            scale = (weighted_dist_sq.mean() / (targets_w.mean() + 1e-8)).sqrt()
            del weighted_dist_sq
        else:
            weighted_dist = dist * targets_w[:, None, :]
            scale = weighted_dist.mean() / (targets_w.mean() + 1e-8)
            del weighted_dist
        info["scale"] = scale
        del targets, targets_w, dist

        scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)

    old_gen_scaled = old_gen / scale_inputs
    pos_scaled = fixed_pos.detach() / scale_inputs
    neg_scaled = fixed_neg.detach() / scale_inputs
    del old_gen, fixed_pos, fixed_neg, weight_gen, weight_pos

    # -- prepare new-CFG components --
    cfg_weight_per_sample = None
    uncond_scaled = None
    if use_new_cfg and fixed_uncond is not None:
        fixed_uncond = fixed_uncond.float()
        uncond_scaled = fixed_uncond.detach() / scale_inputs
        del fixed_uncond
        if weight_uncond is not None:
            cfg_weight_per_sample = weight_uncond.float()[:, 0]
        neg_target_weights = None
    else:
        neg_target_weights = weight_neg

    # -- accumulate velocity across R values --
    # Reg scales with cost magnitude: sqrt(S) for L2, S for quadratic
    reg_scale = float(S) if use_quadratic_cost else math.sqrt(S)
    R_tuple = tuple(float(r) for r in R_list)
    use_cluster = cluster_mode != "none"
    with torch.no_grad():
        if use_cluster:
            V_agg = torch.zeros_like(old_gen_scaled)
            for R in R_tuple:
                eps_eff = torch.tensor(
                    float(R) * reg_scale,
                    device=old_gen_scaled.device, dtype=old_gen_scaled.dtype,
                )
                V_raw = _compute_V_clustered(
                    old_gen_scaled,
                    pos_scaled,
                    neg_scaled,
                    eps=eps_eff,
                    cluster_mode=cluster_mode,
                    n_clusters=n_clusters,
                    mask_lambda=mask_lambda,
                    num_iter=sinkhorn_num_iter,
                    neg_target_weights=neg_target_weights,
                    cfg_weight=cfg_weight_per_sample,
                    w_uncond=uncond_scaled,
                    use_quadratic_cost=use_quadratic_cost,
                    cluster_centroids=(
                        cluster_centroids.to(old_gen_scaled) / scale_inputs
                        if cluster_centroids is not None else None
                    ),
                    use_per_cluster_sinkhorn=use_per_cluster_sinkhorn,
                    use_outer_gamma=use_outer_gamma,
                    outer_gamma_eps=outer_gamma_eps,
                    outer_gamma_iter=outer_gamma_iter,
                    cluster_marg_p_pool=cluster_marg_p_pool,
                )
                f_norm = (V_raw ** 2).mean()
                info[f"loss_{R}"] = f_norm
                force_scale = torch.sqrt(torch.clamp(f_norm, min=1e-8))
                V_agg = V_agg + V_raw / force_scale
        elif batch_sinkhorn:
            V_agg, f_norms = _compute_V_all_R(
                old_gen_scaled,
                pos_scaled,
                neg_scaled,
                R_list=R_tuple,
                reg_scale=reg_scale,
                num_iter=sinkhorn_num_iter,
                neg_diag_mask=(not disable_diag_mask),
                neg_target_weights=neg_target_weights,
                cfg_weight=cfg_weight_per_sample,
                w_uncond=uncond_scaled,
                use_quadratic_cost=use_quadratic_cost,
            )
            for i, R in enumerate(R_tuple):
                info[f"loss_{R}"] = f_norms[i]
        else:
            V_agg = torch.zeros_like(old_gen_scaled)
            for R in R_tuple:
                eps_eff = torch.tensor(
                    float(R) * reg_scale,
                    device=old_gen_scaled.device, dtype=old_gen_scaled.dtype,
                )
                V_raw = _compute_V_debiased(
                    old_gen_scaled,
                    pos_scaled,
                    neg_scaled,
                    eps=eps_eff,
                    num_iter=sinkhorn_num_iter,
                    stop_thr=sinkhorn_stop_thr,
                    neg_diag_mask=(not disable_diag_mask),
                    neg_target_weights=neg_target_weights,
                    cfg_weight=cfg_weight_per_sample,
                    w_uncond=uncond_scaled,
                    use_quadratic_cost=use_quadratic_cost,
                )
                f_norm = (V_raw ** 2).mean()
                info[f"loss_{R}"] = f_norm
                force_scale = torch.sqrt(torch.clamp(f_norm, min=1e-8))
                V_agg = V_agg + V_raw / force_scale

        goal_scaled = old_gen_scaled + V_agg

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = torch.mean(diff ** 2, dim=(-1, -2))
    info = {k: v.mean() for k, v in info.items()}
    return loss, info
