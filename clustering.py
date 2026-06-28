"""Batched k-means used by the cluster-wise OT drift loss.

CWG-E addition over the upstream W-Flow code. Runs entirely on the input
tensor's device / dtype with no Python loops over the batch dimension,
matching the [B, N, D] feature layout used by ``drift_loss_ot``.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def batched_kmeans(
    X: torch.Tensor,
    K: int,
    num_iter: int = 10,
    seed: int = 0,
    init_centroids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-means on each row of a 3-D tensor independently.

    Args:
        X:               [B, N, D] feature tensor
        K:               number of clusters
        num_iter:        Lloyd iterations
        seed:            RNG seed for initial assignment (ignored if init_centroids)
        init_centroids:  optional [B, K, D] warm-start (sticky across calls)

    Returns:
        labels:    [B, N] int64, values in [0, K)
        centroids: [B, K, D]
    """
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    K = max(1, min(int(K), N))

    if init_centroids is not None:
        assert init_centroids.shape == (B, K, D), (
            f"init_centroids must be [B={B}, K={K}, D={D}], got {tuple(init_centroids.shape)}"
        )
        centroids = init_centroids.to(device=device, dtype=dtype).clone()
    else:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        init_idx = torch.randperm(N, generator=g)[:K].to(device)
        centroids = X[:, init_idx, :].clone()

    labels = torch.zeros(B, N, dtype=torch.long, device=device)

    for _ in range(int(num_iter)):
        dist = torch.cdist(X, centroids)
        labels = dist.argmin(dim=-1)

        onehot = torch.nn.functional.one_hot(labels, K).to(dtype)
        counts = onehot.sum(dim=1).clamp_min(1.0)
        new_centroids = torch.einsum("bnk,bnd->bkd", onehot, X) / counts[:, :, None]

        empty = (onehot.sum(dim=1) == 0)
        if empty.any():
            fallback = X[:, :K, :]
            new_centroids = torch.where(empty[:, :, None], fallback, new_centroids)

        centroids = new_centroids

    return labels, centroids


@torch.no_grad()
def assign_to_centroids(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Hard-assign each row of X [B, N, D] to nearest centroid in [B, K, D].

    No Lloyd updates; use when centroids are externally maintained (sticky).
    Returns labels [B, N] int64.
    """
    return torch.cdist(X, centroids).argmin(dim=-1)


def expand_centroids(centroids: torch.Tensor, B: int) -> torch.Tensor:
    """Promote [K, D] -> [B, K, D] (no copy unless contiguous required)."""
    if centroids.dim() == 2:
        return centroids.unsqueeze(0).expand(B, -1, -1).contiguous()
    if centroids.dim() == 3:
        return centroids
    raise ValueError(f"centroids must be [K,D] or [B,K,D], got {tuple(centroids.shape)}")
