"""Sticky cluster statistics for CWG-E training (centroids + marg_p_pool).

The two-level CWG-E coupling (per-cluster Sinkhorn + outer Γ) requires
*sticky* population-level statistics:

- ``centroids``    [K, D] — k-means centres of the target feature
  distribution, reused across mini-batches so cluster assignments are
  consistent over training steps (otherwise Γ absorbs clustering noise
  and the variance gain from Theorem A is destroyed).
- ``marg_p_pool``  [K]   — mass of each cluster under the FULL reference
  pool, used as the column marginal of the outer-Γ Sinkhorn. Sticky-pool
  marg_p is what restores the Prop-1 variance win in §6.2 — see
  ``docs/theory.md §3`` (Theorem B.2) and ``cwge_out_3/prop_variance``.

This module computes both from a token-pool tensor (typically built from
the positive memory bank after feature extraction). Refreshable on a
cadence to track slow drift in feature space.

Usage (in train.py):

    cluster_stats = ClusterStats(
        n_clusters=cfg.n_clusters,
        device=device,
        seed=cfg.seed,
    )
    # ...later, inside the training loop, after the memory bank is warm::
    if cluster_stats.should_refresh(step, cadence):
        feats = build_feature_pool(memory_bank, feature_extractor)  # [Npool, D]
        cluster_stats.refresh(feats, step=step)
    _ot_kw["cluster_centroids"]    = cluster_stats.centroids       # [K, D] or None
    _ot_kw["cluster_marg_p_pool"]  = cluster_stats.marg_p_pool     # [K]    or None

The centroids and marginal are routed through ``ot_loss_kwargs`` into
``drift_loss_ot`` and consumed by the cluster-wise branch.

Distributed (DDP / multi-node) note
-----------------------------------
DDP correctness requires every rank to use the **same centroids and
marg_p_pool** — otherwise the per-cluster Sinkhorn partitions disagree
across ranks and the all-reduced gradient is no longer a valid estimator
of a single coherent loss. ``refresh()`` therefore:

  1. has each rank build a small local pool of size ``max_pool_tokens // world``;
  2. all-gathers the local pools into one global pool (identical on every rank);
  3. runs the **same** k-means (same seed, same data) on every rank.

Step 3 yields bitwise-identical centroids without a separate broadcast
(``batched_kmeans`` is deterministic given identical inputs + seed).
The all-gather bandwidth is bounded by ``max_pool_tokens × D × 4 B``
(≈ 600 MB at the defaults for D=768; tune ``max_pool_tokens`` for your
network).
"""

from __future__ import annotations

from typing import Optional

import torch

from clustering import batched_kmeans, assign_to_centroids
from utils.dist_util import (
    dist_is_initialized,
    process_count,
    process_index,
    process_allgather,
)


class ClusterStats:
    """Holds sticky centroids and pool marginal for CWG-E."""

    def __init__(
        self,
        n_clusters: int,
        device: torch.device,
        seed: int = 0,
        kmeans_iter: int = 30,
        kmeans_iter_warm: int = 10,
    ):
        self.n_clusters: int = int(n_clusters)
        self.device: torch.device = device
        self.seed: int = int(seed)
        self.kmeans_iter: int = int(kmeans_iter)
        # When warm-starting from existing centroids the optimum is nearby —
        # a few Lloyd steps converge. Cuts 3× off subsequent refreshes.
        self.kmeans_iter_warm: int = int(kmeans_iter_warm)

        self.centroids: Optional[torch.Tensor] = None       # [K, D]
        self.marg_p_pool: Optional[torch.Tensor] = None     # [K]
        self.last_refresh_step: int = -1
        self.pool_size: int = 0

    def is_ready(self) -> bool:
        return self.centroids is not None and self.marg_p_pool is not None

    def should_refresh(self, step: int, cadence: int) -> bool:
        """Refresh on first call, then every ``cadence`` steps if cadence>0.

        ``cadence == 0`` => never refresh after the first computation
        (frozen sticky centroids — the default we use in the 2D benchmark).
        """
        if not self.is_ready():
            return True
        if cadence is None or int(cadence) <= 0:
            return False
        return (step - self.last_refresh_step) >= int(cadence)

    @torch.no_grad()
    def refresh(self, feature_pool: torch.Tensor, step: int = 0) -> None:
        """(Re)compute centroids + pool marginal from a flat token pool.

        DDP-aware: when ``torch.distributed`` is initialised, the local
        pool from each rank is all-gathered into a single global pool and
        every rank then runs the SAME deterministic k-means on it
        (identical centroids on every rank without an explicit broadcast).

        Parameters
        ----------
        feature_pool : [Npool_local, D] tensor of features on ``self.device``.
            ``Npool_local`` is the per-rank contribution; downstream global
            pool has size ``Npool_local × world_size``. Must be large enough
            that k-means converges (Npool_global >> K).
        step : int
        """
        if feature_pool.dim() != 2:
            raise ValueError(
                f"feature_pool must be [Npool, D]; got shape {tuple(feature_pool.shape)}"
            )
        feature_pool = feature_pool.to(self.device, non_blocking=True).contiguous()

        # All-gather across ranks to assemble an identical global pool.
        # process_allgather is a no-op when distributed is not initialised.
        if dist_is_initialized() and process_count() > 1:
            feature_pool = process_allgather(feature_pool, tiled=True)

        if feature_pool.shape[0] < self.n_clusters:
            raise ValueError(
                f"global feature_pool size {feature_pool.shape[0]} < n_clusters {self.n_clusters}; "
                "wait for the memory bank to fill up before refresh()."
            )

        # batched_kmeans expects [B, N, D]; warm-start with current centroids
        # if available so successive refreshes track drift instead of jumping.
        warm = (
            self.centroids is not None
            and self.centroids.shape[-1] == feature_pool.shape[-1]
        )
        init = self.centroids.unsqueeze(0) if warm else None
        # Fewer iters on warm-start — optimum is nearby.
        num_iter = self.kmeans_iter_warm if warm else self.kmeans_iter
        _, c = batched_kmeans(
            feature_pool.unsqueeze(0),
            K=self.n_clusters,
            num_iter=num_iter,
            seed=self.seed,
            init_centroids=init,
        )
        new_centroids = c.squeeze(0)                                # [K, D]
        labels = assign_to_centroids(
            feature_pool.unsqueeze(0), new_centroids.unsqueeze(0),
        ).squeeze(0)                                                # [Npool_global]
        counts = torch.bincount(labels, minlength=self.n_clusters).to(new_centroids.dtype)
        new_marg = counts / counts.sum().clamp_min(1.0)             # [K]

        # Safety net against cuDNN non-determinism: explicitly broadcast
        # rank-0 values so every rank holds bitwise-identical tensors.
        # Cheap (K×D + K floats); no-op when single-rank.
        if dist_is_initialized() and process_count() > 1:
            import torch.distributed as _dist
            new_centroids = new_centroids.contiguous()
            new_marg = new_marg.contiguous()
            _dist.broadcast(new_centroids, src=0)
            _dist.broadcast(new_marg, src=0)

        self.centroids = new_centroids
        self.marg_p_pool = new_marg
        self.last_refresh_step = int(step)
        self.pool_size = int(feature_pool.shape[0])

    def state_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "seed": self.seed,
            "kmeans_iter": self.kmeans_iter,
            "centroids": self.centroids.detach().cpu() if self.centroids is not None else None,
            "marg_p_pool": self.marg_p_pool.detach().cpu() if self.marg_p_pool is not None else None,
            "last_refresh_step": self.last_refresh_step,
            "pool_size": self.pool_size,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.n_clusters = int(sd.get("n_clusters", self.n_clusters))
        self.seed = int(sd.get("seed", self.seed))
        self.kmeans_iter = int(sd.get("kmeans_iter", self.kmeans_iter))
        c = sd.get("centroids")
        m = sd.get("marg_p_pool")
        self.centroids = c.to(self.device) if c is not None else None
        self.marg_p_pool = m.to(self.device) if m is not None else None
        self.last_refresh_step = int(sd.get("last_refresh_step", -1))
        self.pool_size = int(sd.get("pool_size", 0))


@torch.no_grad()
def build_feature_pool_from_bank(
    memory_bank,
    extract_fn,
    feature_key: str,
    n_classes: int,
    samples_per_class: int,
    device: torch.device,
    max_pool_tokens: int = 200_000,
    extract_chunk: int = 256,
) -> torch.Tensor:
    """Sample from the positive memory bank, extract features, flatten to [Npool, D].

    DDP-aware: returns a **per-rank** pool of size ``max_pool_tokens //
    world_size`` (rounded up). After ``ClusterStats.refresh`` all-gathers
    across ranks the global pool reaches ``max_pool_tokens``. Per-rank
    sampling distributes feature-extraction cost across GPUs and bounds
    the all-gather payload to one rank's share.

    Parameters
    ----------
    memory_bank : ArrayMemoryBank
        The positive bank populated in the training loop.
    extract_fn : callable(images_tensor) -> dict[str, Tensor] | Tensor
        Wraps the W-Flow ``feature_apply``. Output must contain
        ``feature_key`` (or be a tensor) of shape ``[B, x, d]`` or
        ``[B, x, f, d]`` — flattened to ``[Ntokens, D]``.
    feature_key : str
    n_classes : int
    samples_per_class : int
        How many bank entries per class. Total raw token count is
        ``n_classes × samples_per_class × tokens_per_sample``.
    device : torch.device
    max_pool_tokens : int
        Global-pool cap. Per-rank cap is ``ceil(max_pool_tokens / world)``.

    Returns
    -------
    pool : Tensor of shape [Npool_local, D] on ``device``.
    """
    world = process_count() if dist_is_initialized() else 1
    rank = process_index() if dist_is_initialized() else 0
    per_rank_cap = (int(max_pool_tokens) + world - 1) // world

    # Each rank draws a disjoint slice of class labels so the global pool
    # sees diverse classes (no duplication across ranks).
    classes_per_rank = (n_classes + world - 1) // world
    cstart = rank * classes_per_rank
    cend = min(cstart + classes_per_rank, n_classes)
    if cend <= cstart:
        # More ranks than classes — fall back to all classes everywhere
        # (the dedup happens via the per-rank token cap below).
        cstart, cend = 0, n_classes
    labels_local = torch.arange(cstart, cend).repeat_interleave(samples_per_class)

    images = memory_bank.sample(labels_local, n_samples=1)          # [N, ...] numpy/tensor
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)

    # Chunked extraction: avoid OOM when n_classes × samples_per_class is large.
    # MAE/VAE forwards through hundreds of images at once spike GPU memory.
    parts = []
    for s in range(0, images.shape[0], int(extract_chunk)):
        chunk = images[s:s + int(extract_chunk)].to(device, non_blocking=True)
        feats = extract_fn(chunk)
        if isinstance(feats, dict):
            feats = feats[feature_key]
        parts.append(feats.reshape(-1, feats.shape[-1]))
        del feats, chunk
    pool = torch.cat(parts, dim=0)                                   # [Ntokens, D]
    del parts

    if pool.shape[0] > per_rank_cap:
        # Subsample with a deterministic per-rank generator — different
        # subset on each rank, identical across runs given identical bank.
        g = torch.Generator(device="cpu").manual_seed(int(rank))
        idx = torch.randperm(pool.shape[0], generator=g)[:per_rank_cap].to(pool.device)
        pool = pool[idx]

    return pool.contiguous()
