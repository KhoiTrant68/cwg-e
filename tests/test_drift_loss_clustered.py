"""Unit tests for the CWG-E cluster-wise branch in drift_loss_ot.

Validates the two claims from §7B of the project summary:
    (i)  hard mode has zero off-block mass in Π;
    (ii) cluster mode improves own-cluster targeting vs the global baseline.

Plus a smoke test for the public ``drift_loss_ot`` API across
``cluster_mode in {"none", "hard", "soft"}``.

Run from the repo root:
    DRIFT_COMPILE=0 python -m pytest tests -v
"""

from __future__ import annotations

import os
os.environ.setdefault("DRIFT_COMPILE", "0")  # disable torch.compile in tests

import torch

from clustering import batched_kmeans
from drift_loss_ot import (
    _sinkhorn_block,
    _compute_V_clustered,
    drift_loss_ot,
)


def _make_three_clusters(B: int = 2, n_per: int = 40, D: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([
        [+3.0, +3.0],
        [-3.0, +3.0],
        [ 0.0, -3.0],
    ])
    pad = torch.zeros(3, D - 2)
    centres = torch.cat([centres, pad], dim=1)
    X = torch.cat([
        c + 0.3 * torch.randn(n_per, D, generator=g) for c in centres
    ], dim=0)
    X = X.unsqueeze(0).expand(B, -1, -1).clone()
    return X, centres


def test_off_block_leak_is_zero_in_hard_mode():
    torch.manual_seed(0)
    X, _ = _make_three_clusters(B=2)
    Y = X + 0.05 * torch.randn_like(X)

    labels_x, centroids = batched_kmeans(X, K=3, num_iter=10)
    labels_y, _ = batched_kmeans(Y, K=3, num_iter=10)

    reg = torch.tensor(0.05)
    Pi = _sinkhorn_block(
        X, Y, reg=reg,
        labels_x=labels_x, labels_y=labels_y, centroids=centroids,
        cluster_mode="hard", mask_lambda=0.0, num_iter=50,
    )

    off = (labels_x[:, :, None] != labels_y[:, None, :]).to(Pi.dtype)
    leaked = (Pi * off).sum().item()
    total = Pi.sum().item()
    assert leaked == 0.0, f"hard mode leaked {leaked:.3e} / {total:.3e} off-block"


def test_cluster_mode_improves_own_cluster_targeting():
    torch.manual_seed(0)
    X, _ = _make_three_clusters(B=1, n_per=60)
    Y = X + 0.05 * torch.randn_like(X)

    eps = torch.tensor(0.5)

    V_hard = _compute_V_clustered(
        X, Y, Y, eps=eps,
        cluster_mode="hard", n_clusters=3, mask_lambda=0.0,
        num_iter=50,
    )

    labels, centroids = batched_kmeans(X, K=3, num_iter=10)
    own_means = centroids.gather(1, labels[:, :, None].expand(-1, -1, X.shape[-1]))

    err_hard = (X + V_hard - own_means).pow(2).sum(-1).sqrt().mean()
    # Baseline: zero velocity (i.e., X stays where it is).
    err_none = (X - own_means).pow(2).sum(-1).sqrt().mean()

    assert err_hard < err_none, (
        f"cluster mode should move particles toward their own cluster mean: "
        f"none={err_none.item():.3f} hard={err_hard.item():.3f}"
    )


def test_public_api_runs_in_all_modes():
    torch.manual_seed(0)
    X, _ = _make_three_clusters(B=2, n_per=40)
    Y = X + 0.05 * torch.randn_like(X)

    for mode in ("none", "hard", "soft"):
        loss, info = drift_loss_ot(
            X, Y, Y,
            R_list=(0.05,),
            sinkhorn_num_iter=20,
            cluster_mode=mode,
            n_clusters=3,
            mask_lambda=1.0,
        )
        assert loss.shape == (2,), f"{mode}: bad loss shape {loss.shape}"
        assert torch.isfinite(loss).all(), f"{mode}: non-finite loss"
        assert "scale" in info


if __name__ == "__main__":
    test_off_block_leak_is_zero_in_hard_mode()
    print("[PASS] off-block leak = 0 in hard mode")
    test_cluster_mode_improves_own_cluster_targeting()
    print("[PASS] cluster mode improves own-cluster targeting")
    test_public_api_runs_in_all_modes()
    print("[PASS] public API runs in {none, hard, soft}")
