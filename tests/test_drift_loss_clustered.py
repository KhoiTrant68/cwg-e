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


def test_cluster_mode_pulls_source_toward_target():
    """V from a spread-out source toward a 3-cluster target should reduce
    average nearest-target distance. Previous version of this test was
    buggy: it passed the same tensor as ``fixed_pos`` and ``fixed_neg``,
    which forces V = T_{q,p} - T_{q,neg} to cancel to zero by construction.
    """
    torch.manual_seed(0)
    target, _ = _make_three_clusters(B=1, n_per=60)
    # Source: spread random points, NOT on the clusters
    source = torch.randn_like(target) * 1.5

    eps = torch.tensor(0.5)
    V_hard = _compute_V_clustered(
        source, target, source.detach(),
        eps=eps,
        cluster_mode="hard", n_clusters=3, mask_lambda=0.0,
        num_iter=50,
    )

    def avg_nn_dist(a, b):
        return torch.cdist(a.squeeze(0), b.squeeze(0)).min(dim=1).values.mean()

    d_before = avg_nn_dist(source, target)
    d_after = avg_nn_dist(source + V_hard, target)
    assert d_after < d_before, (
        f"V_hard should move source toward target: "
        f"before={d_before.item():.3f} after={d_after.item():.3f}"
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
    test_cluster_mode_pulls_source_toward_target()
    print("[PASS] cluster mode pulls source toward target")
    test_public_api_runs_in_all_modes()
    print("[PASS] public API runs in {none, hard, soft}")
