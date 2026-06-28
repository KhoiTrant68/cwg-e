"""Prop 1 — variance reduction.

Fixed q (initial generator samples) and a large pool p. Draw M mini-batches
of p (each size N) and compute the drift estimate V̂ for each batch using:
    cluster_mode ∈ {none, hard, soft}
Report mean ||V̂ - mean(V̂)||^2 across draws — i.e., the estimator variance.

Claim (proposal §5): cluster-wise estimators have strictly lower variance
than the global estimator when the target has clear cluster structure,
because most spurious cross-mode pairings are removed by the block-mask.

Usage (Kaggle):
    !python experiments/prop_variance.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import device, out_dir, sample_ring8
from drift_loss_ot import _compute_V_clustered, _cluster_marginals  # noqa: E402
from clustering import batched_kmeans, assign_to_centroids  # noqa: E402


def _V(q: torch.Tensor, p: torch.Tensor, cluster_mode: str, K: int,
       mask_lambda: float,
       centroids: torch.Tensor | None = None,
       use_outer_gamma: bool = False,
       marg_p_pool: torch.Tensor | None = None) -> torch.Tensor:
    eps = torch.tensor(0.05 * (q.shape[-1] ** 0.5), device=q.device)
    return _compute_V_clustered(
        q[None], p[None], q[None].detach(),
        eps=eps,
        cluster_mode=cluster_mode if cluster_mode != "none" else "hard",
        n_clusters=1 if cluster_mode == "none" else K,
        mask_lambda=mask_lambda,
        num_iter=30,
        cluster_centroids=centroids,
        use_per_cluster_sinkhorn=True,
        use_outer_gamma=use_outer_gamma,
        outer_gamma_eps=0.01,
        cluster_marg_p_pool=marg_p_pool,
    )[0]


def _variance(V_stack: torch.Tensor) -> float:
    """V_stack: [M, N, D]; return mean per-point variance across the M draws."""
    mu = V_stack.mean(dim=0, keepdim=True)              # [1, N, D]
    return (V_stack - mu).pow(2).sum(-1).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256, help="mini-batch size of p")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--n-draws", type=int, default=40)
    parser.add_argument("--seed-q", type=int, default=0)
    parser.add_argument("--pool", type=int, default=8192)
    args = parser.parse_args()

    dev = device()
    od = out_dir()
    print(f"[prop_variance] device={dev}  n={args.n}  M={args.n_draws}  "
          f"K={args.n_clusters}")

    # Fixed q
    torch.manual_seed(args.seed_q)
    q = torch.randn(args.n, 2, device=dev) * 0.5    # spread-out source
    # Large reference pool of p
    p_pool = torch.as_tensor(sample_ring8(args.pool, seed=42),
                             dtype=torch.float32, device=dev)

    # STICKY CENTROIDS: cluster the pool once, reuse across all mini-batch draws.
    # This is the fix: without it, each batch re-clusters and the partition
    # drift adds noise to V (inflating estimator variance).
    _, centroids_pool = batched_kmeans(
        p_pool.unsqueeze(0), K=args.n_clusters, num_iter=20,
    )                                                # [1, K, D]
    centroids_fixed = centroids_pool.squeeze(0)      # [K, D]

    # Pool-level cluster marginal for p — used by outer Γ so it doesn't
    # absorb mini-batch noise (which would erase the variance win).
    labels_pool = assign_to_centroids(p_pool.unsqueeze(0), centroids_pool)
    marg_p_pool = _cluster_marginals(labels_pool, args.n_clusters).squeeze(0)  # [K]

    # Last tuple element: use_outer_gamma flag
    modes = [
        ("none",         1,                0.0, False),
        ("hard",         args.n_clusters,  0.0, False),
        ("soft",         args.n_clusters,  1.0, False),
        ("hard_outerG",  args.n_clusters,  0.0, True),
    ]
    stacks: dict[str, list[torch.Tensor]] = {m[0]: [] for m in modes}

    for d in range(args.n_draws):
        idx = torch.randint(0, args.pool, (args.n,), device=dev)
        p_batch = p_pool[idx]
        for mode, K, lam, og in modes:
            cents = None if mode == "none" else centroids_fixed
            inner_mode = "hard" if mode in ("hard", "hard_outerG") else mode
            mpool = marg_p_pool if og else None
            V = _V(q, p_batch, inner_mode, K, lam, centroids=cents,
                   use_outer_gamma=og, marg_p_pool=mpool)
            stacks[mode].append(V)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_x = np.arange(len(modes))
    bar_h, bar_e = [], []

    for mode, K, lam, og in modes:
        Vs = torch.stack(stacks[mode], dim=0)
        var = _variance(Vs)
        norm = Vs.mean(dim=0).pow(2).sum(-1).mean().item()
        snr = norm / max(var, 1e-12)
        rows.append(dict(mode=mode, K=K, mask_lambda=lam,
                         use_outer_gamma=og,
                         variance=round(var, 6),
                         signal_norm=round(norm, 6),
                         snr=round(snr, 3)))
        bar_h.append(var)
        bar_e.append(0.0)
        print(f"[prop_variance] mode={mode:12s}  var={var:.5f}  "
              f"signal={norm:.5f}  SNR={snr:.2f}")

    colors = {"none": "#185FA5", "hard": "#0F6E56",
              "soft": "#D85A30", "hard_outerG": "#7A55C9"}
    ax.bar(bar_x, bar_h, color=[colors[m[0]] for m in modes])
    ax.set_xticks(bar_x)
    ax.set_xticklabels([m[0] for m in modes])
    ax.set_ylabel("estimator variance")
    ax.set_title(f"Prop 1: V-estimator variance across {args.n_draws} mini-batch draws")
    for x, h in zip(bar_x, bar_h):
        ax.text(x, h, f"{h:.3g}", ha="center", va="bottom", fontsize=9)
    fig_path = od / "prop_variance.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"[prop_variance] figure -> {fig_path}")

    csv_path = od / "prop_variance.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[prop_variance] csv    -> {csv_path}")


if __name__ == "__main__":
    main()
