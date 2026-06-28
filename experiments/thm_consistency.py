"""Thm 1 — consistency under cluster separation.

For K well-separated Gaussian clusters with separation s, the cluster-wise
Sinkhorn estimator V_hard converges to the "ideal" V (estimated with a
large-batch reference) faster than the global Sinkhorn V_none as s grows.

Setup:
    - p: K=4 isotropic Gaussians, intra-cluster std 0.15, inter-cluster
      separation s ∈ [1, 6].
    - q: independent draw from the same p.
    - "Ideal" V_*  : computed from large pools (N = 4096) with cluster_mode=hard.
    - Estimator V̂ : computed from small mini-batch (N = 128) using
                    cluster_mode ∈ {none, hard}.
    - Error: mean ||V̂(x) - V_*(x)||^2 averaged over the small batch.

Claim: error(none) grows with s (target barycentric collapses into voids
between far-apart clusters); error(hard) stays small.

Usage (Kaggle):
    !python experiments/thm_consistency.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import device, out_dir
from drift_loss_ot import _compute_V_clustered  # noqa: E402
from clustering import batched_kmeans  # noqa: E402


def _make_clusters(n: int, K: int, sep: float, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    centres = np.stack([np.cos(ang), np.sin(ang)], 1) * sep
    idx = rng.integers(0, K, n)
    return centres[idx] + std * rng.standard_normal((n, 2))


def _V(q: torch.Tensor, p: torch.Tensor, cluster_mode: str, K: int,
       centroids: torch.Tensor | None = None,
       use_outer_gamma: bool = False) -> torch.Tensor:
    eps = torch.tensor(0.05 * (q.shape[-1] ** 0.5), device=q.device)
    return _compute_V_clustered(
        q[None], p[None], q[None].detach(),
        eps=eps,
        cluster_mode=cluster_mode if cluster_mode != "none" else "hard",
        n_clusters=1 if cluster_mode == "none" else K,
        mask_lambda=0.0,
        num_iter=30,
        cluster_centroids=centroids,
        use_per_cluster_sinkhorn=True,
        use_outer_gamma=use_outer_gamma,
        outer_gamma_eps=0.01,
    )[0]


def _ideal_V_at(q_small: torch.Tensor, p_big: torch.Tensor, K: int,
                centroids: torch.Tensor | None) -> torch.Tensor:
    """Compute V at the same q points using a large p reference (low-variance)."""
    return _V(q_small, p_big, "hard", K, centroids=centroids,
              use_outer_gamma=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-small", type=int, default=64,
                        help="small batch — must be small vs intra-cluster size for the bias to show")
    parser.add_argument("--n-big", type=int, default=4096)
    parser.add_argument("--seps", nargs="+", type=float,
                        default=[0.5, 0.8, 1.2, 1.6, 2.0, 3.0])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--intra-std", type=float, default=0.25,
                        help="larger intra-cluster std stresses small-batch global Sinkhorn")
    args = parser.parse_args()

    dev = device()
    od = out_dir()
    print(f"[thm_consistency] device={dev}  K={args.K}  "
          f"n_small={args.n_small}  n_big={args.n_big}")

    rows = []
    err_none = np.zeros(len(args.seps))
    err_hard = np.zeros(len(args.seps))
    err_outerG = np.zeros(len(args.seps))
    sd_none = np.zeros(len(args.seps))
    sd_hard = np.zeros(len(args.seps))
    sd_outerG = np.zeros(len(args.seps))

    for i, s in enumerate(args.seps):
        en, eh, eo = [], [], []
        for r in range(args.repeats):
            p_big = torch.as_tensor(
                _make_clusters(args.n_big, args.K, s, args.intra_std, seed=100 + r),
                dtype=torch.float32, device=dev,
            )
            q_small = torch.as_tensor(
                _make_clusters(args.n_small, args.K, s, args.intra_std, seed=200 + r),
                dtype=torch.float32, device=dev,
            )
            p_small = torch.as_tensor(
                _make_clusters(args.n_small, args.K, s, args.intra_std, seed=300 + r),
                dtype=torch.float32, device=dev,
            )

            # Shared centroids from p_big — both estimators see the same partition
            # (eliminates clustering noise as a confound; isolates the OT estimator).
            _, cents = batched_kmeans(p_big.unsqueeze(0), K=args.K, num_iter=30)
            cents_K = cents.squeeze(0)

            V_ideal = _ideal_V_at(q_small, p_big, args.K, centroids=cents_K)
            V_none = _V(q_small, p_small, "none", args.K, centroids=None)
            V_hard = _V(q_small, p_small, "hard", args.K, centroids=cents_K)
            V_outerG = _V(q_small, p_small, "hard", args.K, centroids=cents_K,
                          use_outer_gamma=True)

            en.append((V_none - V_ideal).pow(2).sum(-1).mean().item())
            eh.append((V_hard - V_ideal).pow(2).sum(-1).mean().item())
            eo.append((V_outerG - V_ideal).pow(2).sum(-1).mean().item())

        err_none[i] = float(np.mean(en)); sd_none[i] = float(np.std(en))
        err_hard[i] = float(np.mean(eh)); sd_hard[i] = float(np.std(eh))
        err_outerG[i] = float(np.mean(eo)); sd_outerG[i] = float(np.std(eo))
        rows.append(dict(sep=s,
                         err_none=round(err_none[i], 6),
                         err_hard=round(err_hard[i], 6),
                         err_outerG=round(err_outerG[i], 6)))
        print(f"[thm_consistency] sep={s:.1f}  none={err_none[i]:.4f}  "
              f"hard={err_hard[i]:.4f}  outerG={err_outerG[i]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(args.seps, err_none, yerr=sd_none, label="Global Sinkhorn (none)",
                color="#185FA5", marker="o", capsize=3)
    ax.errorbar(args.seps, err_hard, yerr=sd_hard, label=f"CWG-E hard (K={args.K})",
                color="#0F6E56", marker="o", capsize=3)
    ax.errorbar(args.seps, err_outerG, yerr=sd_outerG, label=f"CWG-E hard + outer Γ (K={args.K})",
                color="#7A55C9", marker="D", capsize=3)
    ax.set_xlabel("Inter-cluster separation s")
    ax.set_ylabel(r"$\|\hat V - V_\ast\|^2$")
    ax.set_yscale("log")
    ax.set_title("Thm 1: cluster-wise estimator stays consistent as separation grows")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig_path = od / "thm_consistency.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"[thm_consistency] figure -> {fig_path}")

    csv_path = od / "thm_consistency.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[thm_consistency] csv    -> {csv_path}")


if __name__ == "__main__":
    main()
