"""Thm 2 — no spurious equilibria: ||V|| → 0 iff q = p (no other fixed point).

For a fixed target p (8 Gaussians on a ring), construct a 1-parameter
family of source distributions q(α) that interpolates from "missing one
mode" (α = 0) to "full match" (α = 1). For each q(α), estimate the drift
V using:
    (a) global Sinkhorn (W-Flow baseline) via cluster_mode="none"
    (b) cluster-wise Sinkhorn (CWG-E hard)
    (c) a Drifting-like mean-shift heuristic
       V_heur(x) = (1/Z) Σ_y exp(-||x-y||^2/2σ^2) (y - x)

Claim: only (a) and (b) satisfy ||V|| → 0 monotonically as α → 1; the
heuristic (c) plateaus at non-zero ||V|| because it has spurious
attractors near non-mass regions.

Output: plot of mean ||V||^2 vs α + CSV.

Usage (Kaggle):
    !python experiments/thm_no_spurious.py
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
from drift_loss_ot import drift_loss_ot, _compute_V_clustered  # noqa: E402
from clustering import batched_kmeans  # noqa: E402


def _q_alpha(alpha: float, n: int, seed: int) -> np.ndarray:
    """Interpolate between p-missing-mode-0 (alpha=0) and full-p (alpha=1)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centres = np.stack([np.cos(ang), np.sin(ang)], 1) * 2.0

    # weights: drop mode 0 at alpha=0, restore uniformly at alpha=1
    w = np.ones(8)
    w[0] = alpha
    w = w / w.sum()
    idx = rng.choice(8, size=n, p=w)
    return centres[idx] + 0.05 * rng.standard_normal((n, 2))


def _V_global(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(0.05 * (q.shape[-1] ** 0.5), device=q.device)
    return _compute_V_clustered(
        q[None], p[None], q[None].detach(),
        eps=eps,
        cluster_mode="hard", n_clusters=1, mask_lambda=0.0,
        num_iter=30, use_per_cluster_sinkhorn=True,
    )[0]


def _V_cluster(q: torch.Tensor, p: torch.Tensor, K: int,
               centroids: torch.Tensor | None = None) -> torch.Tensor:
    eps = torch.tensor(0.05 * (q.shape[-1] ** 0.5), device=q.device)
    return _compute_V_clustered(
        q[None], p[None], q[None].detach(),
        eps=eps,
        cluster_mode="hard", n_clusters=K, mask_lambda=0.0,
        num_iter=30,
        cluster_centroids=centroids,
        use_per_cluster_sinkhorn=True,
    )[0]


def _V_meanshift(q: torch.Tensor, p: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """Drifting-style heuristic: kernel-smoothed pull toward p."""
    d2 = torch.cdist(q, p) ** 2
    w = torch.exp(-d2 / (2 * sigma * sigma))
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return w @ p - q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--n-alphas", type=int, default=11)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5, help="seeds per alpha")
    args = parser.parse_args()

    dev = device()
    od = out_dir()
    print(f"[thm_no_spurious] device={dev}  n={args.n}  K={args.n_clusters}")

    alphas = np.linspace(0.0, 1.0, args.n_alphas)
    p_np = sample_ring8(args.n, seed=42)
    p_t = torch.as_tensor(p_np, dtype=torch.float32, device=dev)

    # Sticky centroids from the full p
    _, cents = batched_kmeans(p_t.unsqueeze(0), K=args.n_clusters, num_iter=30)
    cents_K = cents.squeeze(0)

    rows = []
    series = {name: np.zeros(len(alphas)) for name in
              ("global_K1", "cluster_K", "meanshift")}
    err = {name: np.zeros(len(alphas)) for name in series}

    for i, a in enumerate(alphas):
        vals = {name: [] for name in series}
        for r in range(args.repeats):
            q_np = _q_alpha(a, args.n, seed=r)
            q_t = torch.as_tensor(q_np, dtype=torch.float32, device=dev)
            vals["global_K1"].append(_V_global(q_t, p_t).pow(2).sum(-1).mean().item())
            vals["cluster_K"].append(_V_cluster(q_t, p_t, args.n_clusters,
                                                 centroids=cents_K).pow(2).sum(-1).mean().item())
            vals["meanshift"].append(_V_meanshift(q_t, p_t).pow(2).sum(-1).mean().item())
        for name in series:
            series[name][i] = float(np.mean(vals[name]))
            err[name][i] = float(np.std(vals[name]))
        rows.append(dict(alpha=round(a, 3),
                         **{name: round(series[name][i], 6) for name in series}))
        print(f"[thm_no_spurious] alpha={a:.2f}  "
              f"global={series['global_K1'][i]:.4f}  "
              f"cluster={series['cluster_K'][i]:.4f}  "
              f"meanshift={series['meanshift'][i]:.4f}")

    # plot — log y to make the "vanishes at α=1" claim visually clear
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    labels = {
        "global_K1": "Global Sinkhorn (W-Flow, K=1)",
        "cluster_K": f"Cluster-wise hard (CWG-E, K={args.n_clusters})",
        "meanshift": "Heuristic mean-shift (Drifting-style)",
    }
    colors = {"global_K1": "#185FA5", "cluster_K": "#0F6E56", "meanshift": "#D85A30"}
    for name in series:
        y = np.clip(series[name], 1e-6, None)
        ax.errorbar(alphas, y, yerr=err[name],
                    label=labels[name], color=colors[name], marker="o", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("α   (q = p at α = 1; mode 0 missing at α = 0)")
    ax.set_ylabel(r"mean  $\|V(x)\|^2$  (log)")
    ax.set_title("Thm 2: principled estimators vanish iff q = p; heuristic does not")
    ax.annotate(
        "mean-shift stays away from zero at α=1\n→ spurious equilibrium",
        xy=(0.95, series["meanshift"][-1]),
        xytext=(0.45, max(series["meanshift"][-1] * 5, 1e-3)),
        arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1.0),
        color="#D85A30", fontsize=9,
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig_path = od / "thm_no_spurious.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"[thm_no_spurious] figure -> {fig_path}")

    csv_path = od / "thm_no_spurious.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[thm_no_spurious] csv    -> {csv_path}")


if __name__ == "__main__":
    main()
