"""Gate 2 — train a one-step generator on 2D toys with each cluster_mode.

For three toy distributions (ring8, grid25, ring+minority), train a small
MLP one-step generator using ``drift_loss_ot`` in each
``cluster_mode in {none, hard, soft}``. Report final W2^2, mode coverage,
minority recall, and MMD. Save scatter plots.

Pass criterion (proposal §6):
    For ≥ 2 / 3 toys, hard (or soft) ≥ none on (W2^2 ↓) AND minority recall ↑.

Usage (Kaggle):
    !python experiments/gate2_train_2d.py
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    device, out_dir,
    sample_ring8, sample_grid25, sample_ring_with_minority,
    w2_squared, mmd_rbf, mode_coverage, minority_recall,
)

from drift_loss_ot import drift_loss_ot  # noqa: E402
from clustering import batched_kmeans  # noqa: E402


class MLP(nn.Module):
    def __init__(self, d_in=2, d_out=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, z):
        return self.net(z)


def _sample_real(toy: str, n: int, seed=None):
    if toy == "ring8":
        return sample_ring8(n, seed=seed), None, None
    if toy == "grid25":
        return sample_grid25(n, seed=seed), None, None
    if toy == "ring_minority":
        return sample_ring_with_minority(n, seed=seed)
    raise ValueError(toy)


def _train_one(
    toy: str,
    cluster_mode: str,
    n_clusters: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    dev: torch.device,
    use_outer_gamma: bool = False,
    cluster_pos: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    G = MLP().to(dev)
    opt = torch.optim.Adam(G.parameters(), lr=lr)

    # Sample a large reference pool of real points
    real_pool, centres, n_dom = _sample_real(toy, n=8192, seed=seed)
    real_pool_t = torch.as_tensor(real_pool, dtype=torch.float32, device=dev)

    # Pre-cluster the FIXED real pool once. Sticky centroids → consistent
    # partition across training steps → no clustering noise driving collapse.
    centroids_fixed = None
    if cluster_mode != "none":
        _, c = batched_kmeans(real_pool_t.unsqueeze(0), K=n_clusters, num_iter=30)
        centroids_fixed = c.squeeze(0)        # [K, D]

    # For training, hard mode uses BLOCK-MASK (not per-cluster): per-cluster
    # Sinkhorn produces V = 0 for points whose cluster is empty on the gen
    # side, which permanently traps mode-collapsed runs. Block-mask leaks a
    # small amount of mass through dual potentials and lets training recover.
    # outer-Gamma is only wired in the per-cluster path; enable it there.
    use_per_cluster_sinkhorn = bool(use_outer_gamma) and cluster_mode == "hard"

    t0 = time.time()
    for step in range(steps):
        z = torch.randn(batch, 2, device=dev)
        gen = G(z)

        # mini-batch of reals
        idx = torch.randint(0, len(real_pool_t), (batch,), device=dev)
        real = real_pool_t[idx]

        # drift_loss_ot expects [B, N, D]; here B=1, N=batch
        gen_b = gen.unsqueeze(0)
        real_b = real.unsqueeze(0)
        neg_b = gen_b.detach()

        loss, _ = drift_loss_ot(
            gen=gen_b,
            fixed_pos=real_b,
            fixed_neg=neg_b,
            R_list=(0.05,),
            sinkhorn_num_iter=20,
            disable_diag_mask=False,
            cluster_mode=cluster_mode,
            n_clusters=n_clusters,
            mask_lambda=1.0,
            cluster_centroids=centroids_fixed,
            use_per_cluster_sinkhorn=use_per_cluster_sinkhorn,
            use_outer_gamma=use_outer_gamma,
            cluster_pos=cluster_pos,
        )
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
    wall = time.time() - t0

    # Evaluation
    with torch.no_grad():
        z = torch.randn(2048, 2, device=dev)
        samples = G(z).cpu().numpy()
    real_eval, centres_eval, n_dom_eval = _sample_real(toy, n=2048, seed=seed + 1)

    metrics = dict(
        toy=toy,
        cluster_mode=cluster_mode,
        use_outer_gamma=use_outer_gamma,
        cluster_pos=cluster_pos,
        n_clusters=n_clusters,
        steps=steps,
        wall_s=round(wall, 2),
        w2_sq=round(w2_squared(samples, real_eval, max_n=1024), 4),
        mmd=round(mmd_rbf(samples, real_eval), 4),
    )

    if centres is not None:
        cov, counts = mode_coverage(samples, centres)
        metrics["mode_coverage"] = round(cov, 3)
        metrics["minority_recall"] = round(
            minority_recall(samples, centres, n_dom=n_dom), 3,
        )
    else:
        # Estimate "centres" from the real pool's nearest neighbors via the toy's
        # native helper would require special-casing — for ring8/grid25 we get
        # them analytically here:
        if toy == "ring8":
            ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            centres_e = np.stack([np.cos(ang), np.sin(ang)], 1) * 2.0
        else:  # grid25
            xs, ys = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
            centres_e = np.stack([xs.flatten(), ys.flatten()], 1)
        cov, _ = mode_coverage(samples, centres_e)
        metrics["mode_coverage"] = round(cov, 3)
        metrics["minority_recall"] = float("nan")

    return metrics, samples, (centres if centres is not None else None), (n_dom if n_dom is not None else None)


def _scatter(ax, samples, centres, n_dom, title):
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.4, c="#0F6E56", label="gen")
    if centres is not None:
        if n_dom is not None and n_dom < len(centres):
            ax.scatter(centres[:n_dom, 0], centres[:n_dom, 1], s=80, marker="X",
                       c="#185FA5", label="dominant", zorder=5)
            ax.scatter(centres[n_dom:, 0], centres[n_dom:, 1], s=120, marker="*",
                       c="#D85A30", label="minority", zorder=5)
        else:
            ax.scatter(centres[:, 0], centres[:, 1], s=80, marker="X",
                       c="#185FA5", label="modes", zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)


# variant name -> (cluster_mode, use_outer_gamma, cluster_pos)
VARIANTS = {
    "none":           ("none", False, True),
    "hard":           ("hard", False, True),
    "soft":           ("soft", False, True),
    "hard_outerG":    ("hard", True,  True),   # only variant with non-degenerate field (thm_no_spurious)
    "hard_globalpos": ("hard", False, False),  # global T_pq + clustered T_qneg
    "soft_globalpos": ("soft", False, False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--toys", nargs="+",
                        default=["ring8", "grid25", "ring_minority"])
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    args = parser.parse_args()

    dev = device()
    print(f"[gate2-v2] device={dev}  steps={args.steps}  batch={args.batch}")
    print(f"[gate2-v2] variants: {args.variants}")

    od = out_dir()
    rows = []

    fig, axes = plt.subplots(len(args.toys), len(args.variants),
                             figsize=(3.4 * len(args.variants), 3.4 * len(args.toys)),
                             squeeze=False)

    for i, toy in enumerate(args.toys):
        for j, vname in enumerate(args.variants):
            cmode, oG, cpos = VARIANTS[vname]
            m, samples, centres, n_dom = _train_one(
                toy=toy, cluster_mode=cmode, n_clusters=args.n_clusters,
                steps=args.steps, batch=args.batch, lr=args.lr,
                seed=args.seed, dev=dev,
                use_outer_gamma=oG, cluster_pos=cpos,
            )
            m["variant"] = vname
            print(f"[gate2-v2] toy={toy:14s}  variant={vname:14s}  "
                  f"W2^2={m['w2_sq']:.4f}  cov={m['mode_coverage']:.3f}  "
                  f"min={m['minority_recall']}  mmd={m['mmd']:.4f}")
            rows.append(m)
            _scatter(axes[i][j], samples, centres, n_dom,
                     f"{toy} / {vname}\nW2^2={m['w2_sq']:.3f}  cov={m['mode_coverage']:.2f}")

    plt.suptitle("Gate 2 v2 — one-step generator, variant sweep", fontsize=11)
    plt.tight_layout()
    fig_path = od / "gate2_v2_samples.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"[gate2-v2] figure -> {fig_path}")

    csv_path = od / "gate2_v2_metrics.csv"
    # union of keys across rows (variants differ in fields)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[gate2-v2] metrics -> {csv_path}")


if __name__ == "__main__":
    main()