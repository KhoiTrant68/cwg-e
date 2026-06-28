"""Wall-clock cost of drift estimation: cluster vs global as N and K vary.

Sweep mini-batch size N and report ms / drift call for each
``cluster_mode`` and ``n_clusters``. Useful for the proposal's "rẻ 10–20×"
claim (which only holds when the global Sinkhorn over-batches; clustering
splits the same N×N problem into K smaller blocks).

Usage (Kaggle):
    !python experiments/bench_cost.py
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import device, out_dir, sample_ring8
from drift_loss_ot import _compute_V_clustered  # noqa: E402
from clustering import batched_kmeans  # noqa: E402


def _bench_one(q: torch.Tensor, p: torch.Tensor, mode: str, K: int,
               num_iter: int, centroids: torch.Tensor | None,
               warmup: int = 2, repeats: int = 5) -> float:
    eps = torch.tensor(0.05 * (q.shape[-1] ** 0.5), device=q.device)

    def call():
        return _compute_V_clustered(
            q[None], p[None], q[None].detach(),
            eps=eps,
            cluster_mode=mode if mode != "none" else "hard",
            n_clusters=1 if mode == "none" else K,
            mask_lambda=0.0,
            num_iter=num_iter,
            cluster_centroids=centroids,         # sticky (no k-means each call)
            use_per_cluster_sinkhorn=True,        # real K-fold speedup for hard
        )

    for _ in range(warmup):
        call()
    if q.is_cuda:
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(repeats):
        call()
    if q.is_cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000.0  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", nargs="+", type=int,
                        default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--Ks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=20)
    args = parser.parse_args()

    dev = device()
    od = out_dir()
    print(f"[bench_cost] device={dev}  num_iter={args.num_iter}")

    rows = []
    series = {"none": [], **{f"hard_K{K}": [] for K in args.Ks}}

    for N in args.Ns:
        p = torch.as_tensor(sample_ring8(N, seed=42),
                            dtype=torch.float32, device=dev)
        q = torch.randn(N, 2, device=dev) * 0.5

        # Sticky centroids: cluster p once per N, reused across the timed calls
        cents_by_K = {K: batched_kmeans(p.unsqueeze(0), K=K, num_iter=20)[1].squeeze(0)
                      for K in args.Ks}

        ms_none = _bench_one(q, p, "none", 1, args.num_iter,
                             centroids=None, repeats=args.repeats)
        series["none"].append(ms_none)
        row = dict(N=N, none_ms=round(ms_none, 3))
        for K in args.Ks:
            ms_hard = _bench_one(q, p, "hard", K, args.num_iter,
                                 centroids=cents_by_K[K], repeats=args.repeats)
            series[f"hard_K{K}"].append(ms_hard)
            row[f"hard_K{K}_ms"] = round(ms_hard, 3)
        rows.append(row)
        print(f"[bench_cost] N={N:5d}  " +
              "  ".join(f"{k}={v:.2f}ms" for k, v in
                        [("none", ms_none)] + [(f"K{K}", row[f"hard_K{K}_ms"]) for K in args.Ks]))

    # plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"none": "#185FA5"}
    palette = ["#0F6E56", "#D85A30", "#7A55C9", "#C99055"]
    for j, K in enumerate(args.Ks):
        colors[f"hard_K{K}"] = palette[j % len(palette)]
    for name, ys in series.items():
        ax.plot(args.Ns, ys, marker="o", label=name, color=colors[name])
    ax.set_xlabel("mini-batch size N")
    ax.set_ylabel("ms / drift call")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Drift cost vs N — cluster-wise vs global Sinkhorn")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig_path = od / "bench_cost.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"[bench_cost] figure -> {fig_path}")

    csv_path = od / "bench_cost.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[bench_cost] csv    -> {csv_path}")


if __name__ == "__main__":
    main()
