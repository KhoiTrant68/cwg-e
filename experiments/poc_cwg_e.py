# === Kaggle cell: PoC cluster-wise vs global Sinkhorn drift + figure ===
import sys, subprocess
try:
    import ot
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "POT"])
    import ot
import numpy as np, time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def make_target(n_dom=6, N=4000, minority_frac=0.03, std=0.08, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2*np.pi, n_dom, endpoint=False)
    dom = np.stack([np.cos(ang), np.sin(ang)], 1) * 2.0
    minor = np.array([[0.0, 0.0], [0.0, 1.0]])
    centers = np.vstack([dom, minor])
    nm = max(2, int(N*minority_frac/len(minor))); nd = (N - nm*len(minor))//n_dom
    pts = [c + std*rng.standard_normal((nd, 2)) for c in dom]
    pts += [c + std*rng.standard_normal((nm, 2)) for c in minor]
    return np.vstack(pts), centers, n_dom


def sinkhorn_bary(xs, ys, eps=0.05):
    if len(xs) == 0 or len(ys) < 2: return xs.copy()
    a, b = np.ones(len(xs))/len(xs), np.ones(len(ys))/len(ys)
    M = ot.dist(xs, ys, "sqeuclidean"); M /= M.max()+1e-12
    P = ot.sinkhorn(a, b, M, eps)
    return (P @ ys) / (P.sum(1, keepdims=True)+1e-12)


def global_drift(xs, p, eps=0.05):
    return sinkhorn_bary(xs, p, eps) - xs


def cluster_drift(xs, p, plab, K, eps=0.05):
    cents = np.stack([p[plab == k].mean(0) for k in range(K)])
    asg = np.argmin(((xs[:, None] - cents[None])**2).sum(-1), 1)
    T = np.zeros_like(xs)
    for k in range(K):
        idx = np.where(asg == k)[0]
        T[idx] = sinkhorn_bary(xs[idx], p[plab == k], eps)
    return T - xs


# --- chuẩn bị dữ liệu + cụm ---
eps, K = 0.05, 8
p, centers, n_dom = make_target()
plab = KMeans(K, n_init=5, random_state=0).fit(p).labels_

# --- chỉ số định lượng ---
def nmd(T): return np.sqrt(((T[:, None] - centers[None])**2).sum(-1)).min(1)

xs = 0.9*np.random.default_rng(1).standard_normal((2000, 2))
tm = np.argmin(((xs[:, None]-centers[None])**2).sum(-1), 1); m = tm >= n_dom
for name, dr in [("global", global_drift(xs, p, eps)),
                 ("cluster", cluster_drift(xs, p, plab, K, eps))]:
    T = xs + dr; t2 = np.argmin(((T[:, None]-centers[None])**2).sum(-1), 1)
    print(f"{name:8s}  dist→mode={nmd(T).mean():.3f}  void={ (nmd(T)>0.3).mean():.3f}  "
          f"minority_retention={(t2[m]>=n_dom).mean():.3f}")

# --- VẼ FIGURE (đây là phần code vẽ poc drift) ---
xs = 0.9*np.random.default_rng(3).standard_normal((150, 2))
dg, dc = global_drift(xs, p, eps), cluster_drift(xs, p, plab, K, eps)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
for a, (dr, ti) in zip(ax, [(dg, "Global mini-batch Sinkhorn"),
                            (dc, "Cluster-wise Sinkhorn (ours)")]):
    a.scatter(p[:, 0], p[:, 1], s=3, c="#B4B2A9", alpha=.5, label="target $p$")
    a.scatter(centers[:n_dom, 0], centers[:n_dom, 1], s=70, marker="X",
              c="#185FA5", label="dominant modes", zorder=5)
    a.scatter(centers[n_dom:, 0], centers[n_dom:, 1], s=120, marker="*",
              c="#D85A30", label="minority modes", zorder=5)
    a.quiver(xs[:, 0], xs[:, 1], dr[:, 0], dr[:, 1], angles="xy",
             scale_units="xy", scale=1, width=.004, color="#0F6E56", alpha=.7)
    a.set_title(ti); a.set_aspect("equal"); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
ax[0].legend(loc="lower left", fontsize=8, framealpha=.9)
plt.suptitle("Drift targets of $q$-particles (green arrows): "
             "global collapses to voids; clustering stays on-mode", fontsize=10)
plt.tight_layout()
plt.savefig("poc_drift_comparison.png", dpi=130, bbox_inches="tight")
plt.show()
