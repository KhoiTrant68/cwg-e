"""Shared helpers for the 2D CWG-E experiments.

Bootstraps sys.path so scripts can ``from drift_loss_ot import drift_loss_ot``
when run either from the repo root or from ``experiments/``. Provides 2D
toy samplers, metrics (W2^2, MMD, mode coverage, minority recall), and a
small device / output-dir helper.

Auto-installs POT on Kaggle if missing.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# --- path bootstrap (so `from drift_loss_ot import ...` works) -----------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# disable torch.compile by default for these tiny 2D problems
os.environ.setdefault("DRIFT_COMPILE", "0")

# --- dependency bootstrap (idempotent on Kaggle) -------------------------
def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

_ensure("POT", "ot")
_ensure("scikit-learn", "sklearn")

import numpy as np
import torch
import ot

# --- device / output ------------------------------------------------------
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def out_dir() -> Path:
    """Use /kaggle/working/cwge_out on Kaggle, otherwise ./out next to repo."""
    kaggle = Path("/kaggle/working")
    base = kaggle if kaggle.exists() else (_REPO_ROOT / "out")
    d = base / "cwge_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- 2D toy distributions -------------------------------------------------
def sample_ring8(n: int, std: float = 0.05, seed: int | None = None) -> np.ndarray:
    """8 Gaussians on a unit-radius ring."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centres = np.stack([np.cos(ang), np.sin(ang)], 1) * 2.0
    idx = rng.integers(0, 8, n)
    return centres[idx] + std * rng.standard_normal((n, 2))


def sample_grid25(n: int, std: float = 0.05, seed: int | None = None) -> np.ndarray:
    """5x5 grid of Gaussians on [-2, 2]^2."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    centres = np.stack([xs.flatten(), ys.flatten()], 1)
    idx = rng.integers(0, 25, n)
    return centres[idx] + std * rng.standard_normal((n, 2))


def sample_ring_with_minority(
    n: int,
    n_dom: int = 6,
    minority_frac: float = 0.03,
    std: float = 0.08,
    seed: int | None = None,
):
    """Dominant ring of n_dom modes + 2 minority modes near origin."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_dom, endpoint=False)
    dom = np.stack([np.cos(ang), np.sin(ang)], 1) * 2.0
    minor = np.array([[0.0, 0.0], [0.0, 1.0]])
    centres = np.vstack([dom, minor])
    nm = max(2, int(n * minority_frac / len(minor)))
    nd = (n - nm * len(minor)) // n_dom
    pts = [c + std * rng.standard_normal((nd, 2)) for c in dom]
    pts += [c + std * rng.standard_normal((nm, 2)) for c in minor]
    return np.vstack(pts), centres, n_dom  # (samples, centres, idx-where-minority-starts)


TOY_REGISTRY: dict = {
    "ring8": dict(sample=lambda n, seed=None: sample_ring8(n, seed=seed),
                  centres=None, n_dom=None),
    "grid25": dict(sample=lambda n, seed=None: sample_grid25(n, seed=seed),
                   centres=None, n_dom=None),
    "ring_minority": dict(sample=None, centres=None, n_dom=None),  # uses helper directly
}


# --- metrics --------------------------------------------------------------
def w2_squared(x: np.ndarray, y: np.ndarray, max_n: int = 1024) -> float:
    """Exact W_2^2 via POT. Subsamples if too large to keep cost manageable."""
    if len(x) > max_n:
        x = x[np.random.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[np.random.choice(len(y), max_n, replace=False)]
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = ot.dist(x, y, "sqeuclidean")
    return float(ot.emd2(a, b, M, numItermax=200000))


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float = 0.5) -> float:
    """MMD^2 with an RBF kernel (median-heuristic-ish bandwidth)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    s2 = 2 * sigma * sigma
    return float(
        torch.exp(-xx / s2).mean()
        + torch.exp(-yy / s2).mean()
        - 2 * torch.exp(-xy / s2).mean()
    )


def mode_coverage(samples: np.ndarray, centres: np.ndarray, radius: float = 0.3):
    """Returns (coverage_fraction, per_mode_counts) — # generated within radius of each centre."""
    d = ((samples[:, None] - centres[None]) ** 2).sum(-1)
    nearest = d.argmin(1)
    within = (d.min(1) ** 0.5) < radius
    counts = np.zeros(len(centres), dtype=int)
    for m in range(len(centres)):
        counts[m] = int(((nearest == m) & within).sum())
    return float((counts > 0).mean()), counts


def minority_recall(samples: np.ndarray, centres: np.ndarray, n_dom: int, radius: float = 0.3):
    """Fraction of samples that landed on a minority-mode centre."""
    if n_dom is None or n_dom >= len(centres):
        return float("nan")
    d = ((samples[:, None] - centres[None]) ** 2).sum(-1)
    nearest = d.argmin(1)
    within = (d.min(1) ** 0.5) < radius
    on_minority = within & (nearest >= n_dom)
    return float(on_minority.mean())


# --- pretty printing ------------------------------------------------------
def fmt_row(*cells, widths=(14, 10, 10, 10, 10, 10)):
    return " ".join(str(c).ljust(w) for c, w in zip(cells, widths))
