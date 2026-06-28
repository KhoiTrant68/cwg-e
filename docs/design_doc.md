# CWG-E — Implementation Design

This fork sits directly on top of W-Flow. The only behavioural change is
the new cluster-wise branch in `drift_loss_ot.py`; everything else
(training loop, sampling, FID eval) is untouched. Setting
`ot_kwargs.cluster_mode: "none"` in any config reproduces W-Flow
exactly — that is the baseline / ablation arm.

## 1. File-level deltas vs upstream W-Flow

| Path                                       | Status     | Purpose                                 |
|--------------------------------------------|------------|-----------------------------------------|
| `clustering.py`                            | **added**  | batched k-means utility                 |
| `drift_loss_ot.py`                         | **patched**| + cluster-wise branch (block-mask)      |
| `train.py`                                 | **patched**| pipe `cluster_mode` / `n_clusters` / `mask_lambda` |
| `models/energy_head.py`                    | **added**  | Gate-4 stubs (`EnergyHead`, `langevin_sample`) |
| `configs/gen/cwge_ablation_1node.yaml`     | **added**  | CWG-E ablation config (clone of `ablation_ot_1node.yaml`) |
| `scripts/train/cwge_ablation.sh`           | **added**  | launcher                                |
| `experiments/poc_cwg_e.py`                 | **added**  | 2D PoC (Gate 0)                         |
| `tests/test_drift_loss_clustered.py`       | **added**  | unit tests (Gate 1)                     |
| `docs/proposal.md`, `docs/design_doc.md`   | **added**  | research docs                           |

## 2. Flow

```
train.py
    │
    ├── rearrange  "b x f d -> (b f) x d"
    │
    ├── drift_loss_ot(..., cluster_mode=<from cfg>, n_clusters=..., mask_lambda=...)
    │       │
    │       ├── if cluster_mode == "none":  EXACT upstream W-Flow path
    │       │       (batched_multi_R OR sequential _compute_V_debiased)
    │       │
    │       └── else:  _compute_V_clustered
    │               ├── batched_kmeans on (z ⊕ pos ⊕ neg [⊕ uncond])
    │               ├── _sinkhorn_block (cost + block-mask)
    │               │       └── "hard": zero off-block of Π
    │               └── barycentric T_{q,p}, T_{q,neg} [, T_{q,uncond}]
    │
    └── sg-MSE against (z + ηV)
```

## 3. Block-mask details

Penalty added to the cost matrix (`drift_loss_ot.py::_block_penalty`):

- `"hard"`: `1e6` on off-block pairs. After Sinkhorn, Π is multiplied
  by the on-block indicator to remove the ~1% dual-potential leak.
- `"soft"`: `mask_lambda * ||c_i − c_j||²` (smooth cross-cluster routing).

A single batched k-means runs on `cat([z, pos, neg [, uncond]], dim=1)`
so every barycentric map sees the same partition.

## 4. Wiring details (`train.py`)

Added to `ot_loss_kwargs` (line ~237) — only three new keys, read from
the same `ot_kwargs` dict that already carries `sinkhorn_num_iter`, etc:

```python
ot_loss_kwargs["cluster_mode"] = _ot_kw.get("cluster_mode", "none")
ot_loss_kwargs["n_clusters"]   = _ot_kw.get("n_clusters", 8)
ot_loss_kwargs["mask_lambda"]  = _ot_kw.get("mask_lambda", 1.0)
```

Defaults match W-Flow behaviour exactly (no change unless a config opts in).

## 5. Energy head wiring (TODO Gate 4)

1. Build `EnergyHead` from generator backbone features.
2. Add `L_OT + L_CD` to total loss; reuse `_compute_V_clustered`'s Γ.
3. Generator-initialised Langevin negatives via `langevin_sample`.
4. Inference: gradient descent on `V_ψ + data_fidelity` for inverse;
   Hessian of `V_ψ` for LID.

## 6. Hyper-parameter cheat-sheet

| Setting          | uncond | cond | notes                              |
|------------------|--------|------|------------------------------------|
| `cluster_mode`   | hard   | soft | cond already has per-class OT; soft lets sub-class clustering help without splitting too thin |
| `n_clusters`     | 8–32   | 4–8  | floor at `max(1, min(K, N))`       |
| `mask_lambda`    | n/a    | 0.5–2.0 | only for `"soft"`               |
| `n_gen / n_pos`  | 256+   | 128+ | larger if clusters thin out        |
| `R_list`         | as upstream | — | unchanged                       |

## 7. Technical risks

- k-means convergence on early-training features — `num_iter=10` is
  usually enough; centroid sharing across z/pos/neg keeps them aligned.
- Empty clusters under `"hard"` — falls back to nearest-non-empty via
  the centroid fallback in `batched_kmeans` (TODO: explicit re-seeding
  or fall back to global Sinkhorn for that batch).
- `torch.compile` and dynamic mask shapes — set `DRIFT_COMPILE=0` if
  recompilation churn shows up in profiles.
- `batch_sinkhorn=True` is bypassed when `cluster_mode != "none"`; the
  cluster path uses sequential per-R Sinkhorn. Re-fold into the batched
  path once measured cost requires it.
