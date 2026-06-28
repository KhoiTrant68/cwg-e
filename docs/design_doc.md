# CWG-E — Implementation Design

This fork sits directly on top of W-Flow. The only behavioural change is
the new cluster-wise branch in `drift_loss_ot.py`; the training loop,
sampling, and FID evaluation are untouched. Setting
`ot_kwargs.cluster_mode: "none"` in any config reproduces W-Flow
bit-for-bit — that is the ablation arm.

## 1. File-level deltas vs upstream W-Flow

| Path | Status | Purpose |
|---|---|---|
| `clustering.py` | **added** | `batched_kmeans`, `assign_to_centroids`, `expand_centroids` |
| `drift_loss_ot.py` | **patched** | per-cluster Sinkhorn + outer Γ + sticky-marg_p path |
| `train.py` | **patched** | pipe new kwargs from `ot_kwargs` into `drift_loss_ot` |
| `models/energy_head.py` | **added** | Gate-4 stub (`EnergyHead`, `langevin_sample`) |
| `configs/gen/cwge_ablation_1node.yaml` | **added** | clone of `ablation_ot_1node.yaml` + cluster kwargs |
| `scripts/train/cwge_ablation.sh` | **added** | launcher (requires full W-Flow data pipeline) |
| `experiments/` | **added** | 7 scripts (Gate 0 / Gate 2 / 3 theory / bench / runner) |
| `tests/test_drift_loss_clustered.py` | **added** | 3 unit tests |
| `docs/proposal.md`, `docs/design_doc.md` | **added** | research / implementation docs |
| `cwge_out_3/` | **added** | validated benchmark outputs |

## 2. Flow

```
train.py
    │
    ├── rearrange  "b x f d -> (b f) x d"
    │
    ├── drift_loss_ot(...,
    │                 cluster_mode=<from cfg>,
    │                 n_clusters=K,
    │                 cluster_centroids=<sticky, precomputed>,
    │                 cluster_marg_p_pool=<sticky, precomputed>,
    │                 use_outer_gamma=True,
    │                 outer_gamma_eps=0.01,
    │                 use_per_cluster_sinkhorn=True)
    │       │
    │       ├── if cluster_mode == "none":  EXACT upstream W-Flow path
    │       │       (batched_multi_R OR sequential _compute_V_debiased)
    │       │
    │       └── else (cluster_mode in {"hard","soft"}): _compute_V_clustered
    │               │
    │               ├── if cluster_centroids given:
    │               │       labels via assign_to_centroids (no Lloyd, sticky)
    │               │   else:
    │               │       batched_kmeans on (z ⊕ pos ⊕ neg [⊕ uncond])
    │               │
    │               ├── if cluster_mode == "hard" AND use_per_cluster_sinkhorn:
    │               │       _per_cluster_bary  (K problems batched in 1 Sinkhorn call)
    │               │   else:
    │               │       block-mask Sinkhorn  (single batched call, +1e6 off-block)
    │               │
    │               ├── if use_outer_gamma:
    │               │       marg_q = bincount(labels_z) / N
    │               │       marg_p = cluster_marg_p_pool  (sticky)  or  bincount(labels_p)/M
    │               │       Γ = sinkhorn(C_centroids, marg_q, marg_p, eps_gamma)
    │               │       β, off_target = decompose(Γ)
    │               │       T̃ = T_within + (1-β_k)(off_target_k - x_i)   # additive
    │               │
    │               └── V = T̃_pq - T̃_qneg [+ cfg_weight·(T̃_pq - T̃_quncond)]
    │
    └── sg-MSE against (z + ηV)
```

## 3. Per-cluster Sinkhorn (`_per_cluster_bary`)

Batched implementation: gather padded tensors `[K, max_nz, D]` and
`[K, max_ms, D]`, route a single call through `_sinkhorn_batched`
(B'=K). Empty support clusters get a dummy weight then their rows of T
are discarded via the nonempty mask. Total Sinkhorn cost ≈
`K · max_nz · max_ms` (≈ N²/K balanced), vs N² for block-mask.

Real wall-clock speedup at moderate-to-large N: **3.3× at N=8192, K=4**
(see `cwge_out_3/bench_cost.csv`).

## 4. Outer Γ correction (`_outer_gamma_targets` + `_apply_outer_gamma`)

The centroid-to-centroid Sinkhorn is computed with `_sinkhorn_AB`
(log-domain, both marginals, zero-safe). The decomposition gives
per-cluster diagonal weight β and off-cluster target:

```python
β_k        = Γ_norm[:, k, k]                           # [B, K]
off_k      = (Γ_off @ centroids) / off_sum             # [B, K, D]
T̃(x_i)    = T_within(x_i) + (1 - β_{k_i}) · (off_{k_i} - x_i)
```

The **additive** form (not convex combination) is critical: at q=p the
marginals match → Γ ≈ I → β = 1 → correction = 0 → T̃ = T_within and V
vanishes. A convex form `β·T_within + (1-β)·off_target` would shrink T̃
toward off_target = 0 even with tiny off-coupling, leaving a constant
residual V ≈ (β-1)·x_i at q=p (we hit this — see commit `32f919f`).

**Sticky `marg_p` is required** (`cluster_marg_p_pool`). Without it, the
per-mini-batch `marg_p` injects ~11% per-cluster noise into Γ, which
in turn injects variance into the correction, erasing the Prop 1 win
(commit `3a3e8d4` fix).

## 5. Wiring (`train.py`)

Added to `ot_loss_kwargs` (around line 237). Reads from `ot_kwargs`
config dict that already carries `sinkhorn_num_iter`, etc:

```python
ot_loss_kwargs["cluster_mode"]            = _ot_kw.get("cluster_mode", "none")
ot_loss_kwargs["n_clusters"]              = _ot_kw.get("n_clusters", 8)
ot_loss_kwargs["mask_lambda"]             = _ot_kw.get("mask_lambda", 1.0)
ot_loss_kwargs["use_outer_gamma"]         = _ot_kw.get("use_outer_gamma", False)
ot_loss_kwargs["outer_gamma_eps"]         = _ot_kw.get("outer_gamma_eps", 0.01)
ot_loss_kwargs["outer_gamma_iter"]        = _ot_kw.get("outer_gamma_iter", 200)
ot_loss_kwargs["use_per_cluster_sinkhorn"]= _ot_kw.get("use_per_cluster_sinkhorn", True)
# cluster_centroids and cluster_marg_p_pool must be precomputed once
# (e.g. from the real-feature memory bank) and passed in by the trainer.
```

Defaults match W-Flow behaviour exactly when `cluster_mode == "none"`.

## 6. Energy head wiring (TODO Gate 4)

1. Build `EnergyHead` from generator backbone features.
2. Add `L_OT + L_CD` to total loss; reuse `_compute_V_clustered`'s Γ.
3. Generator-initialised Langevin negatives via `langevin_sample`.
4. Inference: gradient descent on `V_ψ + data_fidelity` for inverse
   problems; Hessian of `V_ψ` for LID.

## 7. Hyper-parameter cheat-sheet

| Setting | uncond | cond | Notes |
|---|---|---|---|
| `cluster_mode` | hard | soft | cond already does per-class OT; soft sub-class clustering safer |
| `n_clusters` | 8–32 | 4–8 | clamp to `max(1, min(K, N))`; 8 used in 2D benchmark |
| `use_outer_gamma` | **true** | true | recommended — restores Thm 2 with no Prop 1 cost |
| `outer_gamma_eps` | 0.01 | 0.01 | sharper Sinkhorn → β closer to 1 at q=p |
| `outer_gamma_iter` | 200 | 200 | needed at small eps for convergence |
| `use_per_cluster_sinkhorn` | true | true | real speedup at N≥2048; collapses without outer Γ in unstable training |
| `mask_lambda` | n/a | 0.5–2.0 | only for `cluster_mode="soft"` |
| `cluster_centroids` | precomputed from real-pool k-means | same | sticky |
| `cluster_marg_p_pool` | precomputed from `assign(p_pool, centroids)` | same | sticky — required for outer Γ variance fix |
| `R_list` | as upstream | as upstream | unchanged |

## 8. Technical risks

- **Per-cluster hard mode can collapse during training** when many
  z-clusters become empty (gradient signal vanishes). Mitigation: enable
  `use_outer_gamma=True` so off-cluster mass keeps flowing, or use soft
  mode. Documented in proposal §6.5.
- **k-means convergence on early-training features** — sticky centroids
  from a fixed reference pool sidestep this entirely.
- **Empty clusters under `"hard"`** — `_per_cluster_bary` masks them
  out (T stays = z for empty z-cluster; empty support cluster gets a
  dummy weight then is discarded).
- **`torch.compile` and dynamic mask shapes** — set `DRIFT_COMPILE=0`
  if recompilation churn shows up in profiles (default for the 2D
  experiments via `_common.py`).
- **`batch_sinkhorn=True`** (W-Flow's optimised multi-R path) is
  bypassed when `cluster_mode != "none"`. Cluster path uses sequential
  per-R Sinkhorn. Re-fold into the batched path if profile demands it.
- **Outer Γ adds one K×K Sinkhorn per drift call**. Negligible for K ≤ 32
  (≪ per-cluster cost). May dominate for very large K.
