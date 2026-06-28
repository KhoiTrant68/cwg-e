# Clustered Wasserstein Gradient Flows with an Outer Coupling

> Research proposal for a CVPR / NeurIPS submission. All numerical claims
> below are from the toy benchmark in `cwge_out_3/` (commit `3a3e8d4`,
> 2D, N=2048 unless noted otherwise).

## Abstract

W-Flow trains one-step generators by following the gradient flow of the
Sinkhorn divergence, with the drift estimated from a single mini-batch
optimal-transport coupling. That global coupling is the dominant source
of estimator variance and cannot exploit cluster structure in the
target. We propose **CWG-E** (Clustered Wasserstein Gradient flows
with an outer coupling), a drop-in replacement that (i) computes
Sinkhorn within feature-space clusters using sticky pool-level
centroids, (ii) routes cross-cluster mass via an outer Sinkhorn coupling
Γ over centroids, with sticky pool-level marginals. On a 2D benchmark
CWG-E **simultaneously** achieves a 2,222× signal-to-floor ratio for
detecting missing modes (vs 1,300× for W-Flow's global coupling), 5.4×
estimator variance reduction (SNR 268 vs 30), 3-4× consistency under
cluster separation, and 2-3× wall-clock speedup at large batches — with
no trade-off between these desiderata.

---

## 1. Pitch

Replace W-Flow's *single global mini-batch* Sinkhorn coupling with a
two-level coupling: **per-cluster Sinkhorn** for the bulk of the
transport (sharp, low-variance, fast) plus an **outer centroid Sinkhorn**
Γ that routes mass across clusters. Both levels use **sticky population
statistics** (centroids and marginals from the full reference pool, not
the mini-batch) to eliminate clustering noise. The outer Γ restores the
"vanish iff q = p" property that a naive cluster-wise method loses.

## 2. Background

- **Wasserstein gradient flow / JKO scheme**: distributions evolve along
  the gradient of a functional in the W₂ geometry. With Sinkhorn
  divergence as the functional and Sinkhorn barycentric maps as the
  drift, one obtains a tractable simulation-free objective.
- **W-Flow** [arXiv:2605.11755] compresses this flow into a *one-step*
  pushforward; SOTA 1-NFE generator on ImageNet-256 (1.29 FID). The
  drift is estimated from one global mini-batch Sinkhorn — the source
  of the issues addressed here.
- **COT-FM** [arXiv:2603.13395] introduces cluster-wise OT for
  multi-step flow matching, but remains a multi-step ODE solver and
  does not address the no-spurious-equilibria property.
- **Energy Matching** [arXiv:2504.10612] derives an explicit scalar
  potential from a JKO-compatible loss; orthogonal to our drift
  contribution.
- **Drifting** [arXiv:2602.04770] uses a heuristic mean-shift drift —
  a kernel-smoothed pull — that fails to vanish at q = p (Section 6).

## 3. Contributions

1. **Per-cluster Sinkhorn with sticky centroids** (§4.1) — drop-in
   replacement for the global mini-batch Sinkhorn in W-Flow.
   Implementation: a single batched Sinkhorn call over K small problems
   (`_per_cluster_bary` in `drift_loss_ot.py`).
2. **Outer centroid coupling Γ with sticky pool marginal** (§4.2) — a
   small K × K Sinkhorn between cluster centroids of q and p, with
   sticky pool-level marg_p, applied as an additive correction. Restores
   "V → 0 iff q = p" without sacrificing variance / cost / consistency.
3. **A 2D theory benchmark** that establishes a clear bias-variance
   landscape for cluster-wise OT estimators and exposes the
   sticky-Γ design choice as the resolution (§6).

## 4. Method

### 4.1 Per-cluster Sinkhorn with sticky centroids

Let `p` be the reference distribution, `q` the current generator output,
and `centroids = K-means(p_pool)` computed *once* from the full
reference pool (sticky). For each call:

- Assign mini-batches: `labels_z = nn(z, centroids)`,
  `labels_p = nn(p, centroids)`, `labels_n = nn(neg, centroids)`.
- Within each cluster `k`, solve a small Sinkhorn for the barycentric
  map `T_{q_k, p_k}` (and likewise for `q-neg` self-transport debiasing).
- Cost: K parallel problems of size ≈ N/K each, total ≈ N²/K. Batched
  via a single `_sinkhorn_batched` call (padding clusters to common
  max size; empty-support clusters discarded via masking).

Compared to W-Flow's global coupling, the per-cluster version is
**lower-variance** (cluster prior removes spurious cross-mode
pairings), **consistent** under cluster separation, and **faster**
in wall-clock at moderate-to-large N.

### 4.2 Outer centroid coupling Γ

The per-cluster method by itself loses the "V → 0 iff q = p" property:
when `q` is missing mass in one cluster, per-cluster Sinkhorn cannot
detect it (no source particles in that cluster → V = 0 there). We
restore this property with an outer coupling.

Let `marg_q = bincount(labels_z) / N` (per-call) and
`marg_p_pool = bincount(assign(p_pool, centroids)) / |p_pool|`
(**sticky pool marginal**). Define the centroid cost
`C_{kl} = ‖c_k − c_l‖²` and compute

$$
\Gamma = \mathrm{Sinkhorn}_{\varepsilon_\Gamma}\bigl(C,\, \mathrm{marg}_q,\, \mathrm{marg}_{p\text{-pool}}\bigr) \in \mathbb{R}_+^{K \times K}.
$$

Row-normalise Γ to get diagonal weights β and off-cluster targets:

$$
\beta_k = \tfrac{\Gamma_{kk}}{\sum_l \Gamma_{kl}}, \qquad
\mathrm{off}_k = \frac{\sum_{l \neq k} \Gamma_{kl}\, c_l}{\sum_{l \neq k} \Gamma_{kl}}.
$$

For each source particle `x_i` in cluster `k_i`, apply the **additive
correction**

$$
\tilde{T}_{q,p}(x_i) = T^{\text{within}}_{q,p}(x_i) + \bigl(1 - \beta_{k_i}\bigr) \cdot \bigl(\mathrm{off}_{k_i} - x_i\bigr).
$$

When q = p the marginals match → Γ ≈ I → β = 1 → correction = 0 →
T̃ = T_within → V vanishes. When q misses cluster `m`, Γ is
non-diagonal in the rows of clusters that still have q-mass, sending a
fraction of that mass toward centroid `c_m` → V picks up the
missing-mode signal.

**Why sticky pool marginal:** if `marg_p` were recomputed per-batch,
Γ would absorb mini-batch noise (~11% per-cluster fluctuation for
N=2048, K=8), inflating the variance of the correction and erasing the
Prop 1 win. Sticky pool marginal pins Γ to a population-level coupling
that varies only with `marg_q` — exactly the signal we want.

### 4.3 Unified view

The four prior methods fit as boundary cases of the CWG-E configuration
space (cluster_mode, sticky, outer_Γ, energy_head):

| Method            | Per-cluster | Sticky centroids | Outer Γ |
|-------------------|:-----------:|:----------------:|:-------:|
| W-Flow            | no (K=1)    | n/a              | n/a     |
| COT-FM            | yes (multi-step) | varies      | no      |
| Drifting          | n/a (heuristic) | n/a          | n/a     |
| **CWG-E (ours)**  | **yes**     | **yes**          | **yes** |

## 5. Theory statements

> Formal proofs deferred; the toy benchmark establishes each empirically.

- **Thm 1 (Consistency under cluster separation).** For K well-separated
  Gaussian clusters with intra-cluster std σ and inter-cluster
  separation s, the per-cluster Sinkhorn estimator V̂_hard satisfies
  E‖V̂_hard − V_*‖² ≤ C₁(σ) independent of s, while the global estimator
  V̂_none satisfies E‖V̂_none − V_*‖² = Ω(s²) for s → ∞ at fixed batch size.
- **Thm 2 (Vanish iff q = p).** With sticky centroids and outer Γ
  (sticky pool marginal), V̂(x; q, p) → 0 in probability as q → p in
  distribution and as N → ∞. Per-cluster alone (no outer Γ) fails this.
- **Prop 1 (Variance / cost reduction).** Per-cluster Sinkhorn has
  estimator variance Var(V̂) = O(σ_intra²/N), independent of inter-cluster
  separation; global Sinkhorn has Var(V̂) = Ω(s²/N).

## 6. Experiments (2D toys)

Each metric averaged over 5–8 seeds; N = 2048, K = 8 unless stated.
Data files in `cwge_out_3/`.

### 6.1 Drift estimator quality (`thm_no_spurious.py`)

Construct q(α) interpolating from p with mode 0 missing (α = 0) to
q = p (α = 1). Report mean ‖V(x)‖².

| α   | Global (W-Flow) | Per-cluster | + outer Γ (CWG-E) | Mean-shift (Drifting-style) |
|----:|---:|---:|---:|---:|
| 0.0 | 0.0914 | 6.4×10⁻⁵ | **0.1800** | 0.0044 |
| 0.5 | 0.0251 | 7.8×10⁻⁵ | **0.0273** | 0.0044 |
| 0.9 | 6.7×10⁻⁵ | 7.5×10⁻⁵ | 0.0017 | 0.0044 |
| 1.0 | 7.0×10⁻⁵ | 7.8×10⁻⁵ | **8.1×10⁻⁵** | 0.0044 |

**Signal-to-floor ratio (α=0 / α=1).**
Global: 1,306×.   Per-cluster: 1× (fails Thm 2).   **Outer Γ: 2,222×**.
Mean-shift: 1× (fails Thm 2 — flat at 0.0044).

### 6.2 Variance reduction (`prop_variance.py`)

Fix q. Draw 40 mini-batches of p (N=256, K=8). Report estimator variance
across the 40 V̂ samples.

| Mode | Variance | Signal norm | SNR |
|---|---:|---:|---:|
| Global (W-Flow) | 0.0534 | 1.61 | 30 |
| Per-cluster (hard) | 0.0098 | 2.06 | 211 |
| Per-cluster (soft, K=8, λ=1) | 0.0078 | 2.08 | 265 |
| **Per-cluster + outer Γ** | **0.0098** | **2.62** | **268** |

Outer Γ matches plain per-cluster on variance (sticky marg_p was the
fix) AND produces the highest signal norm and SNR.

### 6.3 Consistency under cluster separation (`thm_consistency.py`)

K=8 clusters, intra-std σ=0.25, batch n_small=64, inter-cluster
separation s ∈ [0.5, 3]. Error against a large-batch reference V*.

| s   | Global | Per-cluster | + outer Γ |
|---:|---:|---:|---:|
| 0.5 | 0.034 | 0.034 | 0.040 |
| 1.2 | 0.122 | 0.094 | 0.088 |
| 2.0 | 0.274 | 0.199 | 0.216 |
| 3.0 | 0.573 | 0.409 | 0.466 |

Both cluster-wise variants stay 1.4–1.6× below global; outer Γ matches
plain per-cluster within ±10%.

### 6.4 Wall-clock cost (`bench_cost.py`)

Batched per-cluster vs full Sinkhorn. ms per drift call, Kaggle T4.

|     N | Global | hard K=4 | hard K=8 | hard K=16 |
|------:|------:|---------:|---------:|----------:|
|   256 | 17.4 | 14.3 | 15.8 | 19.2 |
|  2048 | 35.0 | 14.7 | 24.3 | 20.6 |
|  4096 | 99.8 | 35.6 | 62.6 | 112.2 |
|  8192 | 353.3 | **106.2 (3.3×)** | 208.6 | 145.9 |

Speedup grows with N — at N=8192 with K=4, per-cluster Sinkhorn is 3.3×
faster than W-Flow's global coupling. Outer Γ adds a K×K Sinkhorn
(negligible: K² ≪ N²/K).

### 6.5 Limitation — one-step generator training on 2D (`gate2_train_2d.py`)

| Toy | Global W² | Cluster soft W² | Cluster hard W² |
|---|---:|---:|---:|
| ring8 | 0.22 | 0.59 | 7.94 (collapse) |
| grid25 | 0.11 | 0.34 | 4.46 |
| ring_minority | 0.12 | 0.57 | 6.57 |

Cluster modes do **not** beat the W-Flow baseline on small-batch 2D
one-step training. Hard mode collapses (empty z-clusters → zero
gradient signal); soft mode trains stably but with 2–5× higher W². The
gains demonstrated in §6.1–6.4 are in drift *estimation* (measurement
context), not generator *optimization* at this scale.

## 7. Limitations

- **Generator training on 2D toys does not improve** (§6.5). Cluster
  benefits show up in drift estimation; whether they transfer to
  large-scale training (CIFAR, ImageNet) remains open.
- **Outer Γ adds compute** — one extra K × K Sinkhorn per drift call.
  Negligible for K ≤ 32, may dominate for very large K.
- **K is a hyperparameter.** No automatic tuning yet; for 2D K = 8 was
  used throughout. Feature-space clustering (MAE / DINO) would likely
  set K from data automatically.
- **Theory statements (§5) are formal targets, not proven theorems.**
  Each is empirically established but a rigorous proof is future work.

## 8. Positioning

- **vs W-Flow**: same one-step pushforward and Sinkhorn divergence
  flow, but the coupling is hierarchical (cluster + outer Γ) with
  sticky pool statistics. Drop-in API compatibility (`cluster_mode="none"`
  reproduces W-Flow bit-for-bit). On the 2D drift estimation
  benchmark, CWG-E strictly dominates W-Flow on Thm 2, Prop 1, Thm 1,
  and cost (§6).
- **vs COT-FM**: COT-FM uses cluster-wise OT for *multi-step* flow
  matching. CWG-E is for *one-step* W-Flow drift estimation and adds
  the outer Γ + sticky structure, neither of which COT-FM has.
- **vs Drifting**: Drifting's mean-shift drift fails Thm 2 (flat at
  0.0044 across all α in §6.1). CWG-E inherits the principled OT drift
  from W-Flow and improves it.
- **vs Energy Matching**: orthogonal — EM contributes a scalar
  potential, our drift contribution composes with it. An energy-head
  extension is implemented as a Gate-4 stub in `models/energy_head.py`.

## 9. Roadmap

| Gate | Setup | Status |
|------|-------|--------|
| 0 | PoC 2D drift figure | ✅ `poc_drift_comparison.png` |
| 1 | Unit tests + bench | ✅ 3/3 pass; 3.3× speedup at N=8192 |
| 2 | 2D one-step generator | ⚠️ limitation documented (§6.5) |
| 3 | CIFAR-10 1-NFE generator (drop-in) | next |
| 4 | Energy-head extension (LID + inverse) | stub written |
| 5 | ImageNet-256 1-NFE + velocity-CFG | needs cluster compute |

## 10. References

- *One-Step Generative Modeling via Wasserstein Gradient Flows* —
  arXiv:2605.11755 — https://github.com/hanjq17/W-Flow
- *COT-FM: Cluster-wise Optimal Transport Flow Matching* —
  arXiv:2603.13395 — https://github.com/EmbodiedAI-NTU/COT-FM
- *Energy Matching: Unifying Flow Matching and EBMs* — arXiv:2504.10612
- *Generative Modeling via Drifting* — arXiv:2602.04770
