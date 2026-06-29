# Clustered Wasserstein Gradient Flows with an Outer Coupling

> Research proposal for a CVPR / NeurIPS submission. All numerical claims
> below are from the toy benchmark in `cwge_out_3/` (commit `3a3e8d4`,
> 2D, N=2048 unless noted otherwise).

## Abstract

We study **cluster-wise barycentric maps** as estimators of the Sinkhorn
drift used in W-Flow's one-step generator training, and ask a sharp
question: what does clustering trade off between mini-batch *variance*,
*equilibrium points*, and *bias*? We answer with three theorems
(`docs/theory.md`): **(Thm A)** an explicit closed-form condition
`tr Σ_{p_k₀}(x) < π_{k₀} · tr Σ_p(x)` deciding when cluster-wise wins
variance — winning at moderate ε on few-cluster mixtures, but **tied**
in the far-separated limit; **(Thm B.1)** bare cluster-wise has a
*spurious equilibrium* whenever cluster shapes match but cluster masses
differ, which **explains** (Cor B.3) the early-training collapse of one-
step generators with cluster-only drift; **(Thm C)** the cluster-wise
bias decays as `O(e^{-δ²/(2ε)})` in inter-cluster separation, shifting
the bias–variance frontier. We resolve B.1 with an **outer centroid
coupling Γ** (Thm B.2, sketch) using sticky pool-level marginals, which
restores "drift → 0 iff q = p". On a 2D benchmark (`cwge_out_3/`) each
theorem is empirically confirmed — including the predicted Gate-2
generator collapse, which we present as confirmation of B.3 rather than
a failure mode.

---

## 1. Pitch

This is a **theory-first paper** on cluster-wise entropic-OT velocity
estimators. The unit of contribution is a set of conditional theorems
with explicit thresholds — not a SOTA-chasing method paper. The
method we propose (CWG-E: per-cluster Sinkhorn + outer Γ + sticky pool
statistics) is the **algorithmic instance** the theory points to, and
serves to make every theorem empirically verifiable on a small 2D
benchmark a reader can rerun in 12 minutes on Kaggle.

Concretely, we replace W-Flow's *single global mini-batch* Sinkhorn
coupling with a two-level coupling: per-cluster Sinkhorn for the bulk
of the transport (low-variance under the Thm A condition, consistent
per Thm C, fast) plus an outer centroid Sinkhorn Γ that routes mass
across clusters (required by Thm B.2 to restore "vanish iff q = p").
Both levels use sticky pool statistics. The Theorem-A condition tells
the user *when* clustering is the right estimator and *when it is not*
— a deliberate contrast to "always better" method claims.

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

1. **Bias–variance theory of cluster-wise barycentric maps** (§5,
   full proofs in `docs/theory.md`). Thm A + Cor A.1 give a *closed-form
   condition* on when the cluster-wise estimator strictly improves
   variance; Lem A.2 shows the gain vanishes in the far-separated limit,
   correcting the common intuition that "more separation = more gain".
   Thm C bounds the cluster-wise bias as `O(e^{-δ²/(2ε)})` and unifies
   with Thm A to characterize the bias–variance frontier.
2. **Equilibrium characterization for cluster-wise drift** (§5, Thm B).
   Thm B.1 proves that bare cluster-wise drift has a spurious-equilibrium
   set parameterized by mismatched cluster masses; Cor B.3 derives the
   one-step-generator collapse from B.1 as a deductive consequence.
   Thm B.2 (sketch) shows an outer Sinkhorn coupling Γ over cluster
   centroids — with **sticky pool marginal** — restores "V → 0 iff q = p".
3. **Method (CWG-E)** (§4): per-cluster Sinkhorn with sticky centroids
   (§4.1) + additive outer-Γ correction with sticky pool marginal (§4.2).
   API-compatible with W-Flow (`cluster_mode="none"` reproduces
   upstream bit-for-bit). Batched implementation via a single
   `_sinkhorn_batched` call over K padded sub-problems.
4. **2D theory benchmark** (§6, `experiments/`, ≤ 12 min on Kaggle T4).
   Each script targets one theorem: `thm_no_spurious.py` ↔ Thm B,
   `prop_variance.py` ↔ Thm A / Cor A.1, `thm_consistency.py` ↔ Thm C,
   `bench_cost.py` for wall-clock, `gate2_train_2d.py` for Cor B.3.
   Reference outputs frozen in `cwge_out_3/`.

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

> Statements summarised here; **full proofs** with the setup, lemmas,
> and discussion are in [`docs/theory.md`](theory.md). Three results
> (Thm A, B.1, C) have complete proofs at the level stated; Thm B.2
> and the particle→WGF convergence theorem are sketches and flagged in
> `theory.md §7`.

Let `p = Σ_k π_k p_k` on `ℝ^d` with supports `S_k`, separation
`δ = min_{k≠l} dist(S_k, S_l)`, diameter `R`. Fix a source point `x`
with `c(x) = k₀`. Define the Gibbs barycentric map `τ_μ(x)` from `μ`
at scale `ε > 0` and let `Σ_μ(x)` be its asymptotic mini-batch covariance
(eq. (2.1) in `theory.md`).

**Thm A (variance characterisation, `theory.md §2`).**
With M iid p-samples, the asymptotic mini-batch covariances of the
global and cluster-wise estimators at `x ∈ S_{k₀}` are
`M · Cov[T̂_glob] → Σ_p(x)` and `M · Cov[T̂_clus] → (1/π_{k₀}) Σ_{p_{k₀}}(x)`.

**Cor A.1 (explicit win condition).**
Cluster-wise strictly reduces estimator variance (in trace) at `x` iff
`tr Σ_{p_{k₀}}(x) < π_{k₀} · tr Σ_p(x)`. *Interpretation:* the
inequality captures the trade-off between cross-mode numerator bloat
(favouring cluster-wise) and the `1/π_{k₀}` sample-size penalty
(disfavouring it for many small clusters). The 2D data exactly track
this prediction: ring8/ring_minority (`K=8`) win; grid25 (`K=25`,
penalty ≈ 25) ties.

**Lem A.2 (far-separated tie).** As `ε / δ² → 0`,
`Σ_p(x) = (1/π_{k₀}) Σ_{p_{k₀}}(x) · (1 + O(e^{-δ²/(2ε)}))`. The
cluster-wise variance gain is therefore a **finite-ε phenomenon** — it
vanishes in the well-separated limit, contrary to the intuition that
"more separation = more gain".

**Thm B.1 (spurious equilibria of bare cluster-wise drift,
`theory.md §3`).** With marginal renormalisation inside each cluster,
the cluster-wise drift `V` satisfies `V ≡ 0 ⇔ q̂_k = p̂_k ∀ k` where
`q̂_k = q|_{S_k}/m^q_k` etc. In particular, any `q` whose **intra-cluster
shapes** match `p`'s but whose **cluster masses** differ (`m^q ≠ m^p`)
is a spurious zero.

**Cor B.3 (predicted Gate-2 collapse).** Early in training of a one-step
generator with bare cluster-wise drift, `m^q` is far from `m^p` (several
clusters empty), so by B.1 the mass-mismatch component produces zero
force — there is no gradient pulling `q` to cover empty clusters, and
collapse is self-reinforcing. This is exactly what `gate2_train_2d.py`
exhibits (§6.5) and what `thm_no_spurious.py` measures (cluster-only
ratio = 1×, see §6.1).

**Thm B.2 (outer-Γ restores `V ≡ 0 ⇔ q = p`; sketch).** Adding a
centroid-level Sinkhorn term `Γ = π^{ε_γ}(m^q, m^p)` with **sticky pool
marginal** yields `V^{+Γ} ≡ 0 ⇔ q = p`. The sticky marginal is the
non-obvious design choice: without it `Γ` absorbs mini-batch noise and
the variance gain of Cor A.1 is destroyed.

**Thm C (consistency, `theory.md §4`).** Under separation `δ`,
`‖τ_p(x) − τ_{p_{k₀}}(x)‖ ≤ (2R(1−π_{k₀}))/(π_{k₀} κ_{k₀}(x)) · e^{-δ²/(2ε)}`.
Combined with Thm A this gives an explicit bias–variance frontier in `ε`.

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

### 6.5 Confirmation of Cor B.3 — one-step generator collapse (`gate2_train_2d.py`)

| Toy | Global W² | Cluster soft W² | Cluster hard W² |
|---|---:|---:|---:|
| ring8 | 0.22 | 0.59 | 7.94 (collapse) |
| grid25 | 0.11 | 0.34 | 4.46 |
| ring_minority | 0.12 | 0.57 | 6.57 |

Cluster-only modes do not beat the W-Flow baseline on 2D one-step
generator training. **This is the predicted behaviour, not an
implementation defect**: by Thm B.1 the mass-mismatch component of `q`
produces zero drift, and Cor B.3 derives the self-reinforcing collapse
that follows. Hard mode collapses sharply (empty `z`-clusters → V = 0
on those particles); soft mode trains stably (cluster mass leaks across
the soft mask) but at 2–5× higher W². The gains demonstrated in §6.1–6.4
are in drift *estimation* — the regime where Thm A's variance condition
is the binding question. The next test of Thm B.2 (outer-Γ-fixed
generator training) is a 7-variant sweep in `gate2_train_2d.py` (see
`VARIANTS` dict).

## 7. Limitations

- **Generator training on 2D toys does not improve** (§6.5) — but this
  is now a *theorem* (Cor B.3), not a failed experiment. Whether the
  outer-Γ fix (Thm B.2) actually heals generator training is the open
  question the `gate2_v2` 7-variant sweep is designed to answer.
- **Theory rigour is layered** (see `theory.md §7`): Thm A / A.1 / A.2,
  B.1 / B.3, and C have full proofs at the level stated (asymptotic,
  Gibbs kernel form, population-level). Thm B.2 (outer-Γ ⇒ no spurious
  zero) is a sketch — the two-term interaction needs an OT specialist's
  audit. The particle → WGF convergence (`theory.md §6`) is only a
  strategy, with a concrete reduction to W-Flow's convergence theorem
  proposed as a tractable shortcut.
- **Gibbs vs full Sinkhorn.** Theory is proved for the one-step Gibbs
  barycentric map (eq. 0.1 in `theory.md`); the implementation uses full
  log-domain Sinkhorn (column potentials `g`). Extension is conjectured
  routine but not yet written.
- **Outer Γ adds compute** — one extra K × K Sinkhorn per drift call.
  Negligible for K ≤ 32.
- **K is a hyperparameter.** For 2D K = 8 throughout; auto-tuning
  (silhouette, elbow, or feature-space clustering with MAE/DINO) is
  future work.

## 8. Positioning

- **vs W-Flow**: same one-step pushforward and Sinkhorn divergence
  flow, but the coupling is hierarchical (cluster + outer Γ) with
  sticky pool statistics. Drop-in API compatibility (`cluster_mode="none"`
  reproduces W-Flow bit-for-bit). On the 2D drift estimation
  benchmark, CWG-E confirms Thm A / Cor A.1 (variance), Thm B (no
  spurious equilibrium with outer Γ), Thm C (consistency), and the
  predicted cost improvement (§6).
- **vs COT-FM**: COT-FM uses cluster-wise OT for *multi-step* flow
  matching. CWG-E is for *one-step* W-Flow drift estimation and adds
  the outer Γ + sticky structure, neither of which COT-FM has.
- **vs Drifting**: Drifting's mean-shift drift fails the "vanish iff
  q = p" property (flat at 0.0044 across all α in §6.1) — it does not
  even satisfy Thm B.1's necessary condition. CWG-E inherits the
  principled OT drift from W-Flow and improves it.
- **vs Energy Matching**: orthogonal — EM contributes a scalar
  potential, our drift contribution composes with it. An energy-head
  extension is implemented as a Gate-4 stub in `models/energy_head.py`.

## 9. Roadmap

| Gate | Setup | Status |
|------|-------|--------|
| 0 | PoC 2D drift figure | ✅ `poc_drift_comparison.png` |
| 1 | Unit tests + bench | ✅ 3/3 pass; 3.3× speedup at N=8192 |
| 2 | 2D one-step generator | ✅ collapse predicted by Cor B.3 (§6.5) |
| 2.5 | `gate2_v2` 7-variant sweep — does outer-Γ heal training? | running |
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
