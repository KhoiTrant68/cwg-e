# CWG-E — Research Proposal (skeleton)

> Skeleton imported from project summary. Fill in marked TODOs before
> using as a submission-ready proposal.

## 1. One-sentence pitch

Replace W-Flow's global mini-batch Sinkhorn coupling with a hierarchical
cluster-wise Sinkhorn velocity field, and bolt on an explicit scalar
potential trained from the same per-cluster coupling — keeping one-step
generation and unlocking inverse problems, diversity control, and LID
estimation that W-Flow / Drifting cannot do.

## 2. Background & motivation

- WGF / JKO scheme.
- Sinkhorn divergence and its gradient flow.
- COT-FM: cluster-wise OT for flow matching — semantic clustering fixes
  mode-mixing in mini-batch OT.
- Energy Matching: a single scalar potential unifies FM far-field and
  EBM near-field via Boltzmann well.
- W-Flow: one-step pushforward of the Sinkhorn-divergence WGF — SOTA
  but uses a single global mini-batch coupling.

TODO: expand each bullet with citations and clarify the gap.

## 3. Contributions

1. **Hierarchical cluster-wise Sinkhorn velocity field** (§4.1) —
   generalises W-Flow's `diag_mask` to a block-mask on the cost matrix;
   single batched Sinkhorn call (no Python loop over clusters).
2. **Energy head trained from the same per-cluster coupling** (§4.2) —
   `L_OT` (far) + `L_CD` (near, generator-initialised Langevin negatives).
3. **Unified JKO/WGF lens** (§4.3) — W-Flow, COT-FM, Energy Matching,
   Drifting are ablations of one functional
   `transport_cost + potential + entropy`.

## 4. Method

### 4.1 Cluster-wise velocity field

Cluster `p` and `q` jointly in feature space (MAE / DINO). Within each
cluster solve a Sinkhorn problem; a coarse centroid-to-centroid plan Γ
routes mass across clusters (Γ = I gives pure within-cluster transport).

```
Ṽ(x_i) = T̃_{q,p}(x_i) − T̃_{q,q}(x_i)
T̃_{q,p}(x_i) = Σ_l Γ_{k(i),l} · T_{Q_{k(i)},P_l}(x_i) / Σ_l Γ_{k(i),l}
```

Plug into the one-step loss

```
L_gen = Σ_i || x_i − sg( x_i + η Ṽ(x_i) ) ||²
```

Implemented in `drift_loss_ot.py::_compute_V_clustered` via a block-mask
on the cost matrix (hard mode zeros off-block entries of Π before the
barycentric projection to remove the ~1% dual-potential leak).

### 4.2 Energy head

`V_ψ` shares the generator backbone, trained with `L_OT + L_CD` reusing
the same Γ. Negatives are generator-initialised short-chain Langevin —
much cheaper than EM's long chains. See `models/energy_head.py` (stubs).

TODO: write down the loss equations (Eq. 5–7 of the proposal).

### 4.3 Unified JKO/WGF lens

| Prior method      | Instance in CWG-E                |
|-------------------|----------------------------------|
| Drifting          | MMD-flow heuristic (Γ=I, no energy) |
| W-Flow            | Sinkhorn-divergence flow (K=1)   |
| COT-FM            | Block-diagonal coupling (multi-step) |
| Energy Matching   | Pure potential + entropy term    |

## 5. Theory (placeholders)

- **Thm 1 (Consistency under cluster separation).** TODO.
- **Thm 2 (No spurious equilibria).** TODO.
- **Prop 1 (Variance / cost reduction).** TODO.

## 6. Experiments

| Gate | Setup                              | Pass criterion                        |
|------|------------------------------------|---------------------------------------|
| 0    | PoC 2D drift                       | hard sharp + minority retention + cheap |
| 1    | Unit tests on drift_loss_ot        | off-block=0; targeting ↓ ~50%         |
| 2    | 2D one-step generator              | W₂² and minority recall, cluster ≥ none on ≥2/3 toys |
| 3    | CIFAR-10 1-NFE                     | cluster ≤ none FID; OT/step ≥1.5× faster |
| 4    | Energy head on; inverse + LID      | LID ≥ Energy Matching; inverse fewer steps |
| 5    | ImageNet-256 1-NFE + velocity-CFG  | FID ≤ 1.29 with secondary wins        |

## 7. Risks

- **Must beat 1.29 FID with secondary wins**, otherwise incremental.
- **Compute** — gate at 2D → CIFAR before ImageNet-256.
- **Conditional mode** — W-Flow already does per-class OT; the
  cluster-wise win is strongest at unconditional + sub-class clustering.

## 8. Positioning vs prior work

TODO: spell out reviewer-defence per paper.

## 9. References

See top-level `README.md`.
