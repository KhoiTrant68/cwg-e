# 2D Experiments — theory validation + Gate 2

All scripts here are 2D, self-contained, and runnable on a single Kaggle
T4 / P100 in ~12 minutes total. They write CSVs and PNGs to:

- `/kaggle/working/cwge_out/` when run on Kaggle, or
- `<repo>/out/cwge_out/` otherwise.

Reference outputs from a validated run are in
[`cwge_out_3/`](../cwge_out_3/) (committed; matches the numbers in
[`docs/proposal.md`](../docs/proposal.md) §6).

## Files

| Script | What it shows | Maps to |
|---|---|---|
| `_common.py` | Shared helpers: 2D samplers, W², MMD, mode coverage | — |
| `poc_cwg_e.py` | Cluster-wise vs global Sinkhorn drift on a 2D mixture | Gate 0 |
| `gate2_train_2d.py` | One-step generator on 3 toys × {none, hard, soft} | Gate 2 |
| `thm_no_spurious.py` | `‖V‖ → 0  iff  q = p` along a 1-param family (4 methods compared) | Thm 2 |
| `thm_consistency.py` | Estimator error vs cluster separation | Thm 1 |
| `prop_variance.py` | V variance across mini-batch draws (4 modes, incl. outer Γ) | Prop 1 |
| `bench_cost.py` | Wall-clock per drift call vs N ∈ {256..8192} and K ∈ {4, 8, 16} | Cost |
| `run_all.py` | Runs all of the above sequentially | — |

## Kaggle quick-start

In a Kaggle notebook (GPU enabled):

```python
!git clone https://github.com/KhoiTrant68/cwg-e.git /kaggle/working/cwg-e
%cd /kaggle/working/cwg-e
!python experiments/run_all.py
```

The first run installs POT + scikit-learn automatically (≤ 30s).
Outputs go to `/kaggle/working/cwge_out/`.

### Single-experiment runs

```python
!python experiments/gate2_train_2d.py --steps 6000 --batch 256
!python experiments/thm_no_spurious.py --n 2048 --repeats 8
!python experiments/bench_cost.py --Ns 256 512 1024 2048 4096 8192
!python experiments/prop_variance.py --n-draws 40
!python experiments/thm_consistency.py --K 8 --seps 0.5 0.8 1.2 1.6 2.0 3.0
```

## Approx. wall-times (Kaggle T4)

| Script | Time |
|---|---|
| `poc_cwg_e.py` | ~ 10s |
| `gate2_train_2d.py` (3000 steps × 3 toys × 3 modes) | ~ 6 min |
| `thm_no_spurious.py` (N=2048) | ~ 5 min |
| `thm_consistency.py` | ~ 90s |
| `prop_variance.py` | ~ 90s |
| `bench_cost.py` (N up to 8192) | ~ 90s |

CPU also works (no CUDA fall-back trap); `gate2_train_2d.py` is ~5×
slower on CPU.

## Validated pass criteria

| Experiment | Pass = | Measured (cwge_out_3, N=2048) |
|---|---|---|
| `thm_no_spurious.py` | outer-Γ vanishes at α=1, signal at α=0 ≥ global | ✅ ratio 2,222× (vs 1,306× global) |
| `prop_variance.py` | hard+outer-Γ variance ≤ hard alone; SNR > none | ✅ var 0.0098 (= hard); SNR 268 (vs 30 none) |
| `thm_consistency.py` | cluster error < global error at all separations | ✅ 1.4–1.6× lower at every s |
| `bench_cost.py` | hard cheaper than none at large N | ✅ 3.3× speedup at N=8192, K=4 |
| `gate2_train_2d.py` | cluster modes ≥ none on W² | ⚠️ documented limitation (proposal §6.5) |

## Wiring back into the paper

Each CSV is one row per setting; the tables in
[`docs/proposal.md`](../docs/proposal.md) §6 are populated directly from
the `cwge_out_3/*.csv` files. Each PNG is a candidate figure for the
paper.
