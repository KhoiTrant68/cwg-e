# 2D Experiments — theory validation + Gate 2

All scripts here are 2D, self-contained, and runnable on a single Kaggle
T4 / P100 in ≤ 30 minutes total. They write CSVs and PNGs to:

- `/kaggle/working/cwge_out/` when run on Kaggle, or
- `<repo>/out/cwge_out/` otherwise.

## Files

| Script                  | What it shows                                          | Maps to     |
|-------------------------|--------------------------------------------------------|-------------|
| `poc_cwg_e.py`          | Cluster-wise vs global drift on a 2D mixture (figure)  | Gate 0      |
| `gate2_train_2d.py`     | One-step generator on 3 toys × {none, hard, soft}      | Gate 2      |
| `thm_no_spurious.py`    | `‖V‖ → 0  iff  q = p` along a 1-param family            | Thm 2       |
| `thm_consistency.py`    | Estimator error vs cluster separation                  | Thm 1       |
| `prop_variance.py`      | V variance across mini-batch draws                     | Prop 1      |
| `bench_cost.py`         | Wall-clock per drift call vs N and K                   | Cost claim  |
| `run_all.py`            | Runs the above sequentially                            | —           |

## Kaggle quick-start

In a Kaggle notebook (GPU enabled):

```python
!git clone <your repo URL> /kaggle/working/cwg-e
%cd /kaggle/working/cwg-e
!python experiments/run_all.py
```

Or if you uploaded the repo as a dataset:

```python
import shutil, os
shutil.copytree("/kaggle/input/<dataset-name>/cwg-e", "/kaggle/working/cwg-e")
%cd /kaggle/working/cwg-e
!python experiments/run_all.py
```

The first run installs POT + scikit-learn automatically (≤30s).

### Single-experiment runs

```python
!python experiments/gate2_train_2d.py --steps 6000 --batch 256
!python experiments/thm_no_spurious.py --repeats 8
!python experiments/bench_cost.py --Ns 256 512 1024 2048 4096
```

## Approx. wall-times (Kaggle T4)

| Script               | Time   |
|----------------------|--------|
| `poc_cwg_e.py`       |  ~ 20s |
| `gate2_train_2d.py`  |  ~6 min (3 toys × 3 modes × 3k steps) |
| `thm_no_spurious.py` |  ~ 90s |
| `thm_consistency.py` |  ~ 90s |
| `prop_variance.py`   |  ~ 90s |
| `bench_cost.py`      |  ~ 60s |

CPU works too (no CUDA fall-back trap) but `gate2_train_2d.py` is ~5× slower.

## Pass criteria

| Experiment              | What "pass" looks like                                      |
|-------------------------|-------------------------------------------------------------|
| `gate2_train_2d.py`     | On ≥ 2 / 3 toys: hard (or soft) ≤ none W²; minority recall ↑ |
| `thm_no_spurious.py`    | Global + cluster curves monotone → 0 at α=1; heuristic plateaus |
| `thm_consistency.py`    | err(none) grows with separation; err(hard) flat-ish         |
| `prop_variance.py`      | var(hard) < var(none); var(soft) in between                  |
| `bench_cost.py`         | hard cheaper than none for moderate / large N               |

## Wiring back into the paper

Each CSV is one row per setting; copy-paste into the proposal's
`docs/proposal.md` §6 / §5 tables. Each PNG is a candidate figure.
