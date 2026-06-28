"""One-shot runner: PoC + Gate 2 + theorem/proposition experiments + bench.

Usage (Kaggle):
    !python experiments/run_all.py
    # ~10-25 minutes on a Kaggle T4 with default settings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


SCRIPTS = [
    ("poc_cwg_e.py",         []),
    ("gate2_train_2d.py",    ["--steps", "3000"]),
    ("thm_no_spurious.py",   []),
    ("thm_consistency.py",   []),
    ("prop_variance.py",     []),
    ("bench_cost.py",        []),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[],
                        help="script filenames to skip")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these script filenames")
    args = parser.parse_args()

    chosen = [(s, a) for s, a in SCRIPTS
              if s not in args.skip and (args.only is None or s in args.only)]

    t_total = time.time()
    for script, extra in chosen:
        path = HERE / script
        print("\n" + "=" * 64)
        print(f"[run_all] {script}  {' '.join(extra)}")
        print("=" * 64)
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(path), *extra])
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[run_all] {script} -> {status}  ({dt:.1f}s)")
        if rc != 0:
            sys.exit(rc)

    print(f"\n[run_all] all done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
