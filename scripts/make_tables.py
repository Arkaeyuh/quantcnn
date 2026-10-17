#!/usr/bin/env python3
"""Reproduce the paper's summary statistics from raw sweep logs.

Checks every cell has seeds {0, 1, 2}, all runs completed their configured
epochs, and all runs in a sweep share the same settings apart from
head/re-upload/subset/seed. Prints mean +- sample SD of best test-split accuracy,
final-epoch accuracy, and the best-vs-final difference per cell.

From repository root::

    PYTHONPATH=. python -m scripts.make_tables runs/logs/sweep_20260627_095118 runs/logs/sweep_20260628_223253
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ["linear", "mlp", "quantum::none", "quantum::light", "quantum::strong"]
VARYING = {"head", "reupload_mode", "train_subset_size", "seed", "mlp_hidden"}


def load(sweep: Path):
    cells = defaultdict(dict)
    shared = None
    for run in sorted(p for p in sweep.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((run / "meta.json").read_text())
        rows = list(csv.DictReader((run / "metrics.csv").open()))
        if len(rows) != int(meta["epochs"]):
            raise SystemExit(f"{run.name}: {len(rows)}/{meta['epochs']} epochs logged")
        fixed = {k: v for k, v in meta.items() if k not in VARYING}
        if shared is None:
            shared = fixed
        elif fixed != shared:
            raise SystemExit(f"{run.name}: settings differ from the rest of the sweep: {fixed} vs {shared}")
        series = meta["head"] if meta["head"] != "quantum" else f"quantum::{meta['reupload_mode']}"
        best = max(float(r["val_acc"]) for r in rows) * 100
        final = float(rows[-1]["val_acc"]) * 100
        cells[(int(meta["train_subset_size"]), series)][int(meta["seed"])] = (best, final)
    for key, by_seed in cells.items():
        if sorted(by_seed) != [0, 1, 2]:
            raise SystemExit(f"{sweep.name} {key}: seeds {sorted(by_seed)}")
    return cells, shared


def main():
    for sweep in map(Path, sys.argv[1:]):
        cells, shared = load(sweep)
        print(f"\n== {sweep.name}  ({len(cells)} cells x 3 seeds)  {shared}")
        print(f"{'n':>6} " + " ".join(f"{s:>16}" for s in ORDER))
        diffs = []
        for n in sorted({k[0] for k in cells}):
            line = []
            for s in ORDER:
                v = cells[(n, s)]
                b = [v[i][0] for i in range(3)]
                diffs.append(st.mean(v[i][0] - v[i][1] for i in range(3)))
                line.append(f"{st.mean(b):6.1f} +- {st.stdev(b):4.1f}  ")
            print(f"{n:>6} " + " ".join(line))
        print(f"best-minus-final (mean over seeds): max {max(diffs):.2f}; "
              f"<=0.5 in {sum(d <= 0.5 for d in diffs)}/{len(diffs)} cells")


if __name__ == "__main__":
    main()
