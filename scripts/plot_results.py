#!/usr/bin/env python3
"""
Aggregate ``metrics.csv`` files under a sweep directory and plot best per-run accuracy.

Each (subset size, head) group must be seeds of one configuration; a directory
that mixes learning rates or MLP widths (e.g. ``results/protocol_v2/*``) is
rejected; summarize those with ``scripts.summarize_v2``. The logged ``val_acc``
column is the test split for legacy runs (``val_size`` 0) and the held-out
validation split otherwise, and the axis label says which.

Usage::

    PYTHONPATH=. python -m scripts.plot_results --log_dir runs/logs/sweep_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def best_val_acc(csv_path: Path) -> tuple[float | None, int]:
    """Best logged ``val_acc`` and the number of epochs logged."""
    best = None
    n_rows = 0
    try:
        with csv_path.open(encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                n_rows += 1
                val = float(row["val_acc"])
                if best is None or val > best:
                    best = val
        return best, n_rows
    except FileNotFoundError:
        return None, 0


def detect_dataset(log_dir: Path) -> str:
    """Return the dataset name from the first meta.json found, or 'mnist' as fallback."""
    for mv in sorted(log_dir.rglob("meta.json")):
        try:
            cfg = json.loads(mv.read_text(encoding="utf-8"))
            if "dataset" in cfg:
                return cfg["dataset"]
        except Exception:
            continue
    return "mnist"


# Settings that may differ within one plotted group (one point = seeds of one configuration).
SEED_LEVEL = {"seed", "out_dir", "git_commit", "git_dirty"}


def collect(log_dir: Path) -> tuple[dict[tuple, list[float]], str]:
    """Key: (subset_size, series_label) -> best logged accuracies across seeds; plus the y-label."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    group_cfg: dict[tuple, dict] = {}
    splits = set()
    for p in sorted(log_dir.rglob("metrics.csv")):
        meta_parent = p.parent
        mv = meta_parent / "meta.json"
        if not mv.exists():
            continue
        ba, n_rows = best_val_acc(p)
        if ba is None:
            continue
        cfg = json.loads(mv.read_text(encoding="utf-8"))
        if n_rows < int(cfg["epochs"]):
            # An interrupted run is not a replicate of a completed one.
            print(f"skipping incomplete run {meta_parent.name}: {n_rows}/{cfg['epochs']} epochs")
            continue
        subset = cfg.get("train_subset_size")
        if subset in (None, "full"):
            subset_key = subset
        else:
            subset_key = int(subset)

        hd = cfg["head"]
        ru = cfg.get("reupload_mode", "none")
        series = hd if hd != "quantum" else f"quantum::{ru}"

        key = (subset_key, series)
        fixed = {k: v for k, v in cfg.items() if k not in SEED_LEVEL}
        if key in group_cfg and group_cfg[key] != fixed:
            diff = sorted(k for k in fixed.keys() | group_cfg[key].keys() if fixed.get(k) != group_cfg[key].get(k))
            raise SystemExit(f"{meta_parent.name}: runs in group {key} differ in {diff}; this plotter averages "
                             "seeds of one configuration only (use scripts.summarize_v2 for protocol_v2 grids)")
        group_cfg[key] = fixed
        splits.add("validation" if int(cfg.get("val_size") or 0) > 0 else "test-split")
        groups[key].append(ba)

    if len(splits) > 1:
        raise SystemExit("mixed legacy (test-split) and validation-protocol runs under one directory")
    label = f"Best {splits.pop() if splits else 'test-split'} accuracy (mean ± sample std over seeds)"
    return dict(groups), label


def plot_curves(series_map: dict[tuple, list[float]], out_png: Path, title: str | None = None,
                ylabel: str = "Best test-split accuracy (mean ± sample std over seeds)") -> None:
    subsets = sorted({k[0] for k in series_map.keys() if isinstance(k[0], int)})
    if not subsets:
        raise SystemExit("No integer subset sizes parsed; verify logs include meta.json & metrics.")

    series_names = sorted({k[1] for k in series_map})
    palette = plt.get_cmap("tab10")

    plt.figure(figsize=(9, 5))
    for i, series in enumerate(series_names):
        means = []
        stds = []
        xs = []
        for n in subsets:
            lst = series_map.get((n, series), [])
            if not lst:
                continue
            xs.append(n)
            means.append(statistics.mean(lst))
            stds.append(statistics.stdev(lst) if len(lst) > 1 else 0.0)  # sample SD, as in the paper
        if not xs:
            continue
        color = palette(i % 10)
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4, label=series, color=color)

    plt.xlabel("Stratified training subset size")
    plt.ylabel(ylabel)
    plt.title(title or "Hybrid quantum–classical sweeps")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=Path, required=True)
    ap.add_argument(
        "--out_png",
        type=Path,
        default=None,
        help="Output figure path. Default <log_dir>/accuracy_vs_subset.png",
    )
    ap.add_argument("--title", default=None, help="Plot title override.")
    args = ap.parse_args()
    out = args.out_png or (args.log_dir / "accuracy_vs_subset.png")
    data, ylabel = collect(args.log_dir)
    if not data:
        raise SystemExit(f"No metrics found under {args.log_dir}")
    dataset = detect_dataset(args.log_dir)
    title = args.title or f"Hybrid quantum–classical {dataset.upper()} sweeps"
    plot_curves(data, out, title=title, ylabel=ylabel)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
