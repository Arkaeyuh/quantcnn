#!/usr/bin/env python3
"""
Aggregate ``metrics.csv`` files under a sweep directory and plot best validation accuracy.

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


def best_val_acc(csv_path: Path) -> float | None:
    best = None
    try:
        with csv_path.open(encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            for row in r:
                val = float(row["val_acc"])
                if best is None or val > best:
                    best = val
        return best
    except FileNotFoundError:
        return None


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


def collect(log_dir: Path) -> dict[tuple, list[float]]:
    """Key: (subset_size, series_label) -> val accs across seeds."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for p in sorted(log_dir.rglob("metrics.csv")):
        meta_parent = p.parent
        mv = meta_parent / "meta.json"
        if not mv.exists():
            continue
        ba = best_val_acc(p)
        if ba is None:
            continue
        cfg = json.loads(mv.read_text(encoding="utf-8"))
        subset = cfg.get("train_subset_size")
        if subset in (None, "full"):
            subset_key = subset
        else:
            subset_key = int(subset)

        hd = cfg["head"]
        ru = cfg.get("reupload_mode", "none")
        series = hd if hd != "quantum" else f"quantum::{ru}"

        groups[(subset_key, series)].append(ba)

    return dict(groups)


def plot_curves(series_map: dict[tuple, list[float]], out_png: Path, title: str | None = None) -> None:
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
            stds.append(statistics.pstdev(lst) if len(lst) > 1 else 0.0)
        if not xs:
            continue
        color = palette(i % 10)
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4, label=series, color=color)

    plt.xlabel("Stratified training subset size")
    plt.ylabel("Best validation accuracy (mean ± std over seeds)")
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
    data = collect(args.log_dir)
    if not data:
        raise SystemExit(f"No metrics found under {args.log_dir}")
    dataset = detect_dataset(args.log_dir)
    title = args.title or f"Hybrid quantum–classical {dataset.upper()} sweeps"
    plot_curves(data, out, title=title)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
