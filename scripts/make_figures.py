#!/usr/bin/env python3
"""Regenerate the paper's accuracy-vs-subset figures from the released raw logs.

Uses ``scripts.make_tables.load`` (which checks epoch completion, seeds and
settings) and plots the per-cell mean of best test-split accuracy as a line,
with each seed as a faint dot.

From repository root::

    PYTHONPATH=. python -m scripts.make_figures
"""

from __future__ import annotations

import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.make_tables import ORDER, load

PANELS = [
    ("results/raw/sweep_20260627_095118", "MNIST", "paper/figures/mnist_accuracy.png", True),
    ("results/raw/sweep_20260628_223253", "CIFAR-10", "paper/figures/cifar10_accuracy.png", False),
]
STYLE = {
    "linear": ("Linear", "#2878d6", "o"),
    "mlp": ("MLP", "#1aae7f", "s"),
    "quantum::none": ("Quantum (none)", "#eda100", "^"),
    "quantum::light": ("Quantum (light)", "#008300", "D"),
    "quantum::strong": ("Quantum (strong)", "#4a3aa7", "v"),
}


def main():
    for sweep, title, out, legend in PANELS:
        cells, _ = load(Path(sweep))
        sizes = sorted({k[0] for k in cells})
        fig, ax = plt.subplots(figsize=(4.5, 3.6), dpi=200)
        for s in ORDER:
            label, color, marker = STYLE[s]
            per_seed = [[cells[(n, s)][i][0] for i in range(3)] for n in sizes]
            for n, v in zip(sizes, per_seed):
                ax.scatter([n] * 3, v, s=10, color=color, alpha=0.4, linewidths=0)
            ax.plot(sizes, [st.mean(v) for v in per_seed], color=color, marker=marker, lw=2, ms=8, label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Training samples (stratified)")
        ax.set_ylabel("Test-split accuracy (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if legend:
            ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
