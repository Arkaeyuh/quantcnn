#!/usr/bin/env python3
"""Fig 1a (inset): the supports S_k of each readout <Z_k> at zero variational rotations,
for four CNOT structures, where <Z_k> = prod_{j in S_k} cos(pi tanh z_j). Computed by
``cnot_ring_supports``. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_fig1a_supports.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import CIRCUIT, plt, save_fig  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402

N = 8
# label -> CNOT structure (n_var_layers, last_only, pattern), or None for no CNOTs
PANELS = {
    "no CNOT / pairs": None,
    "pairs, last": (4, True, "pairs"),
    "ring (default)": (4, False, "ring"),
    "ring, last": (4, True, "ring"),
}


def main():
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 2, figsize=(2.5, 2.75))
    for ax, (label, structure) in zip(axes.flat, PANELS.items()):
        supports = [[k] for k in range(N)] if structure is None else cnot_ring_supports(N, *structure)
        m = np.zeros((N, N))
        for k, s in enumerate(supports):
            m[k, s] = 1
        ax.imshow(m, cmap=ListedColormap(["#F2F2F2", CIRCUIT]), vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(np.arange(-0.5, N), minor=True)
        ax.set_yticks(np.arange(-0.5, N), minor=True)
        ax.grid(which="minor", color="white", lw=0.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel(f"{label}, order {st.mean(len(s) for s in supports):g}", fontsize=7, labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("readout $k$", fontsize=7, labelpad=1)
    for ax in axes[0]:
        ax.annotate("input $j$", xy=(0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(wspace=0.12, hspace=0.32)
    save_fig(fig, "fig1a_supports")


if __name__ == "__main__":
    main()
