#!/usr/bin/env python3
"""Fig 1b: backbone linear-probe accuracy vs the head's zero-rotation product order.

CIFAR-10, 4000 training images. Each point is a head (circuit variant or classical
head trained end to end on the same zero-rotation products) at its validation-selected
lr; mean +- SD over seeds (5, or 3 where marked). Product order is mean |S_k| from
``cnot_ring_supports``. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_fig1b_probe_vs_order.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import CIRCUIT, CLASSICAL, REFERENCE, plt, save_fig  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402
from scripts.summarize_v2 import FIVE_SEED_PROBES, five_seed_data  # noqa: E402

# key -> (label, circuit?, CNOT structure (n_var_layers, last_only, pattern) or None for none,
#         multiplicative x jitter (log axis), label offset in points, label alignment)
HEADS = {
    "quantum+noent": ("no CNOT", True, None, 0.93, (-4, -2), "right"),
    "quantum+pairs": ("pairs", True, (4, False, "pairs"), 0.97, (-5, -4), "right"),
    "cos": ("cos", False, None, 1.03, (4, -3), "left"),
    "quantum+pairs_last": ("pairs, last", True, (4, True, "pairs"), 0.965, (-4, -9), "right"),
    "cosprod_pairs_last": ("cos-prod.", False, (4, True, "pairs"), 1.035, (4, -3), "left"),
    "quantum": ("ring (default)", True, (4, False, "ring"), 0.97, (-4, -8), "right"),
    "cosprod": ("cos-prod.", False, (4, False, "ring"), 1.03, (4, 0), "left"),
    "quantum+L2": ("ring, 2 layers$^{\\dagger}$", True, (2, False, "ring"), 0.91, (-4, 0), "right"),
    "quantum+entlast": ("ring, last", True, (4, True, "ring"), 0.955, (-4, -3), "right"),
    "quantum+L1": ("ring, 1 layer$^{\\dagger}$", True, (1, False, "ring"), 1.0, (-4, -7), "right"),
    "cosprod_last": ("cos-prod.", False, (4, True, "ring"), 1.05, (4, 0), "left"),
}


def order(structure) -> float:
    if structure is None:
        return 1.0
    layers, last_only, pattern = structure
    return st.mean(len(s) for s in cnot_ring_supports(8, layers, last_only, pattern))


def main():
    _, probes = five_seed_data(Path("results/protocol_v2"), FIVE_SEED_PROBES)
    fig, ax = plt.subplots(figsize=(4.0, 2.75))
    lin = list(probes["linear"].values())
    m, sd = st.mean(lin), st.stdev(lin)
    ax.axhspan(m - sd, m + sd, color=REFERENCE, alpha=0.18, lw=0)
    ax.axhline(m, color=REFERENCE, lw=0.8, ls="--")
    ax.text(6.9, m + 0.35, f"linear head ({m:.1f}, n = {len(lin)})", ha="right", va="bottom",
            fontsize=7, color="#505050")
    for key, (label, is_circuit, structure, dx, off, ha) in HEADS.items():
        v = list(probes[key].values())
        x, y = order(structure) * dx, st.mean(v)
        color = CIRCUIT if is_circuit else CLASSICAL
        ax.errorbar(x, y, yerr=st.stdev(v), fmt="o" if is_circuit else "s", ms=4.2, color=color,
                    mfc=color if is_circuit else "white", mew=1.0, elinewidth=0.8, capsize=1.8, zorder=3)
        assert ("dagger" in label) == (len(v) == 3), key  # dagger marks n = 3
        ax.annotate(label, (x, y), xytext=off,
                    textcoords="offset points", ha=ha, va="center", fontsize=6.5, color=color)
    ax.set_xlabel(r"product order at zero rotations, mean $|S_k|$")
    ax.set_ylabel("backbone linear-probe test acc. (%)")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2.5, 5.25], ["1", "1.5", "2.5", "5.25"])
    ax.minorticks_off()
    ax.set_xlim(0.5, 7.0)
    ax.plot([], [], "o", color=CIRCUIT, ms=4.2, label="VQC head")
    ax.plot([], [], "s", color=CLASSICAL, mfc="white", ms=4.2, label="classical head, same products")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    save_fig(fig, "fig1b_probe_vs_order")


if __name__ == "__main__":
    main()
