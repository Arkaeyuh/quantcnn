#!/usr/bin/env python3
"""Appendix table: out-of-sample test of product order on a second family of CNOT layouts
(``results/family_probe.csv``; CIFAR-10, 4,000 images, small CNN). Predictions come from a
linear fit of probe accuracy on order over the seven ring-family VQC circuits (seeds 0-2),
fixed before the runs (``findings.md``). Prints the pre-registered statistics: the Spearman
correlation between order and mean probe accuracy over the new layouts, and how many land
within 4 points of their prediction. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_family.py
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, write_table  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402

# key -> (label, gates description)
FAMILY = {
    "star": ("star", r"$\mathrm{CNOT}(0,q)$, every layer"),
    "chain": ("chain", r"$\mathrm{CNOT}(q,q{+}1)$, $q<7$, every layer"),
    "star_in_last": ("star-in, last", r"$\mathrm{CNOT}(q,0)$, last layer"),
    "ladder2_last": ("ladder-2, last", r"$\mathrm{CNOT}(q,q{+}2)$, last layer"),
    "brick": ("brickwork", "alternating pairs, every layer"),
    "chain_last": ("chain, last", "open chain, last layer"),
}
FIT = (76.69, -3.22)  # probe = a + b * order over the ring family, recorded before the runs
MARGIN = 4.0


def _order(key: str) -> float:
    last = key.endswith("_last")
    return st.mean(len(s) for s in cnot_ring_supports(8, 4, last_only=last, pattern=key[:-5] if last else key))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def main():
    probe = defaultdict(dict)
    for r in csv.DictReader(Path("results/family_probe.csv").open()):
        probe[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    lines, orders, means, hits = [], [], [], 0
    for key, (label, gates) in FAMILY.items():
        vals = probe[f"quantum+{key}"]
        o, pred = _order(key), FIT[0] + FIT[1] * _order(key)
        m = st.mean(vals.values())
        hits += abs(m - pred) <= MARGIN
        orders.append(o)
        means.append(m)
        lines.append(f"{label} & {gates} & {o:g} & {pred:.1f} & {ms(vals.values())} & {m - pred:+.1f} & {len(vals)} \\\\")
    rho = _spearman(orders, means)
    print(f"Spearman(order, probe) over {len(orders)} layouts: {rho:.3f}; within {MARGIN:g} points: {hits}/{len(orders)}")
    body = r"""\begin{table}[h]
\centering
\caption{Out-of-sample test of product order on a second family of CNOT layouts (CIFAR-10,
4,000 images, small CNN, protocol of \cref{sec:setup}). The layouts were chosen by their
zero-rotation product order alone, and their probe accuracy was predicted, before training,
from a linear fit over the seven ring-family VQC circuits (probe $=76.7-3.22\times$ order).
Spearman correlation between order and mean probe accuracy over the six layouts: """ + f"{rho:.2f}" + r"""; """ + \
        f"{hits} of {len(orders)}" + r""" within 4 points of the prediction.}
\label{tab:family}
\small
\begin{tabular}{llccccc}
\toprule
Layout & CNOTs & Order & Predicted & Probe & Obs.\ $-$ pred. & $n$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_family", body)


if __name__ == "__main__":
    main()
