#!/usr/bin/env python3
"""Appendix tables: effective rank of the backbone's loss gradient by head
(``results/grad_rank.csv``), and the large-initialization control
(``results/init_probe.csv`` against the near-identity runs in ``backbone_probe.csv`` /
``circuit_probe.csv``). CIFAR-10, 4,000 training images. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_gradrank_init.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

RANK_HEADS = [("linear", "linear", "---"), ("quantum+noent", "VQC, no CNOT", "1"),
              ("quantum", "VQC, ring (default)", "2.5"), ("quantum+entlast", "VQC, ring, last", "5.25")]


def _rank_table() -> str:
    rank = defaultdict(dict)
    for r in csv.DictReader(Path("results/grad_rank.csv").open()):
        rank[(r["head"], r["stage"])][int(r["seed"])] = float(r["erank"])
    lines = []
    for key, label, order in RANK_HEADS:
        cells = [ms(rank[(key, st)].values()) for st in ("init", "final")]
        diff = [paired(rank[(key, st)], rank[("quantum", st)]) if key != "quantum" else "---" for st in ("init", "final")]
        lines.append(f"{label} & {order} & {cells[0]} & {diff[0]} & {cells[1]} & {diff[1]} \\\\")
    return r"""\begin{table}[h]
\centering
\caption{Effective rank (of 8) of the class signal the backbone receives: $\exp H(p)$, with $p$
the normalized eigenvalues of $\frac1N\sum_i g_ig_i^\top$, $g_i=\partial\mathcal{L}_i/\partial z$
(cross-entropy) on the 2,000 validation embeddings (CIFAR-10, 4,000 training images; 5 seeds).
Final: the trained run at its validation-selected lr. Init: a freshly initialized model of the
same kind built with the run's seed. $\Delta$: paired difference to the default circuit.}
\label{tab:gradrank}
\small
\begin{tabular}{lccccc}
\toprule
& & \multicolumn{2}{c}{Init} & \multicolumn{2}{c}{Final} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Head & Order & rank & $\Delta$ & rank & $\Delta$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _init_table() -> str:
    probe = defaultdict(dict)
    lrs = {}
    for name in ("backbone_probe", "circuit_probe", "init_probe"):
        for r in csv.DictReader(Path(f"results/{name}.csv").open()):
            if r["dataset"] == "cifar10" and r["train_subset_size"] == "4000" and int(r["seed"]) < 3:
                probe[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
                lrs[r["source_head"]] = r["source_lr"]
    rows = [("VQC, ring (default)", "quantum"), ("VQC, no CNOT", "quantum+noent")]
    lines = []
    for label, key in rows:
        big = key + "+init3.14"
        lines.append(f"{label} & {ms(probe[key].values())} & {ms(probe[big].values())} & "
                     f"{paired(probe[big], probe[key])} & {lrs[key]} / {lrs[big]} \\\\")
    lines.append(r"\midrule")
    lines.append(f"no CNOT $-$ default & {paired(probe['quantum+noent'], probe['quantum'])} & "
                 f"{paired(probe['quantum+noent+init3.14'], probe['quantum+init3.14'])} & & \\\\")
    return r"""\begin{table}[h]
\centering
\caption{Initialization control: probe accuracy (\%) with the variational rotations initialized
in $\mathcal{U}(-0.1,0.1)$ (near the identity, the protocol default) or $\mathcal{U}(-\pi,\pi)$
(CIFAR-10, 4,000 training images; seeds 0--2; head lr selected on validation from
$\{3\times10^{-3},10^{-2},3\times10^{-2}\}$ at the large initialization). The last row is the
paired difference between the two circuits.}
\label{tab:init}
\small
\begin{tabular}{lcccc}
\toprule
Circuit & $\mathcal{U}(-0.1,0.1)$ & $\mathcal{U}(-\pi,\pi)$ & large $-$ small & selected lr \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    write_table("tableA_gradrank_init", _rank_table() + "\n" + _init_table())


if __name__ == "__main__":
    main()
