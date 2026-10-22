#!/usr/bin/env python3
"""Appendix table: the gradient-floor test of the attenuation hypothesis (``results/floor_probe.csv``,
``results/floor_mlp_probe.csv``; coupled heads from ``results/{cosprod,seeds,mlp}_probe.csv``;
CIFAR-10, 4,000 images, small CNN, lr 1e-2 selected for every head). Each product head's backward
pass floors the magnitude of every path's other-factor product while keeping its sign; the
forward pass is unchanged. Predictions, the training guard and the refutation criterion were
committed before the runs (``predictions/prospective_predictions.md``). From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_floor.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

SUPPORTS = {"cosprod": "default (order 2.5)", "cosprod_last": "last ring (order 5.25)"}
FLOORS = {"": "0", "_fl03": "0.3", "_fl1": "1"}
GRIDS = ("cifar_cosprod", "cifar_seeds", "cifar_floor", "cifar_floor_seeds")


def _probes(paths: list[str]) -> dict:
    out = defaultdict(dict)
    for p in paths:
        for r in csv.DictReader(Path(p).open()):
            out[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _train(head: str) -> list[float]:
    return [json.loads(f.read_text())["train_clean_acc"] * 100 for g in GRIDS
            for f in Path("results/protocol_v2", g).glob(f"cifar10_subset4000_{head}_lr0.01_s*/final.json")]


def main():
    lin = _probes(["results/cosprod_probe.csv", "results/seeds_probe.csv", "results/floor_probe.csv"])
    mlp = _probes(["results/mlp_probe.csv", "results/floor_mlp_probe.csv"])
    lines = []
    for base, label in SUPPORTS.items():
        for suffix, floor in FLOORS.items():
            h = base + suffix
            tr = _train(h)
            assert len(tr) == len(lin[h]) == len(mlp[h]) == 5, h
            delta = "---" if not suffix else paired(lin[h], lin[base])
            lines.append(f"{label if not suffix else ''} & {floor} & {ms(lin[h].values())} & {delta} & "
                         f"{ms(mlp[h].values())} & {st.mean(tr):.1f} \\\\")
            print(h, f"probe {st.mean(lin[h].values()):.2f}", delta, f"train {st.mean(tr):.1f}")
        lines.append(r"\addlinespace" if base == "cosprod" else "")
    body = r"""\begin{table}[h]
\centering
\caption{Gradient-floor test of the attenuation hypothesis (CIFAR-10, 4,000 images, small CNN,
5 seeds, head learning rate $10^{-2}$ selected for every head). Classical product heads keep the
forward products; in the backward pass each path's derivative, the product $P$ of the other
factors in the readout's support, is replaced by $\mathrm{sign}(P)\max(|P|,\tau)$ (sign $+1$ at $P=0$): $\tau=0$ is the
true gradient, $\tau=1$ removes all magnitude attenuation but keeps each path's sign. Predictions,
a training guard (clean training accuracy within 10 points of $\tau=0$) and a refutation criterion
were committed before the runs. $\Delta$: paired probe difference to $\tau=0$. Train: clean
training accuracy (\%).}
\label{tab:floor}
\small
\begin{tabular}{lccccc}
\toprule
Supports & $\tau$ & Probe & $\Delta$ & MLP probe & Train \\
\midrule
""" + "\n".join(l for l in lines if l) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_floor", body)


if __name__ == "__main__":
    main()
