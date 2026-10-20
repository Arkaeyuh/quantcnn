#!/usr/bin/env python3
"""Appendix tables for the pretrained ResNet-18 backbone (CIFAR-10):

* training budget: probe accuracy at 30 epochs (``pretrained_1k_probe.csv``,
  ``pretrained_probe.csv``/``night_seeds_probe.csv``, ``pretrained_10k_probe.csv``) and at a
  matched number of optimizer updates (``pretrained_budget_probe.csv``), seeds 0-2;
* lockstep: three heads trained from copied weights on identical batches
  (``lockstep_probe.csv``), paired differences over seeds 0-4 x lr {3e-4, 1e-3}.

From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_pretrained_controls.py
"""

from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

HEADS = [("linear+resnet18_pt", "linear"), ("quantum+resnet18_pt", "VQC, ring (default)"),
         ("quantum+noent+resnet18_pt", "VQC, no CNOT")]
# size -> (30-epoch probe files, matched-budget epochs)
SIZES = {1000: (["pretrained_1k_probe"], 120), 4000: (["pretrained_probe"], 30), 10000: (["pretrained_10k_probe"], 12)}
BATCH = 32


def _probes(files: list[str], n: int) -> dict:
    out = defaultdict(dict)
    for name in files:
        for r in csv.DictReader(Path(f"results/{name}.csv").open()):
            if int(r["train_subset_size"]) == n and int(r["seed"]) < 3:
                out[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _budget_table() -> str:
    lines = []
    for n, (files, matched_ep) in SIZES.items():
        steps = math.ceil(n / BATCH)
        settings = [(30, _probes(files, n))]
        if n != 4000:
            settings.append((matched_ep, _probes(["pretrained_budget_probe"], n)))
        for epochs, p in settings:
            lin = p["linear+resnet18_pt"]
            cells = [ms(lin.values())] + [paired(p[k], lin) for k, _ in HEADS[1:]]
            tag = "" if epochs == 30 else r" (matched)"
            lines.append(f"{n:,} & {epochs}{tag} & {steps * epochs:,} & " + " & ".join(cells) + r" \\")
    return r"""\begin{table}[h]
\centering
\caption{Training budget on the pretrained ResNet-18 (CIFAR-10; seeds 0--2; probe accuracy, \%).
The protocol trains 30 epochs at every size, so smaller training sets get fewer optimizer
updates. Matched rows train 120 epochs at 1,000 images and 12 at 10,000, the update count of
4,000 images $\times$ 30 epochs; head learning rates are those selected at 30 epochs for each
size. Differences are paired over seeds, against the linear head.}
\label{tab:ptbudget}
\small
\begin{tabular}{lccccc}
\toprule
Images & Epochs & Updates & linear & default $-$ linear & no CNOT $-$ linear \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _lockstep_table() -> str:
    p = defaultdict(dict)
    for r in csv.DictReader(Path("results/lockstep_probe.csv").open()):
        p[r["source_head"]][(float(r["source_lr"]), int(r["seed"]))] = float(r["probe_test_acc"])
    q, z, c = p["quantum+resnet18_pt"], p["quantum+frozen+init0+resnet18_pt"], p["cosprod+resnet18_pt"]

    def row(label, a, b, la, lb):
        d = [a[(la, s)] - b[(lb, s)] for s in range(5)]
        half = 2.776 * st.stdev(d) / math.sqrt(len(d))  # t_{0.975, 4}: five seeds, paired
        m = st.mean(d)
        sign = "" if abs(m) < 0.05 else ("+" if m > 0 else "-")
        return (f"{label} & ${sign}{abs(m):.1f} \\pm {st.stdev(d):.1f}$ & [{m - half:+.1f}, {m + half:+.1f}] & "
                f"{sum(x < 0 for x in d)}/5 \\\\")

    lines = []
    for lr, tag in [(3e-4, r"$3\times10^{-4}$"), (1e-3, r"$10^{-3}$")]:
        lines.append(row(f"trained $-$ products, both at {tag}", q, c, lr, lr))
        lines.append(row(f"frozen $-$ products, both at {tag}", z, c, lr, lr))
    lines.append(r"\midrule")
    lines.append(row(r"trained at $3\times10^{-4}$ $-$ products at $10^{-3}$ (main-grid rates)", q, c, 3e-4, 1e-3))
    means = "; ".join(f"{tag}: " + " / ".join(f"{st.mean(v[(lr, s)] for s in range(5)):.1f}" for v in (q, z, c))
                      for lr, tag in [(3e-4, r"$3\times10^{-4}$"), (1e-3, r"$10^{-3}$")])
    return r"""\begin{table}[h]
\centering
\caption{Lockstep control on the pretrained ResNet-18 (CIFAR-10, 4,000 images). The trained
default circuit, the same circuit with its rotations frozen at exactly zero (forward map and
input gradients equal to the products head's), and the classical head on the default
circuit's zero-rotation products are trained in one process from copied initial weights on
identical batches; paired probe differences over seeds 0--4 at each head lr, with a 95\% paired
$t$ interval. The last row pairs the heads at the rates validation selected for them in the
main grid. Mean probe accuracy (trained / frozen / products), """ + means + r""".}
\label{tab:lockstep}
\small
\begin{tabular}{lccc}
\toprule
Paired difference & mean $\pm$ SD & 95\% interval & negative \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    write_table("tableA_pretrained_controls", _budget_table() + "\n" + _lockstep_table())


if __name__ == "__main__":
    main()
