#!/usr/bin/env python3
"""Appendix table: the common linear probe vs a nonlinear (MLP) probe on the same frozen
bottlenecks, and gradient-decoupled product heads (CIFAR-10, 4000 images). From repository
root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_mechanism.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

from scripts.summarize_v2 import FIVE_SEED_PROBES, BASE_PROBES, circuit_data, select_lr  # noqa: E402

ROOT = Path("results/protocol_v2")
HEADS = {
    "linear": "linear", "mlp8": "MLP-8", "quantum+noent": "VQC, no CNOT", "quantum+pairs": "VQC, pairs",
    "quantum+pairs_last": "VQC, pairs, last", "quantum": "VQC, ring (default)", "quantum+entlast": "VQC, ring, last",
    "cos": "cos", "cosprod": "cos-prod.\\ (default supports)", "cosprod_last": "cos-prod.\\ (last-ring supports)",
}
DECOUPLED = {"cosprod_sg": ("cosprod", "default supports"), "cosprod_last_sg": ("cosprod_last", "last-ring supports")}


def _probes(paths: list[Path]) -> dict:
    out = {}
    for path in paths:
        for r in csv.DictReader(path.open()):
            if r["dataset"] == "cifar10" and r["train_subset_size"] == "4000":
                out.setdefault(r["source_head"], {})[int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _clean_train(grid: str, key: str, lr: float) -> list[float]:
    vals = []
    for fin in (ROOT / grid).glob(f"cifar10_subset4000_{key}_lr{lr:g}_s*/final.json"):
        if int(json.loads((fin.parent / "meta.json").read_text())["seed"]) < 3:
            vals.append(json.loads(fin.read_text())["train_clean_acc"] * 100)
    return vals


def main():
    lin = _probes(FIVE_SEED_PROBES)
    mlp = _probes([Path("results/mlp_probe.csv")])
    rows = []
    for key, name in HEADS.items():
        seeds = sorted(set(lin[key]) & set(mlp[key]))
        a, b = {s: mlp[key][s] for s in seeds}, {s: lin[key][s] for s in seeds}
        rows.append(f"{name} & {len(seeds)} & {ms(b.values())} & {ms(a.values())} & {paired(a, b)} \\\\")
    gap_lin = paired({s: lin["quantum"][s] for s in range(5)}, {s: lin["linear"][s] for s in range(5)})
    gap_mlp = paired({s: mlp["quantum"][s] for s in range(5)}, {s: mlp["linear"][s] for s in range(5)})

    cells, probes = circuit_data(ROOT / "cifar_sg", BASE_PROBES + [Path("results/sg_probe.csv")])
    _, ref_probes = circuit_data(ROOT / "cifar_cosprod", BASE_PROBES + [Path("results/cosprod_probe.csv")])
    dec = []
    for key, (ref, label) in DECOUPLED.items():
        by_lr = cells[("cifar10", 4000, key)]
        lr = select_lr(by_lr)
        p = {s: v for s, v in probes[("cifar10", 4000, key)].items() if s < 3}
        pr = {s: v for s, v in ref_probes[("cifar10", 4000, ref)].items() if s < 3}
        ref_lr = select_lr(circuit_data(ROOT / "cifar_cosprod", [])[0][("cifar10", 4000, ref)])
        dec.append(f"{label} & {lr:g} & {ms(_clean_train('cifar_cosprod', ref, ref_lr))} & "
                   f"{ms(_clean_train('cifar_sg', key, lr))} & {ms(pr.values())} & {ms(p.values())} \\\\")
    body = r"""\begin{table}[h]
\centering
\caption{Linear vs nonlinear probe on the same frozen bottlenecks (CIFAR-10, 4,000 training
images; MLP probe: standardization, then $8\to256\to256\to10$ with ReLU, probe learning rate and
epoch selected on validation; mean $\pm$ SD over $n$ seeds; last column paired). The MLP probe
recovers at most a fraction of a point more than the linear probe for any head, and VQC $-$
linear is """ + gap_mlp + r""" under the MLP probe vs """ + gap_lin + r""" under the linear
probe.}
\label{tab:mlpprobe}
\small
\begin{tabular}{lcccc}
\toprule
Head the backbone was trained with & $n$ & Linear probe & MLP probe & MLP $-$ linear \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Gradient-decoupled classical product heads: same forward products, but each
input's gradient passes only through its own cosine factor (straight-through on the other
factors). CIFAR-10, 4,000 images, 3 seeds; decoupled heads at their validation-selected
learning rate (grid $10^{-3}$--$3\times10^{-2}$), coupled heads as in \cref{tab:support}.
Decoupling fails to fit the training set, so it does not isolate the effect of coupling.}
\label{tab:decoupled}
\small
\begin{tabular}{lccccc}
\toprule
& & \multicolumn{2}{c}{Clean-train acc.} & \multicolumn{2}{c}{Probe} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Products & lr & coupled & decoupled & coupled & decoupled \\
\midrule
""" + "\n".join(dec) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_mechanism", body)


if __name__ == "__main__":
    main()
