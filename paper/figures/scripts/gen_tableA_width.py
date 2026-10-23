#!/usr/bin/env python3
"""Appendix table: width replication, 8 vs 12 qubits (12-dimensional bottleneck:
``results/q12_probe.csv``, ``results/q12_mlp_probe.csv``; 8 qubits: the five-seed headline probes;
CIFAR-10, 4,000 images, small CNN). Predictions and criteria for the 12-qubit runs were committed
before the runs (``predictions/prospective_predictions.md``). From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_width.py
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

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402
from scripts.summarize_v2 import FIVE_SEED_PROBES, five_seed_data  # noqa: E402

# head key -> (label, last_only or None for no CNOT, 12-qubit run-name stem, selected lr at 12)
HEADS = {"linear": ("linear", "skip", "linear_b12", "0.03"),
         "quantum+noent": ("VQC, no CNOT", None, "quantum_noent_b12", "0.01"),
         "quantum": ("VQC, ring (default)", False, "quantum_b12", "0.01"),
         "quantum+entlast": ("VQC, ring, last", True, "quantum_entlast_b12", "0.01")}
SLOPE = 3.22  # ring-family slope at 8 qubits, used for the committed order-only extrapolation


def _order(n: int, last) -> str:
    if last == "skip":
        return "---"
    if last is None:
        return "1"
    return f"{st.mean(len(s) for s in cnot_ring_supports(n, 4, last_only=last)):.2f}"


def main():
    _, p8 = five_seed_data(Path("results/protocol_v2"), FIVE_SEED_PROBES)
    p12, m12 = defaultdict(dict), defaultdict(dict)
    for path, store in (("results/q12_probe.csv", p12), ("results/q12_mlp_probe.csv", m12)):
        for r in csv.DictReader(Path(path).open()):
            store[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    lines = []
    for h, (label, last, stem, lr) in HEADS.items():
        train = [json.loads(f.read_text())["train_clean_acc"] * 100 for g in ("cifar_q12", "cifar_q12_ext", "cifar_q12_seeds")
                 for f in Path("results/protocol_v2", g).glob(f"cifar10_subset4000_{stem}_lr{lr}_s*/final.json")]
        assert len(train) == len(p12[h]) == len(m12[h]) == len(p8[h]) == 5, h
        d8 = "---" if h in ("linear", "quantum+noent") else paired(p8[h], p8["quantum+noent"])
        d12 = "---" if h in ("linear", "quantum+noent") else paired(p12[h], p12["quantum+noent"])
        lines.append(f"{label} & {_order(8, last)} & {ms(p8[h].values())} & {d8} & {_order(12, last)} & "
                     f"{ms(p12[h].values())} & {d12} & {ms(m12[h].values())} & {st.mean(train):.1f} \\\\")
    o12 = {h: st.mean(len(s) for s in cnot_ring_supports(12, 4, last_only=HEADS[h][1])) for h in ("quantum", "quantum+entlast")}
    ext = {h: -SLOPE * (o - 1) for h, o in o12.items()}
    dlin = paired(p12["quantum"], p12["linear"])
    print("order-only extrapolation vs no CNOT:", {h: round(v, 1) for h, v in ext.items()}, "; default - linear", dlin)
    body = r"""\begin{table*}[t]
\centering
\caption{Width replication (CIFAR-10, 4,000 images, small CNN, protocol of \cref{sec:setup}, 5 seeds): the
headline heads with an 8-dimensional bottleneck and 8 qubits, and with a 12-dimensional bottleneck and
12 qubits. Product orders follow \cref{eq:identity} with $q$ the number of qubits. Predictions for the 12-qubit runs were
committed before training: the ordering no CNOT $>$ default $>$ last ring (met), and a paired mean
default $-$ no CNOT $\le-2$ and default $-$ linear $<0$, each negative in at least 4 of 5 seeds (met:
$-4.0\pm1.0$ and """ + dlin + r""", every seed negative).
Extrapolating the 8-qubit ring-family slope ($3.22$ points per unit of order) predicted gaps of
""" + f"${ext['quantum']:.1f}$ and ${ext['quantum+entlast']:.1f}$" + r""" to the no-CNOT head; the observed gaps are much smaller and close to
those at 8 qubits. $\Delta$: paired probe difference to the no-CNOT head. Selected head learning rates at
12 qubits: $3\times10^{-2}$ (linear, after one edge step) and $10^{-2}$ (VQC). Train: clean training accuracy at 12 qubits (\%).}
\label{tab:width}
\small
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{3}{c}{8 qubits} & \multicolumn{5}{c}{12 qubits} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-9}
Head & Order & Probe & $\Delta$ & Order & Probe & $\Delta$ & MLP probe & Train \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("tableA_width", body)


if __name__ == "__main__":
    main()
