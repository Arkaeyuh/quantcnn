#!/usr/bin/env python3
"""Appendix table: trained circuits vs their zero-rotation product model, held-out
(``results/zero_angle_check.csv``; CIFAR-10, 4000 images, validation embeddings).
From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_zero_angle.py
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import write_table  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402

# head key -> (label, CNOT structure for the order, or None for no CNOTs)
HEADS = {
    "quantum+frozen": ("ring (default), frozen at init", (4, False, "ring")),
    "quantum": ("ring (default)", (4, False, "ring")),
    "quantum+entlast": ("ring, last", (4, True, "ring")),
    "quantum+pairs_last": ("pairs, last", (4, True, "pairs")),
    "quantum+pairs": ("pairs", (4, False, "pairs")),
    "quantum+noent": ("no CNOT", None),
}


def main():
    rows = defaultdict(list)
    for r in csv.DictReader(Path("results/zero_angle_check.csv").open()):
        rows[r["head"]].append(r)
    lines = []
    for key, (label, structure) in HEADS.items():
        rs = rows[key]
        m = lambda c: st.mean(float(r[c]) for r in rs)  # noqa: E731
        order = 1 if structure is None else st.mean(len(s) for s in cnot_ring_supports(8, *structure))
        lines.append(f"{label} & {order:g} & {len(rs)} & {m('r2_heldout'):.3f} & {m('acc_heldout'):.1f} & "
                     f"{m('acc_remap_heldout'):.1f} & {m('lossgrad_cos_zeroed'):.2f} & {m('lossgrad_cos_remap'):.2f} & "
                     f"{m('mean_abs_ry'):.2f} / {m('mean_abs_rz'):.2f} \\\\")
    body = r"""\begin{table}[h]
\centering
\caption{Trained circuits against their own zero-rotation product model (CIFAR-10, 4,000
training images; each run's validation embeddings, split by index parity; means over $n$
seeds). An affine map from the readouts of the same circuit with all rotations set to zero,
$q_0$, to the trained readouts $q$ is fit on one half and evaluated on the other. $R^2$:
held-out fraction of trained readout variance it recovers. Accuracy: held-out accuracy of the
trained head, and of its trained readout layer applied to the remapped $q_0$. Loss-gradient
cosine: median over held-out samples of the cosine between $\partial\mathcal{L}/\partial z$
(cross-entropy) of the trained head and of the zeroed head with the trained readout kept
(zeroed) or applied to the remapped $q_0$ (remapped). Last column: mean $|\theta|$ of the
$R_y$ / $R_z$ rotations (rad).}
\label{tab:zeroangle}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
& & & & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Loss-grad.\ cosine} & \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Circuit & Order & $n$ & $R^2$ & trained & remapped & zeroed & remapped & $|R_y|$ / $|R_z|$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_zero_angle", body)


if __name__ == "__main__":
    main()
