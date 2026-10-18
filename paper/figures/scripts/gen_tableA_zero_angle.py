#!/usr/bin/env python3
"""Appendix table: trained circuits vs their zero-rotation product model
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
        lines.append(f"{label} & {order:g} & {len(rs)} & {m('out_r2'):.3f} & {m('jac_cos_q'):.3f} & "
                     f"{m('jac_cos_logits'):.3f} & {m('acc'):.1f} & {m('acc_zeroed'):.1f} & "
                     f"{m('mean_abs_ry'):.2f} / {m('mean_abs_rz'):.2f} \\\\")
    body = r"""\begin{table}[h]
\centering
\caption{Trained circuits against their own zero-rotation product model (CIFAR-10, 4,000
training images; each run's validation embeddings; means over $n$ seeds). $R^2$: fraction of
the trained readout variance a linear map recovers from the readouts of the same circuit with
all rotations set to zero. Jacobian cosine: median over samples of the cosine between the
input Jacobians of the trained and zeroed circuits, for the readouts and for the logits
under the trained readout layer. Accuracy: validation accuracy of the head, and of the same
head with rotations zeroed but the readout layer kept (not refit). Last column: mean
$|\theta|$ of the $R_y$ / $R_z$ rotations (rad).}
\label{tab:zeroangle}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
& & & & \multicolumn{2}{c}{Jacobian cosine} & \multicolumn{2}{c}{Accuracy} & \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Circuit & Order & $n$ & $R^2$ & readouts & logits & trained & zeroed & $|R_y|$ / $|R_z|$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_zero_angle", body)


if __name__ == "__main__":
    main()
