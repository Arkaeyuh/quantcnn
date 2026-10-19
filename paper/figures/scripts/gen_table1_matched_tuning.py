#!/usr/bin/env python3
"""Table 1: test accuracy of linear / MLP-8 / VQC heads under per-head validation lr
selection, with paired VQC differences, and VQC - MLP-8 under the shared lr 1e-3.
From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_table1_matched_tuning.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

from scripts.summarize_v2 import protocol_data  # noqa: E402

ROOT = Path("results/protocol_v2")
HEADS = ("linear", "mlp8", "quantum")


def main():
    rows = []
    for name, grid in (("MNIST", "mnist_search"), ("CIFAR-10", "cifar_val")):
        sel, shared = protocol_data(ROOT / grid)
        sizes = sorted({n for n, _ in sel})
        for i, n in enumerate(sizes):
            best = max(HEADS, key=lambda h: st.mean(sel[(n, h)].values()))
            cells = [ms(sel[(n, h)].values(), bold=h == best) for h in HEADS]
            q = sel[(n, "quantum")]
            first = f"\\multirow{{{len(sizes)}}}{{*}}{{{name}}}" if i == 0 else ""
            rows.append(f"{first} & {n} & " + " & ".join(cells) + f" & {paired(q, sel[(n, 'linear')])}"
                        f" & {paired(q, sel[(n, 'mlp8')])} & {paired(shared[(n, 'quantum')], shared[(n, 'mlp8')])} \\\\")
        rows.append("\\midrule")
    rows.pop()
    body = r"""\begin{table*}[t]
\centering
\caption{Test accuracy (\%) of three heads on the same CNN, each with its head learning rate
selected by mean validation accuracy over the tested grid (backbone lr is coupled at $0.5\times$
the head lr); mean $\pm$ SD over 3 seeds; bold: highest mean in the row. MLP-8: one hidden
layer of width 8; VQC: the default 8-qubit, 4-layer ring circuit. Differences are in
percentage points, paired over seeds (same training subset and split). The last column
repeats VQC $-$ MLP-8 with both heads at one shared head lr ($10^{-3}$). The linear head's
selected lr lies at the top of its grid in five rows, so its lead may be understated.}
\label{tab:matched}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{3}{c}{Selected lr} & \multicolumn{2}{c}{VQC $-$ (selected lr)} & VQC $-$ MLP-8 \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}
Data & $N_\mathrm{train}$ & Linear & MLP-8 & VQC & Linear & MLP-8 & (lr $10^{-3}$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table1_matched_tuning", body)


if __name__ == "__main__":
    main()
