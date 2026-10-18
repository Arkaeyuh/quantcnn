#!/usr/bin/env python3
"""Appendix table: zero-rotation readout supports S_k for each CNOT structure
(``cnot_ring_supports``). From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_supports.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import write_table  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402

N = 8
CIRCUITS = {
    "no CNOT": None,
    "pairs (every layer)": (4, False, "pairs"),
    "pairs, last": (4, True, "pairs"),
    "ring (default)": (4, False, "ring"),
    "ring, last": (4, True, "ring"),
    "ring, $L=1$": (1, False, "ring"),
    "ring, $L=2$": (2, False, "ring"),
}


def main():
    lines = []
    for label, structure in CIRCUITS.items():
        sup = [[k] for k in range(N)] if structure is None else cnot_ring_supports(N, *structure)
        cells = " & ".join("".join(map(str, s)) for s in sup)
        lines.append(f"{label} & {st.mean(len(s) for s in sup):g} & {cells} \\\\")
    body = r"""\begin{table}[h]
\centering
\caption{Zero-rotation supports $S_k$ of each readout $\langle Z_k\rangle$, written as digit strings (e.g.\ 015 is $\{0,1,5\}$), and
the mean product order $\bar{s}$, computed by propagating $Z_k$ backwards through each
circuit's CNOTs.}
\label{tab:supports}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lc""" + "l" * N + r"""}
\toprule
Circuit & $\bar{s}$ & """ + " & ".join(f"$S_{k}$" for k in range(N)) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_supports", body)


if __name__ == "__main__":
    main()
