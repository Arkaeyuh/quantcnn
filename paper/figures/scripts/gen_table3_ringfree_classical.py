#!/usr/bin/env python3
"""Table 3: the ring-free VQC head vs classical trigonometric heads (CIFAR-10, 4000 images):
test accuracy, backbone probe accuracy, and paired differences from the ring-free circuit.
From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_table3_ringfree_classical.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import diffs, ms, paired, write_table  # noqa: E402

from scripts.summarize_v2 import FIVE_SEED_PROBES, five_seed_data  # noqa: E402

REF = "quantum+noent"
HEADS = {
    "quantum+noent": "VQC, no CNOT rings",
    "cos": "cos-only (restriction)",
    "cossin": "cos+sin (unrestricted extension)",
    "linear": "linear (reference)",
}


def main():
    test, probes = five_seed_data(Path("results/protocol_v2"), FIVE_SEED_PROBES)
    rows, trig = [], {"test": [], "probe": []}
    note = ""
    for key, name in HEADS.items():
        t, p = test[key], probes[key]
        assert sorted(t) == sorted(p), key
        mark = ""
        if key == REF:
            dt = dp = "---"
        else:
            seeds = sorted(set(t) & set(test[REF]))
            rt, rp = {s: test[REF][s] for s in seeds}, {s: probes[REF][s] for s in seeds}
            dt, dp = paired({s: t[s] for s in seeds}, rt), paired({s: p[s] for s in seeds}, rp)
            if key != "linear":
                trig["test"].append(st.mean(diffs({s: t[s] for s in seeds}, rt)))
                trig["probe"].append(st.mean(diffs({s: p[s] for s in seeds}, rp)))
            if len(seeds) < len(test[REF]):
                mark = "$^a$"
                note = (f"$^a${len(seeds)} seeds; differences are against the ring-free VQC on the same seeds "
                        f"(test {st.mean(rt.values()):.1f}, probe {st.mean(rp.values()):.1f}).")
        rows.append(f"{name}{mark} & {len(t)} & {ms(t.values())} & {ms(p.values())} & {dt} & {dp} \\\\")
    span = {k: f"{min(-x for x in v):.1f}--{max(-x for x in v):.1f}" for k, v in trig.items()}
    body = r"""\begin{table*}[t]
\centering
\caption{Without CNOT rings the VQC head is exactly a classical per-coordinate sinusoidal model,
$\mathrm{logits} = W[\alpha_j\cos a_j + \beta_j\sin a_j]_{j=1}^{8} + b$ with $a_j = \pi\tanh z_j$
and $\alpha_j, \beta_j$ set by the trained rotations, for all rotation angles. Classical heads
$W[\cos a_j]_j + b$ (a restriction) and $W[\cos a_j, \sin a_j]_j + b$ (an unrestricted
extension), trained end to end with the same recipe and per-head validation lr selection,
trail it by """ + span["probe"] + r""" probe points and """ + span["test"] + r""" test points.
CIFAR-10 with 4,000 training images; test accuracy of the full model and of the common linear
probe on the bottleneck (\%, mean $\pm$ SD over $n$ seeds); differences in percentage points,
paired over seeds.}
\label{tab:ringfree}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
& & & & \multicolumn{2}{c}{$-$ ring-free VQC} \\
\cmidrule(lr){5-6}
Head & $n$ & Test & Probe & Test & Probe \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\smallskip
{\footnotesize """ + note + r"""}
\end{table*}
"""
    write_table("table3_ringfree_classical", body)


if __name__ == "__main__":
    main()
