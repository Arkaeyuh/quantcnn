#!/usr/bin/env python3
"""Table 2: controls that do not recover the backbone probe gap (CIFAR-10, 4000 images,
seeds 0-2), with ring removal as the contrast. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_table2_controls.py
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import paired, write_table  # noqa: E402

from scripts.summarize_v2 import BASE_PROBES, circuit_data, mech_data, protocol_cells  # noqa: E402

ROOT = Path("results/protocol_v2")
SEEDS = (0, 1, 2)


def _seeds(d: dict) -> dict:
    return {k: (v[0] if isinstance(v, tuple) else v) for k, v in d.items() if k in SEEDS}


def _final_gnorm(run: Path) -> float:
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    return float(rows[-1]["backbone_gnorm"])


def _ms2(xs) -> str:
    return f"{st.mean(xs):.2f} \\pm {st.stdev(xs):.2f}"


def main():
    runs, mprobes = mech_data(ROOT / "cifar_mech", ROOT / "cifar_val", Path("results/mech_probe.csv"),
                              Path("results/backbone_probe.csv"))
    _, cprobes = circuit_data(ROOT / "cifar_circuit", BASE_PROBES)
    probe = {k[2]: _seeds(v) for k, v in cprobes.items() if k[:2] == ("cifar10", 4000)}
    q, lin = probe["quantum"], probe["linear"]

    # Backbone lr at fixed head lr: VQC - linear probe at each matched backbone lr where the VQC trains.
    bb_rows, collapsed_at = [], []
    for (head, bb, lr), by in sorted(runs.items()):
        if head != "quantum" or lr != 0.01:
            continue
        if st.mean(x[1] for x in by.values()) < 20:
            collapsed_at.append(bb)
        elif ("linear", bb, 0.03) in mprobes:
            bb_rows.append((bb, paired(_seeds(mprobes[(head, bb, lr)]), _seeds(mprobes[("linear", bb, 0.03)]))))

    # Final-epoch backbone gradient norm at each head's validation-selected lr.
    cells = protocol_cells(ROOT / "cifar_val")
    gn = {}
    for head in ("quantum", "linear"):
        by_lr = cells[(4000, head)]
        lr = max(by_lr, key=lambda k: st.mean(x[1] for x in by_lr[k]))
        gn[head] = [_final_gnorm(ROOT / "cifar_val" / f"cifar10_subset4000_{head}_lr{lr:g}_s{s}") for s in SEEDS]

    # Frozen circuit vs trained circuit (both at backbone lr 0.005, head lr 0.01).
    frozen = {k: v for k, v in mprobes.items() if k[0] == "quantum+frozen"}
    assert len(frozen) == 1, frozen.keys()
    fz = _seeds(next(iter(frozen.values())))
    trained = _seeds(mprobes[("quantum", 0.005, 0.01)])

    bn = {}
    for r in csv.DictReader(Path("results/backbone_probe.csv").open()):
        if r["dataset"] == "cifar10" and r["train_subset_size"] == "4000" and int(r["seed"]) in SEEDS:
            bn.setdefault(r["source_head"], {})[int(r["seed"])] = float(r["probe_test_acc"])

    rows = [
        ("Reference: default VQC", "VQC $-$ linear", paired(q, lin), "the gap under study"),
        *[("Backbone lr$^a$" if i == 0 else "", f"VQC $-$ linear, backbone lr {bb:g}", d,
           "gap persists at every tested backbone lr where the VQC trains" if i == 0 else "")
          for i, (bb, d) in enumerate(bb_rows)],
        ("Freeze circuit at init", "frozen $-$ trained circuit", paired(fz, trained),
         "gap persists with rotations frozen at initialization"),
        ("BatchNorm before encoding", "VQC+BN $-$ VQC", paired(bn["quantum+bn"], bn["quantum"]),
         "pre-encoding BN does not recover the probe"),
        ("", "VQC+BN $-$ linear", paired(bn["quantum+bn"], bn["linear"]), ""),
        ("Monotone encoding$^b$", "monotone $-$ default", paired(probe["quantum+half"], q),
         "neither encoding recovers more than 0.7 points"),
        ("", "symmetric control $-$ default", paired(probe["quantum+half_sym"], q), ""),
        ("", "monotone $-$ symmetric control", paired(probe["quantum+half"], probe["quantum+half_sym"]), ""),
        ("Contrast: remove CNOT rings", "ring-free $-$ default", paired(probe["quantum+noent"], q),
         "recovers most of the gap (\\S\\ref{sec:ring})"),
        ("", "ring-free $-$ linear", paired(probe["quantum+noent"], lin), ""),
    ]
    lines = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines.insert(1, "\\midrule")
    lines.insert(-2, "\\midrule")
    notes = (
        "$^a$Head lr fixed at 0.01 (VQC) and 0.03 (linear) while the backbone lr varies; the VQC collapses "
        f"at the tested backbone lrs {' and '.join(f'{b:g}' for b in collapsed_at)}. Other rows: each head at "
        "its validation-selected head lr, backbone lr $0.5\\times$ head lr. "
        "$^b$Monotone: $R_y(\\frac{\\pi}{2}\\tanh z + \\frac{\\pi}{2})$; symmetric control: "
        "$R_y(\\frac{\\pi}{2}\\tanh z)$, same range width; default: $R_y(\\pi\\tanh z)$. "
        "Not a probe difference: the final-epoch pre-clip backbone gradient norm is larger with the VQC "
        f"head than with the linear head (${_ms2(gn['quantum'])}$ vs ${_ms2(gn['linear'])}$)."
    )
    body = r"""\begin{table*}[t]
\centering
\caption{Controls on CIFAR-10 with 4,000 training images. Entries are differences in the test
accuracy of the common linear probe on the frozen 8-dimensional bottleneck, in percentage
points, paired over 3 seeds (mean $\pm$ SD). The five controls leave a substantial probe gap;
removing the CNOT rings recovers most of it.}
\label{tab:controls}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{>{\raggedright\arraybackslash}p{2.8cm}>{\raggedright\arraybackslash}p{4.2cm}c>{\raggedright\arraybackslash}p{4.4cm}}
\toprule
Control & Comparison & Probe $\Delta$ & Observation \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\smallskip
\begin{minipage}{0.97\textwidth}\footnotesize """ + notes + r"""\end{minipage}
\end{table*}
"""
    write_table("table2_controls", body)


if __name__ == "__main__":
    main()
