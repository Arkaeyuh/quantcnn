#!/usr/bin/env python3
"""Appendix table: effective product order beyond zero rotations
(``results/effective_order_{trained,random_pi}.csv`` from ``scripts/effective_order.py``),
against probe accuracy (CIFAR-10, 4,000 images, small CNN). From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_effective_order.py
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tableA_family import FIT, _spearman  # noqa: E402
from table_utils import write_table  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402
from quantcnn.models.torch_vqc import entangle_layout  # noqa: E402
from scripts.effective_order import LAYOUTS  # noqa: E402
from scripts.summarize_v2 import FIVE_SEED_PROBES, five_seed_data  # noqa: E402

LABELS = {
    "quantum+noent": "no CNOT", "quantum+pairs": "pairs", "quantum+pairs_last": "pairs, last",
    "quantum": "ring (default)", "quantum+L2": "ring, 2 layers", "quantum+L1": "ring, 1 layer",
    "quantum+entlast": "ring, last", "quantum+star": "star", "quantum+chain": "chain",
    "quantum+star_in_last": "star-in, last", "quantum+ladder2_last": "ladder-2, last",
    "quantum+brick": "brickwork", "quantum+chain_last": "chain, last",
}
FAMILY = {"quantum+star", "quantum+chain", "quantum+star_in_last", "quantum+ladder2_last", "quantum+brick",
          "quantum+chain_last"}


def _order0(name: str) -> float:
    layers, ent = LAYOUTS[name]
    layout = entangle_layout(ent)
    return 1.0 if layout is None else st.mean(len(s) for s in cnot_ring_supports(8, layers, layout[1], layout[0]))


def _load(path: str) -> dict:
    rows = defaultdict(list)
    for r in csv.DictReader(Path(path).open()):
        rows[r["head"]].append(r)
    return rows


def main():
    trained, rand = _load("results/effective_order_trained.csv"), _load("results/effective_order_random_pi.csv")
    _, probes = five_seed_data(Path("results/protocol_v2"), FIVE_SEED_PROBES)
    for r in csv.DictReader(Path("results/family_probe.csv").open()):
        probes.setdefault(r["source_head"], {})[int(r["seed"])] = float(r["probe_test_acc"])
    m = lambda rs, c: st.mean(float(r[c]) for r in rs)  # noqa: E731
    lines, o0, ot, orand, pr = [], [], [], [], []
    for name in LAYOUTS:
        p = st.mean(probes[name].values())
        a, b, c = _order0(name), m(trained[name], "order_eff"), m(rand[name], "order_eff")
        o0.append(a), ot.append(b), orand.append(c), pr.append(p)
        tag = r"$^\ast$" if name in FAMILY else ""
        lines.append(f"{LABELS[name]}{tag} & {a:.2f} & {b:.2f} & {c:.2f} & {m(rand[name], 'y_share'):.2f} & {p:.1f} \\\\")
    rho0, rhot, rhor = _spearman(o0, pr), _spearman(ot, pr), _spearman(orand, pr)
    brick = list(LAYOUTS).index("quantum+brick")
    resid0 = pr[brick] - (FIT[0] + FIT[1] * o0[brick])
    resid_t = pr[brick] - (FIT[0] + FIT[1] * ot[brick])
    print(f"Spearman(probe, order): theta=0 {rho0:.3f}; trained {rhot:.3f}; random U(-pi,pi) {rhor:.3f}; "
          f"brickwork residual {resid0:+.1f} -> {resid_t:+.1f}")
    body = r"""\begin{table}[h]
\centering
\caption{Effective order beyond zero rotations (CIFAR-10, 4,000 images, small CNN; $^\ast$: second
family). For rotations $\theta$, readout $k$ is $\langle\psi(z)|O_k|\psi(z)\rangle$ with
$O_k=U(\theta)^\dagger Z_kU(\theta)=\sum_P c_PP$ over Pauli strings; strings with a $Y$ factor
vanish on the encoded state, and an $X$ or $Z$ factor on qubit $j$ makes the readout depend on
input $j$. The effective order is the $c_P^2$-weighted mean number of inputs a visible string
touches, averaged over readouts; at $\theta=0$ it is the product order of \cref{eq:identity}.
Trained: the trained rotations (5 seeds; 3 for the 1- and 2-layer rings). Random: $\theta\sim
\mathcal{U}(-\pi,\pi)$ (20 draws), with the share of weight on invisible ($Y$) strings. Spearman
correlation with probe accuracy over the 13 layouts: """ + f"{rho0:.2f}" + r""" at $\theta=0$, """ + \
        f"{rhot:.2f}" + r""" trained (post hoc), """ + f"{rhor:.2f}" + r""" random.}
\label{tab:efforder}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{Effective order} & & \\
\cmidrule(lr){2-4}
Layout & $\theta=0$ & trained & random & $Y$ share (random) & Probe \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_effective_order", body)


if __name__ == "__main__":
    main()
