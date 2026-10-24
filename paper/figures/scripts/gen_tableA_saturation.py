#!/usr/bin/env python3
"""Appendix tables: bottleneck saturation as the route of the representation cost.

1. Diagnostics (``results/bottleneck_geometry.csv``): Spearman correlation between cell means of
   saturation S, effective dimensionality PRn and class-mean dimensionality CPRn and the cells'
   mean probe accuracy, over 38 cells.
2. Interventions (``results/mech2_*``, ``results/mech3_*`` and their geometry files): product heads
   and 8-qubit VQC heads trained with the anti-saturation penalty (and, for product heads, the
   decorrelation penalty) at the validation-selected weight, against the unregularized heads.
All criteria were committed before the computations and runs (``predictions/prospective_predictions.md``).
From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_saturation.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tableA_family import _spearman  # noqa: E402
from table_utils import ms, paired, write_table  # noqa: E402

from scripts.summarize_v2 import FIVE_SEED_PROBES, five_seed_data  # noqa: E402

GROUPS = {"q8": "VQC, 8 qubits ($\\pi\\tanh$)", "half": "VQC, half encoding", "phase": "VQC, phase encoding",
          "q12": "VQC, 12 qubits", "classical": "classical product heads"}
# label, unregularized probe key, regularized key prefix, baseline run grids, run-name stem, unreg geometry cell
PRODUCTS = [("cos (1)", "cos", "cos", ("cifar_classical", "cifar_seeds"), "cos"),
            ("default supports (2.5)", "cosprod", "cosprod", ("cifar_cosprod", "cifar_seeds"), "cosprod"),
            ("last-ring supports (5.25)", "cosprod_last", "cosprod_last", ("cifar_cosprod", "cifar_seeds"), "cosprod_last")]
CIRCUITS = [("no CNOT (1)", "quantum+noent", ("cifar_circuit", "cifar_seeds"), "quantum_noent"),
            ("ring, default (2.5)", "quantum", ("cifar_val", "cifar_seeds"), "quantum"),
            ("ring, last (5.25)", "quantum+entlast", ("cifar_entangle", "cifar_seeds"), "quantum_entlast")]


def _rows(path: str) -> dict:
    out = defaultdict(dict)
    for r in csv.DictReader(Path(path).open()):
        out[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _geom(path: str, key=lambda r: r["cell"]) -> dict:
    out = defaultdict(list)
    for r in csv.DictReader(Path(path).open()):
        out[key(r)].append(r)
    return out


def _train(grids, stem: str) -> float:
    v = [json.loads(f.read_text())["train_clean_acc"] * 100 for g in grids
         for f in Path("results/protocol_v2", g).glob(f"cifar10_subset4000_{stem}_lr0.01_s*/final.json")]
    assert len(v) == 5, (stem, len(v))
    return st.mean(v)


def _selected(heads: dict, prefix: str, reg: str) -> str:
    """The regularized head key with five seeds (the validation-selected weight)."""
    keys = [k for k in heads if k.startswith(prefix + f"+{reg}") and k.count("+") == prefix.count("+") + 1
            and len(heads[k]) == 5]
    assert len(keys) == 1, (prefix, reg, keys)
    return keys[0]


def diagnostics() -> str:
    cells = _geom("results/bottleneck_geometry.csv")
    m = {c: {k: st.mean(float(r[k]) for r in R) for k in ("probe", "S", "PRn", "CPRn")} | {"group": R[0]["group"]}
         for c, R in cells.items()}
    rho = lambda gs, k: _spearman([v[k] for v in m.values() if v["group"] in gs],  # noqa: E731
                                  [v["probe"] for v in m.values() if v["group"] in gs])
    lines = []
    for g, label in GROUPS.items():
        n = sum(v["group"] == g for v in m.values())
        lines.append(f"{label} & {n} & ${rho({g}, 'S'):.2f}$ & ${rho({g}, 'PRn'):.2f}$ & ${rho({g}, 'CPRn'):.2f}$ \\\\")
    q = {"q8", "half", "phase", "q12"}
    lines.append(r"\midrule")
    lines.append(f"all VQC cells & {sum(v['group'] in q for v in m.values())} & ${rho(q, 'S'):.2f}$ & "
                 f"${rho(q, 'PRn'):.2f}$ & ${rho(q, 'CPRn'):.2f}$ \\\\")
    print("diagnostics:", {k: round(rho(q, k), 3) for k in ("S", "PRn", "CPRn")},
          {k: round(rho({"classical"}, k), 3) for k in ("S", "PRn", "CPRn")})
    return r"""\begin{table}[h]
\centering
\caption{Bottleneck geometry of the trained backbones (CIFAR-10, 4,000 images, small CNN; test
embeddings $z$ of every probed run). $S$: fraction of coordinates with $|\tanh z|>0.9$; PRn:
participation ratio of the correlation matrix of $z$, divided by the width; CPRn: participation ratio
of the between-class covariance of standardized $z$, divided by $\min(q,9)$. Entries: Spearman
correlation between the cell means and the cells' mean probe accuracy. Criteria ($\rho\le-0.7$ for $S$,
$\rho\ge0.7$ for PRn and CPRn, separately over the VQC and the classical cells) were committed before
the computation.}
\label{tab:geometry}
\small
\begin{tabular}{lcccc}
\toprule
Cells & $n$ & $S$ & PRn & CPRn \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def interventions() -> str:
    p_reg, c_reg = _rows("results/mech2_probe.csv"), _rows("results/mech3_probe.csv")
    g_reg = _geom("results/mech2_geometry.csv", lambda r: r["cell"].split("@")[0])
    g_reg |= _geom("results/mech3_geometry.csv", lambda r: r["cell"].split("@")[0])
    g_base = _geom("results/bottleneck_geometry.csv")
    base = defaultdict(dict)
    for f in ("results/classical_probe.csv", "results/cosprod_probe.csv", "results/seeds_probe.csv"):
        for r in csv.DictReader(Path(f).open()):
            if r["source_head"] in ("cos", "cosprod", "cosprod_last") and r["source_lr"] == "0.01":
                base[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    _, b5 = five_seed_data(Path("results/protocol_v2"), FIVE_SEED_PROBES)
    S = lambda R: st.mean(float(r["S"]) for r in R)  # noqa: E731
    lines = [r"\multicolumn{7}{l}{\emph{Classical product heads}} \\"]
    for label, key, prefix, grids, stem in PRODUCTS:
        sat, dec = _selected(p_reg, prefix, "sat"), _selected(p_reg, prefix, "decor")
        tr = lambda k: st.mean(json.loads(f.read_text())["train_clean_acc"] * 100 for g in ("cifar_mech2", "cifar_mech2_seeds")  # noqa: E731
                               for f in Path("results/protocol_v2", g).glob(
                                   f"cifar10_subset4000_{stem}_{k.split('+')[-1]}_lr0.01_s*/final.json"))
        lines.append(f"{label} & {ms(base[key].values())} & {paired(p_reg[sat], base[key])} & {paired(p_reg[dec], base[key])} & "
                     f"{S(g_base[key]):.2f} $\\to$ {S(g_reg[sat]):.2f} & {_train(grids, stem):.1f} $\\to$ {tr(sat):.1f} & "
                     f"{sat.split('+sat')[-1]} \\\\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{7}{l}{\emph{8-qubit VQC heads}} \\")
    for label, key, grids, stem in CIRCUITS:
        sat = _selected(c_reg, key, "sat")
        tr = st.mean(json.loads(f.read_text())["train_clean_acc"] * 100 for g in ("cifar_mech3", "cifar_mech3_seeds")
                     for f in Path("results/protocol_v2", g).glob(
                         f"cifar10_subset4000_{stem}_{sat.split('+')[-1]}_lr0.01_s*/final.json"))
        lines.append(f"{label} & {ms(b5[key].values())} & {paired(c_reg[sat], b5[key])} & --- & "
                     f"{S(g_base[key]):.2f} $\\to$ {S(g_reg[sat]):.2f} & {_train(grids, stem):.1f} $\\to$ {tr:.1f} & "
                     f"{sat.split('+sat')[-1]} \\\\")
        print(label, paired(c_reg[sat], b5[key]))
    rt = _rows("results/retest_probe.csv")
    g_rt = _geom("results/retest_geometry.csv", lambda r: r["cell"].split("@")[0])
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{7}{l}{\emph{Replication on seeds 5--9, both arms (guard: no loss of fit)}} \\")
    for label, key, stem in (("VQC, ring, last (5.25)", "quantum+entlast", "quantum_entlast"),
                             ("last-ring products (5.25)", "cosprod_last", "cosprod_last")):
        trn = lambda st_: st.mean(json.loads(f.read_text())["train_clean_acc"] * 100 for f in  # noqa: E731
                                 Path("results/protocol_v2/cifar_lastring_retest").glob(f"cifar10_subset4000_{st_}_lr0.01_s*/final.json"))
        assert len(rt[key]) == len(rt[key + "+sat10"]) == 5, key
        lines.append(f"{label} & {ms(rt[key].values())} & {paired(rt[key + '+sat10'], rt[key])} & --- & "
                     f"{S(g_rt[key]):.2f} $\\to$ {S(g_rt[key + '+sat10']):.2f} & {trn(stem):.1f} $\\to$ {trn(stem + '_sat10'):.1f} & 10 \\\\")
        print("retest", label, paired(rt[key + "+sat10"], rt[key]))
    return r"""\begin{table*}[t]
\centering
\caption{Interventions on bottleneck saturation (CIFAR-10, 4,000 images, small CNN, head learning rate
$10^{-2}$, 5 seeds). Each head is trained with a penalty on its bottleneck added to the loss:
anti-saturation, $\lambda\,\mathrm{mean}(\mathrm{relu}(\tanh^2z-0.81))$, nonzero only where
$|\tanh z|>0.9$; decorrelation, $\lambda$ times the mean squared off-diagonal entry of the batch
correlation matrix of $z$. $\lambda$ was selected on validation accuracy from $\{3,10\}$
(anti-saturation) and $\{1,3\}$ (decorrelation); every product head selected 10 and 1. $\Delta$:
paired probe difference to the unregularized head (linear probe). Manipulation checks, a training guard
and the causal criteria (mean gains of at least $+3$ and $+1.5$ probe points on the order-5.25 and
order-2.5 heads, positive in at least 4 of 5 seeds, with less than half the order-5.25 gain on the
order-1 head) were committed before the runs. Anti-saturation meets the gain, sign and specificity
criteria on both head families and decorrelation does not; the first last-ring rows exceed the committed
training guard (clean training accuracy within 10 points) by improving fit, so they are inconclusive as
written. The replication on fresh seeds 5--9, committed with the guard stated as no loss of fit (penalty
fixed at 10, both arms re-run), meets every criterion (\cref{app:saturation}).}
\label{tab:saturation}
\small
\begin{tabular}{lcccccc}
\toprule
Head (order) & Probe & $\Delta$ anti-sat. & $\Delta$ decorr. & $S$ & Clean train & $\lambda$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def main():
    write_table("tableA_geometry", diagnostics())
    write_table("tableA_saturation", interventions())


if __name__ == "__main__":
    main()
