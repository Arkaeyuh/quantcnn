#!/usr/bin/env python3
"""Appendix table: the phase encoding (H then RZ(pi tanh z)), where every zero-rotation readout
vanishes (``results/phase_probe.csv``, ``results/phase_mlp_probe.csv``; effective orders from
``results/effective_order_init_phase.csv`` and ``results/effective_order_trained_phase.csv``;
CIFAR-10, 4,000 images, small CNN). The two competing predictions (effective order at the
initialization distribution vs the R_y product order) were committed before the runs
(``predictions/prospective_predictions.md``). The committed initialization orders
(``results/effective_order_init_phase_committed.csv``) came from code that zeroed readouts with
total visible weight below 1e-12, found after the runs; the corrected values are shown beside
them. The trained effective order is post hoc. Prints the pre-registered statistics. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_phase.py
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
from table_utils import ms, write_table  # noqa: E402

# key suffix -> (label, R_y product order); "" is the default ring
LAYOUTS = {"noent": ("no CNOT", 1.0), "star": ("star", 1.0), "chain": ("chain", 1.5), "": ("ring (default)", 2.5),
           "chain_last": ("chain, last", 4.5), "entlast": ("ring, last", 5.25)}


def _probes(path: str) -> dict:
    out = defaultdict(dict)
    for r in csv.DictReader(Path(path).open()):
        out[r["source_head"]][int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _orders(path: str) -> dict:
    out = defaultdict(list)
    for r in csv.DictReader(Path(path).open()):
        out[r["head"]].append(r)
    return out


def main():
    lin, mlp = _probes("results/phase_probe.csv"), _probes("results/phase_mlp_probe.csv")
    init, trained = _orders("results/effective_order_init_phase.csv"), _orders("results/effective_order_trained_phase.csv")
    committed = _orders("results/effective_order_init_phase_committed.csv")
    m = lambda rows, c: st.mean(float(r[c]) for r in rows)  # noqa: E731
    lines, o_com, o_init, o_ry, o_tr, means, means_mlp = [], [], [], [], [], [], []
    for k, (label, ry) in LAYOUTS.items():
        h = "quantum+phase" + (f"+{k}" if k else "")
        key = "quantum" + (f"+{k}" if k else "")
        oc, oi, ot = m(committed[key], "order_eff"), m(init[key], "order_eff"), m(trained[h], "order_eff")
        train = [json.loads(f.read_text())["train_clean_acc"] * 100
                 for g in ("cifar_phase", "cifar_phase_seeds")
                 for f in Path("results/protocol_v2", g).glob(f"cifar10_subset4000_quantum_phase{'_' + k if k else ''}_lr0.01_s*/final.json")]
        o_com.append(oc), o_init.append(oi), o_ry.append(ry), o_tr.append(ot)
        means.append(st.mean(lin[h].values())), means_mlp.append(st.mean(mlp[h].values()))
        lines.append(f"{label} & {oc:.2f} & {oi:.2f} & {ry:g} & {ot:.2f} & {m(trained[h], 'y_share'):.2f} & "
                     f"{ms(lin[h].values())} & {ms(mlp[h].values())} & {st.mean(train):.1f} & {len(lin[h])} \\\\")
    r_com, r_init, r_ry, r_tr = (_spearman(o, means) for o in (o_com, o_init, o_ry, o_tr))
    r_com_m, r_ry_m = _spearman(o_com, means_mlp), _spearman(o_ry, means_mlp)
    r_pred, r_pred_c = _spearman(o_com, o_ry), _spearman(o_init, o_ry)
    fit = lambda o: 76.69 - 3.22 * o  # noqa: E731  (ring-family fit used for the committed point predictions)
    hit_com = sum(abs(y - fit(round(o, 2))) <= 4 for y, o in zip(means, o_com))
    hit_ry = sum(abs(y - fit(o)) <= 4 for y, o in zip(means, o_ry))
    print(f"Spearman with probe: committed init order {r_com:+.3f} (MLP {r_com_m:+.3f}); corrected {r_init:+.3f}; "
          f"R_y order {r_ry:+.3f} (MLP {r_ry_m:+.3f}); trained {r_tr:+.3f}; between predictors {r_pred:+.3f} "
          f"(corrected {r_pred_c:+.3f}); within 4 points: init {hit_com}/6, R_y {hit_ry}/6")
    f2 = lambda x: f"${x:+.2f}$".replace("+", "") if x < 0 else f"${x:+.2f}$"  # noqa: E731
    body = r"""\begin{table}[h]
\centering
\caption{The phase encoding, $H$ then $R_z(\pi\tanh z_j)$ (CIFAR-10, 4,000 images, small CNN, protocol of
\cref{sec:setup}; every layout selected the head learning rate $10^{-2}$). Every zero-rotation readout
vanishes, so \cref{eq:identity} does not apply. Two competing predictions were committed before the
runs: the effective order (\cref{app:efforder}) at the initialization distribution
$\theta\sim\mathcal{U}(-0.1,0.1)$, and the $R_y$ product order. As committed (first column), the
initialization order correlated """ + f2(r_pred) + r""" with the $R_y$ order and """ + f2(r_com) + r""" with probe accuracy (criterion
$\le-0.8$; failed; MLP probe """ + f2(r_com_m) + r"""), and its point predictions put """ + f"{hit_com}" + r""" of 6 layouts within 4 points;
the $R_y$ order correlated """ + f2(r_ry) + r""" (passed; MLP probe """ + f2(r_ry_m) + r""") with """ + f"{hit_ry}" + r""" of 6 within 4 points.
After the runs we found that the committed values came from code that set a readout's order to 0
when its total visible weight was below $10^{-12}$, which happens for the last-layer layouts at
small rotations; with the definition applied (second column, a retrospective correction) the
initialization order equals the $R_y$ order for those layouts and correlates """ + f2(r_init) + r""" with probe
accuracy. Post hoc: the effective order at the trained rotations, and the share of weight on strings
invisible to the encoded state (about 0.99 at initialization, averaged over readouts), with a
rank correlation of """ + f2(r_tr) + r""" between trained order and probe accuracy. Train: clean training accuracy (\%).}
\label{tab:phase}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{Predictors} & \multicolumn{2}{c}{Trained (post hoc)} & & & & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
& \multicolumn{2}{c}{Init.\ order} & & & & & & & \\
Layout & committed & corrected & $R_y$ order & Order & Invisible & Probe & MLP probe & Train & $n$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_phase", body)


if __name__ == "__main__":
    main()
