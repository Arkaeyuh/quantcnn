#!/usr/bin/env python3
"""Appendix table: product order under the monotone ``half`` encoding, phi = (pi/2) tanh z + pi/2
(``results/half_probe.csv``, ``results/half_mlp_probe.csv``; the default ring's seeds 0-2 linear
probe in ``results/circuit_probe.csv``; CIFAR-10, 4,000 images, small CNN). Predictions and
criteria were committed before the runs (``predictions/prospective_predictions.md``): the default
ring's observed probe as anchor and the ring-family slope scaled by the attenuation ratio
ln E[cos^2 phi] (half) / ln E[cos^2 phi] (pi tanh). Prints the pre-registered statistics and the
post hoc slope comparison with the pi tanh encoding over the same layouts (seed bootstrap).
From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_encoding.py
"""

from __future__ import annotations

import csv
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tableA_family import _spearman  # noqa: E402
from table_utils import ms, paired, write_table  # noqa: E402

from scripts.summarize_v2 import FIVE_SEED_PROBES, five_seed_data  # noqa: E402

# key suffix -> (label, order); "" is the default ring
LAYOUTS = {"noent": ("no CNOT", 1.0), "pairs_last": ("pairs, last", 1.5), "": ("ring (default)", 2.5),
           "ladder2_last": ("ladder-2, last", 3.0), "chain_last": ("chain, last", 4.5),
           "entlast": ("ring, last", 5.25)}
ANCHOR, SLOPE = 70.2, -2.449  # recorded before the runs: observed default (3 seeds), scaled slope
MARGIN = 4.0


def _load(paths: list[str], prefix: str) -> dict:
    out = defaultdict(dict)
    for p in paths:
        for r in csv.DictReader(Path(p).open()):
            h = r["source_head"]
            if h == prefix or h.startswith(prefix + "+"):
                out[h][int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _slope(x: list[float], y: list[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def main():
    lin = _load(["results/circuit_probe.csv", "results/half_probe.csv"], "quantum+half")
    mlp = _load(["results/half_mlp_probe.csv"], "quantum+half")
    _, pi = five_seed_data(Path("results/protocol_v2"), FIVE_SEED_PROBES)
    for r in csv.DictReader(Path("results/family_probe.csv").open()):
        pi.setdefault(r["source_head"], {})[int(r["seed"])] = float(r["probe_test_acc"])
    key = lambda base, k: base + (f"+{k}" if k else "")  # noqa: E731
    orders, means, hits, half_vals, pi_vals, lines = [], [], 0, [], [], []
    for k, (label, o) in LAYOUTS.items():
        v, w, p = lin[key("quantum+half", k)], mlp[key("quantum+half", k)], pi[key("quantum", k)]
        pred = ANCHOR + SLOPE * (o - 2.5)
        m = st.mean(v.values())
        hits += abs(m - pred) <= MARGIN
        orders.append(o), means.append(m), half_vals.append(list(v.values())), pi_vals.append(list(p.values()))
        lines.append(f"{label} & {o:g} & {pred:.1f} & {ms(v.values())} & {m - pred:+.1f} & {ms(w.values())} & "
                     f"{ms(p.values())} & {len(v)} \\\\")
    rho = _spearman(orders, means)
    rho_mlp = _spearman(orders, [st.mean(mlp[key("quantum+half", k)].values()) for k in LAYOUTS])
    gap = paired(lin["quantum+half+noent"], lin["quantum+half"])
    s_half, s_pi = _slope(orders, means), _slope(orders, [st.mean(v) for v in pi_vals])
    rng = random.Random(0)
    boot = sorted(_slope(orders, [st.mean(rng.choices(v, k=len(v))) for v in half_vals])
                  - _slope(orders, [st.mean(rng.choices(v, k=len(v))) for v in pi_vals]) for _ in range(5000))
    lo, hi = boot[124], boot[4874]
    print(f"Spearman(order, probe): linear {rho:.3f}, MLP {rho_mlp:.3f}; within {MARGIN:g}: {hits}/{len(orders)}; "
          f"no CNOT - default {gap}; slope half {s_half:.2f}, pi tanh {s_pi:.2f}, difference {s_half - s_pi:+.2f} "
          f"[{lo:+.2f}, {hi:+.2f}]")
    body = r"""\begin{table}[h]
\centering
\caption{Product order under the monotone encoding $\enc{j}=\tfrac{\pi}{2}\tanh z_j+\tfrac{\pi}{2}$
(CIFAR-10, 4,000 images, small CNN, protocol of \cref{sec:setup}; every layout selected the head
learning rate $10^{-2}$). Each zero-rotation factor becomes $\cos\enc{j}=-\sin(\tfrac{\pi}{2}\tanh
z_j)$, which vanishes at $z_j=0$ where $\cos(\pi\tanh z_j)$ is 1; product orders are unchanged.
Predictions were committed before the new runs: the default ring's observed probe (3 existing seeds) plus a
slope of $-2.45$ per unit of order, the ring-family slope scaled by the ratio of
$\ln\mathbb{E}[\cos^2\enc{j}]$ under the two encodings measured on the trained default ring (0.71
vs 0.63, \cref{app:attenuation}). Spearman correlation between order and mean probe accuracy: """ + \
        f"${rho:.2f}$" + r""" (criterion $\le-0.8$; MLP probe """ + f"${rho_mlp:.2f}$" + r"""); """ + \
        f"{hits} of {len(orders)}" + r""" within 4 points (including the anchor); no CNOT $-$ default """ + gap + r""" (criterion $\ge2$). Post hoc, the
probe-on-order slope is """ + f"${s_half:.2f}$" + r""" here and """ + f"${s_pi:.2f}$" + r""" for the same layouts under
$\pi\tanh$ (last column), a difference of """ + f"${s_half - s_pi:+.2f}$" + r""" (seed bootstrap 95\% interval
""" + f"$[{lo:+.2f},{hi:+.2f}]$" + r""").}
\label{tab:encoding}
\small
\begin{tabular}{lccccccc}
\toprule
Layout & Order & Predicted & Probe & Obs.\ $-$ pred. & MLP probe & Probe, $\pi\tanh$ & $n$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_encoding", body)


if __name__ == "__main__":
    main()
