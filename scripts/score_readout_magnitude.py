#!/usr/bin/env python3
"""Score the decoder-compensation pre-registration (predictions/prospective_predictions.md, 2026-10-30)
on ``results/readout_magnitude.csv``. From repository root: ``python scripts/score_readout_magnitude.py``."""

import csv
import statistics as st
from collections import defaultdict

from scipy.stats import spearmanr

rows = list(csv.DictReader(open("results/readout_magnitude.csv")))
cells = defaultdict(list)
for r in rows:
    cells[r["cell"]].append(r)
keys = ["A", "W", "L", "A_hat", "A_min", "middle", "probe", "bound_nonvacuous"]
mean = {c: {k: st.mean(float(r[k]) for r in rr if r[k] != "nan") if any(r[k] != "nan" for r in rr) else float("nan")
            for k in keys} | {"order": float(rr[0]["order"]), "n": len(rr)} for c, rr in cells.items()}
print(f"{'cell':22s} order  n    A     A_min  A_hat    W      L    middle probe nonvac")
for c, m in sorted(mean.items(), key=lambda x: x[1]["order"]):
    print(f"{c:22s} {m['order']:5.2f} {m['n']}  {m['A']:.3f}  {m['A_min']:.3f}  {m['A_hat']:.3f}  {m['W']:.3f}  "
          f"{m['L']:.3f}  {m['middle']:.3f}  {m['probe']:.1f}  {m['bound_nonvacuous']:.2f}")
o = [m["order"] for m in mean.values()]
rho = {k: spearmanr(o, [m[k] for m in mean.values()])[0] for k in ("A", "W", "L", "A_hat", "probe")}
print("Spearman with order:", {k: round(v, 3) for k, v in rho.items()})
nc, last = mean["quantum+noent"], mean["quantum+entlast"]
floor = last["A"] >= 0.5 * nc["A"] and min(m["A"] for m in mean.values()) >= 0.3
nocomp = rho["W"] <= 0.5 and last["W"] / nc["W"] <= 1.5
print(f"A(last)/A(noent) {last['A'] / nc['A']:.3f}; min cell A {min(m['A'] for m in mean.values()):.3f} -> H_floor {'PASS' if floor else 'FAIL'}")
print(f"rho(order,W) {rho['W']:.3f}; W(last)/W(noent) {last['W'] / nc['W']:.3f} -> H_nocomp {'PASS' if nocomp else 'FAIL'}")
print("refutation (compensation):", rho["A"] <= -0.8 and rho["W"] >= 0.8)
