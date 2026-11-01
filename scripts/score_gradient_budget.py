#!/usr/bin/env python3
"""Score the gradient-budget predictions (predictions/prospective_predictions.md, 2026-11-01) on
results/gradient_budget{,_dyn_*,_esc}.csv. From repository root:
``PYTHONPATH=. .venv/bin/python scripts/score_gradient_budget.py``."""

import csv
import glob
import math
import statistics as st
from collections import defaultdict


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def verdict(name, ok):
    print(f"  >> {name}: {'PASS' if ok else 'FAIL'}")


rows = list(csv.DictReader(open("results/gradient_budget.csv")))
cells = defaultdict(list)
for r in rows:
    cells[r["cell"]].append(r)
F = ["own", "att", "J0", "J0_ind", "Jc", "Jh", "S", "probe"]
lay = {c: {f: st.mean(float(r[f]) for r in rs) for f in F} | {"order": float(rs[0]["order"]), "n": len(rs)}
       for c, rs in cells.items()}
print(f"== (E) endpoints: {len(rows)} runs, {len(lay)} layouts ==")
print(f"  {'layout':24s} {'n':>2s} {'order':>5s} " + " ".join(f"{f:>7s}" for f in F))
for c, d in sorted(lay.items(), key=lambda kv: kv[1]["order"]):
    print(f"  {c:24s} {d['n']:2d} {d['order']:5.2f} " + " ".join(f"{d[f]:7.4f}" for f in F))
order = [d["order"] for d in lay.values()]
rho = {f: spearman([d[f] for d in lay.values()], order) for f in F}
print("  Spearman with order: " + "  ".join(f"{f} {rho[f]:+.2f}" for f in F))
verdict("H_own (rho <= -0.7)", rho["own"] <= -0.7)
verdict("H_zero (rho(J0) <= -0.7)", rho["J0"] <= -0.7)
verdict("H_trained (rho(Jc) <= -0.5)", rho["Jc"] <= -0.5)
ne, lr = lay["quantum+noent"], lay["quantum+entlast"]
d_att, d_own = math.log(lr["att"] / ne["att"]), math.log(lr["own"] / ne["own"])
print(f"  no CNOT -> last ring: dlog att {d_att:+.3f}, dlog own {d_own:+.3f}")
verdict("H_split (|dlog att| > |dlog own|)", abs(d_att) > abs(d_own))
rp = spearman([d["Jc"] for d in lay.values()], [d["probe"] for d in lay.values()])
print(f"  Spearman(Jc, probe) {rp:+.2f}")
verdict("H_probe (>= 0.7)", rp >= 0.7)
print(f"  refutation (rho J0 > -0.3 and rho Jc > -0.3): {'MET' if rho['J0'] > -0.3 and rho['Jc'] > -0.3 else 'not met'}")
ratio = [float(r["J0"]) / float(r["J0_ind"]) for r in rows if float(r["J0_ind"]) > 0]
print(f"  independence check J0/J0_ind: median {st.median(ratio):.3f}, range {min(ratio):.3f}-{max(ratio):.3f}")
print(f"  descriptive Spearman(Jh, order) {rho['Jh']:+.2f}")

dyn = [r for f in sorted(glob.glob("results/gradient_budget_dyn*.csv")) for r in csv.DictReader(open(f))]
if dyn:
    print(f"\n== (D) dynamics: {len(dyn)} checkpoints ==")
    by = defaultdict(dict)
    for r in dyn:
        by[(int(r["seed"]), int(r["epoch"]))][r["layout"]] = r
    for f in ("Jc", "own"):
        ok = tot = 0
        for key, d in sorted(by.items()):
            if set(d) >= {"noent", "default", "last"}:
                tot += 1
                ok += float(d["noent"][f]) > float(d["default"][f]) > float(d["last"][f])
        print(f"  {f}: ordering no CNOT > default > last in {ok}/{tot} cells ({ok / tot:.0%})")
        verdict(f"H_dyn[{f}] (>= 80%)", ok / tot >= 0.8)
    print("  descriptive, seed means by epoch (Jc | own | backbone_gnorm):")
    eps = sorted({k[1] for k in by})
    for lay_name in ("noent", "default", "last"):
        line = []
        for ep in eps:
            rs = [d[lay_name] for (s, e), d in by.items() if e == ep and lay_name in d]
            line.append(f"{ep}:{st.mean(float(r['Jc']) for r in rs):.3f}|{st.mean(float(r['own']) for r in rs):.3f}|"
                        f"{st.mean(float(r['backbone_gnorm']) for r in rs):.2f}")
        print(f"   {lay_name:8s} " + "  ".join(line))

try:
    esc = list(csv.DictReader(open("results/gradient_budget_esc.csv")))
except FileNotFoundError:
    esc = []
if esc:
    print("\n== escape-only endpoints (descriptive), seed means vs plain heads (seeds 0-4) ==")
    plain = {"noent": "quantum+noent", "default": "quantum", "last": "quantum+entlast"}
    for lay_name, cell in plain.items():
        e = [r for r in esc if r["layout"] == lay_name]
        p = [r for r in rows if r["cell"] == cell and int(r["seed"]) <= 4]
        print(f"  {lay_name:8s} " + "  ".join(
            f"{f} {st.mean(float(r[f]) for r in p):.4f}->{st.mean(float(r[f]) for r in e):.4f}"
            for f in ("own", "att", "J0", "Jc", "S")))
