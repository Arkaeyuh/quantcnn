#!/usr/bin/env python3
"""Score the SVHN collapse pre-registration (predictions/prospective_predictions.md, 2026-10-30).

Collapse = best-checkpoint validation accuracy < 0.25. Reads ``runs/protocol_v2/svhn*`` and
``results/init_gradients_svhn.csv``. From repository root: ``python scripts/score_svhn_collapse.py``.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

GRIDS = ["svhn", "svhn_seeds", "svhn_collapse"]
LAYOUTS = {"quantum_noent": "noent", "quantum_pairs": "pairs", "quantum": "default",
           "quantum_entlast": "last", "cosprod": "cosprod"}


def runs():
    for g in GRIDS:
        for d in sorted(glob.glob(f"runs/protocol_v2/{g}/*")):
            m = re.match(r"svhn_subset4000_(.+)_lr([\d.e-]+)_s(\d+)$", os.path.basename(d))
            if not m or m.group(1) not in LAYOUTS or not os.path.exists(f"{d}/final.json"):
                continue
            with open(f"{d}/final.json") as f:
                val = json.load(f)["val_acc"]
            with open(f"{d}/metrics.csv") as f:
                g_norm = [float(r["backbone_gnorm"]) for r in csv.DictReader(f)]
            rho = st.mean(g_norm[1:10]) / g_norm[0]
            yield dict(grid=g, layout=LAYOUTS[m.group(1)], lr=float(m.group(2)), seed=int(m.group(3)),
                       val=val, collapsed=val < 0.25, rho=rho)


def auc(pos, neg):
    """P(score_pos > score_neg) with ties counted half."""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    rs = list(runs())
    print("== (1) init sensitivity ==")
    by = defaultdict(list)
    with open("results/init_gradients_svhn.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        by[r["layout"]].append(r)
    for lay, rr in by.items():
        J, Jh = st.mean(float(r["J"]) for r in rr), st.mean(float(r["J_hat"]) for r in rr)
        print(f"{lay:8s} J {J:.3f}  Jhat {Jh:.3f}  ratio {J / Jh:.3f}  in band: {0.75 <= J / Jh <= 1.33}")
    seeds = sorted({r["seed"] for r in rows})
    J = {(r["layout"], r["seed"]): float(r["J"]) for r in rows}
    holds = [J["noent", s] > J["default", s] > J["last", s] for s in seeds]
    print(f"H_init-order holds in {sum(holds)}/{len(seeds)} seeds")

    print("== (2) gradient death ==")
    col = [r["rho"] for r in rs if r["collapsed"]]
    ok = [r["rho"] for r in rs if not r["collapsed"]]
    a = auc([-x for x in col], [-x for x in ok])
    mc, mo = st.median(col), st.median(ok)
    print(f"runs {len(rs)} collapsed {len(col)}  AUC(-rho) {a:.3f}  median rho collapsed {mc:.3f} "
          f"non-collapsed {mo:.3f} ratio {mc / mo:.3f}")
    print(f"H_gdeath: {'PASS' if a >= 0.9 and mc <= 0.5 * mo else 'FAIL'}; refuted: {a <= 0.6}")
    for lay in LAYOUTS.values():
        c = [r["rho"] for r in rs if r["collapsed"] and r["layout"] == lay]
        o = [r["rho"] for r in rs if not r["collapsed"] and r["layout"] == lay]
        if c and o:
            print(f"  within {lay:8s}: n {len(c)}/{len(o)}  AUC {auc([-x for x in c], [-x for x in o]):.3f}")

    print("== (3) collapse by order at lr 5e-3 ==")
    new = [r for r in rs if r["grid"] == "svhn_collapse"]
    n = {lay: sum(r["collapsed"] for r in new if r["layout"] == lay) for lay in ("noent", "pairs", "default", "last")}
    for lay in n:
        vals = [f"{r['val']:.3f}" for r in new if r["layout"] == lay]
        print(f"{lay:8s} {n[lay]}/6  val {' '.join(vals)}")
    tot = sum(n.values())
    if tot in (0, 24):
        print("H_collapse-order: INCONCLUSIVE")
    else:
        h = n["noent"] <= n["default"] <= n["last"] and n["pairs"] <= n["default"] and n["last"] - n["noent"] >= 2
        print(f"H_collapse-order: {'PASS' if h else 'FAIL'}; refuted: {n['noent'] >= n['last']}")


if __name__ == "__main__":
    main()
