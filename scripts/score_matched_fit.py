#!/usr/bin/env python3
"""Score the matched-fit comparison (predictions/prospective_predictions.md, 2026-11-01 02:00).

For each seed, the higher-order head at epoch 80 is compared with the lower-order head at the checkpoint (with a
probe value in results/dyn_curves.csv or dyn2_curves.csv) whose logged training accuracy (metrics.csv) is closest.
From repository root: ``PYTHONPATH=. .venv/bin/python scripts/score_matched_fit.py``."""

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict

HEADS = {"noent": "quantum+noent", "default": "quantum", "last": "quantum+entlast"}
DIRS = {"noent": "quantum_noent", "default": "quantum", "last": "quantum_entlast"}
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--root", default="cifar_dyn")
ap.add_argument("--curves", nargs="+", default=["results/dyn_curves.csv", "results/dyn2_curves.csv"])
ap.add_argument("--min_negative", type=int, default=2, help="seeds with a negative gap required by the hypotheses")
A = ap.parse_args()
SEEDS = tuple(A.seeds)

probe = defaultdict(dict)
for f in A.curves:
    for r in csv.DictReader(open(f)):
        key, ep, v = (r["head"], int(r["seed"])), int(r["epoch"]), float(r["probe_test_acc"])
        if int(r["seed"]) not in SEEDS:
            continue
        if ep in probe[key]:
            assert abs(probe[key][ep] - v) < 1e-6, (f, key, ep)  # dyn2 repeats dyn's runs and seeds
        probe[key][ep] = v

fit, loss = defaultdict(dict), defaultdict(dict)
for lay, d in DIRS.items():
    for sd in SEEDS:
        path = glob.glob(f"runs/protocol_v2/{A.root}/cifar10_subset4000_{d}_lr0.01_s{sd}/metrics.csv")
        assert len(path) == 1, (lay, sd, path)
        for m in csv.DictReader(open(path[0])):
            fit[lay, sd][int(m["epoch"])] = float(m["train_acc"]) * 100
            loss[lay, sd][int(m["epoch"])] = float(m["train_loss"])

jc = {}
for f in glob.glob("results/gradient_budget_dyn_*.csv"):
    for r in csv.DictReader(open(f)):
        jc[r["layout"], int(r["seed"]), int(r["epoch"])] = float(r["Jc"])


def compare(hi, lo):
    print(f"\n-- {hi} (epoch 80) vs {lo} at matched logged training accuracy --")
    gaps = []
    for sd in SEEDS:
        f_star = fit[hi, sd][80]
        eps = [e for e in probe[HEADS[lo], sd] if e in fit[lo, sd]]
        e = min(eps, key=lambda x: abs(fit[lo, sd][x] - f_star))
        df = fit[lo, sd][e] - f_star
        usable = abs(df) <= 3
        g = probe[HEADS[hi], sd][80] - probe[HEADS[lo], sd][e]
        g_ep = probe[HEADS[hi], sd][80] - probe[HEADS[lo], sd][80]
        print(f"  seed {sd}: {hi} fit {f_star:5.1f} loss {loss[hi, sd][80]:.3f} probe {probe[HEADS[hi], sd][80]:5.2f} | "
              f"{lo} epoch {e:2d} fit {fit[lo, sd][e]:5.1f} loss {loss[lo, sd][e]:.3f} probe {probe[HEADS[lo], sd][e]:5.2f} | "
              f"|dfit| {abs(df):.1f}{'' if usable else ' (UNUSABLE)'} gap {g:+6.2f} (matched-epoch gap {g_ep:+6.2f}) "
              f"Jc {jc.get((hi, sd, 80), float('nan')):.3f} vs {jc.get((lo, sd, e), float('nan')):.3f}")
        if usable:
            gaps.append(g)
    m = st.mean(gaps) if gaps else float("nan")
    print(f"  matched-fit gap: mean {m:+.2f} over {len(gaps)} usable seeds, negative in {sum(x < 0 for x in gaps)}")
    return m, sum(x < 0 for x in gaps), len(gaps)


a = compare("last", "noent")
b = compare("default", "noent")
compare("last", "default")
k = A.min_negative
print(f"\n  >> H_fit-last (mean <= -3, >= {k} negative): {'PASS' if a[2] >= k and a[0] <= -3 and a[1] >= k else 'FAIL' if a[2] >= k else 'INCONCLUSIVE'}")
print(f"  >> H_fit-default (mean <= -1.5, >= {k} negative): {'PASS' if b[2] >= k and b[0] <= -1.5 and b[1] >= k else 'FAIL' if b[2] >= k else 'INCONCLUSIVE'}")
print(f"  >> refutation (last-vs-noent mean >= 0): {'MET' if a[0] >= 0 else 'not met'}")
