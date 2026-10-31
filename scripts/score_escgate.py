#!/usr/bin/env python3
"""Score the saturation-gated escape-only control (predictions/prospective_predictions.md, 2026-10-30) on
``results/escgate_probe.csv``. Gains are paired by seed (0-4) against the plain baseline of the same layout at
its selected lr, as in the escape-only test; full escape-only (``results/esc_probe.csv``) is recomputed as a check.
From repository root: ``PYTHONPATH=. .venv/bin/python scripts/score_escgate.py``."""

import csv
import json
import statistics as st

from scripts.bottleneck_geometry import _probe_rows, _run_index

LAYOUTS = {"noent": "+noent", "default": "", "last": "+entlast"}
probes, idx = _probe_rows(), _run_index()


def load(path):
    return {(r["source_head"], int(r["seed"])): r for r in csv.DictReader(open(path))}


esc, gate = load("results/esc_probe.csv"), load("results/escgate_probe.csv")


def train_acc(head, lr, seed):
    run = idx.get((head, lr, seed))
    return json.loads((run / "final.json").read_text())["train_clean_acc"] * 100 if run else float("nan")


res = {}
print(f"{'layout':8s} {'variant':5s} {'probe':>6s} {'gain':>11s} {'+seeds':>6s} {'S%':>5s} {'train':>6s}")
for lay, suf in LAYOUTS.items():
    base = probes["quantum" + suf]
    print(f"{lay:8s} {'plain':5s} {st.mean(v for _, v in base.values()):6.1f} {'':>11s} {'':>6s} {'':>5s} "
          f"{st.mean(train_acc('quantum' + suf, lr, s) for s, (lr, _) in base.items()):6.1f}")
    for var, src in [("full", esc), ("sat", gate), ("mid", gate)]:
        head = f"quantum+pi_tanh_esc{'' if var == 'full' else var}{suf}"
        rows = {s: src[(head, s)] for s in range(5) if (head, s) in src}
        g = {s: float(r["probe_test_acc"]) - base[s][1] for s, r in rows.items() if s in base}
        res[lay, var] = g
        print(f"{lay:8s} {var:5s} {st.mean(float(r['probe_test_acc']) for r in rows.values()):6.1f} "
              f"{st.mean(g.values()):+5.2f}±{st.stdev(g.values()):4.2f} {sum(v > 0 for v in g.values()):>4d}/{len(g)} "
              f"{st.mean(float(r['sat_090_z']) for r in rows.values()):5.2f} "
              f"{st.mean(train_acc(head, '0.01', s) for s in rows):6.1f}")

m = {k: st.mean(v.values()) for k, v in res.items()}
sat, mid = res["last", "sat"], res["last", "mid"]
h_sat = m["last", "sat"] >= 4.5 and sum(v > 0 for v in sat.values()) >= 4
h_mid = m["last", "mid"] <= 2.3 and sum(sat[s] > mid[s] for s in sat) >= 4
print(f"H_sat-route: gain_sat(last) {m['last', 'sat']:+.2f}, positive {sum(v > 0 for v in sat.values())}/5 -> "
      f"{'PASS' if h_sat else 'FAIL'}")
print(f"H_mid-small: gain_mid(last) {m['last', 'mid']:+.2f}, sat>mid {sum(sat[s] > mid[s] for s in sat)}/5 -> "
      f"{'PASS' if h_mid else 'FAIL'}")
print(f"secondary default gain_sat {m['default', 'sat']:+.2f} (>= 0.75: {m['default', 'sat'] >= 0.75}); "
      f"no CNOT sat {m['noent', 'sat']:+.2f} mid {m['noent', 'mid']:+.2f} (both <= 1.5: "
      f"{max(m['noent', 'sat'], m['noent', 'mid']) <= 1.5})")
print("refutation (saturation not the route):", m["last", "mid"] >= m["last", "sat"] or m["last", "sat"] <= 2.3)
for var in ("full", "sat", "mid"):
    print(f"per-seed last-ring gain {var}:", " ".join(f"{res['last', var][s]:+.1f}" for s in sorted(res["last", var])))
