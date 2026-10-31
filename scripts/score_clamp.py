#!/usr/bin/env python3
"""Score the saturation-clamp control (predictions/prospective_predictions.md, 2026-10-31).

Paired by seed over 0-9. Probe accuracy and S (``sat_090_z``, %) come from the linear-probe CSVs (``--mlp`` for the
MLP probe), for CIFAR-10/4,000 runs at lr 0.01; clean training accuracy from each run's final.json.
From repository root: ``PYTHONPATH=. .venv/bin/python scripts/score_clamp.py [--mlp]``."""

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from scripts.probe_backbones import _head_key

SEEDS = range(10)
LAYOUT = {"last": "+entlast", "default": ""}
CLAMP = {"last": "+satup43", "default": "+satup31"}


def probes(mlp: bool):
    out = defaultdict(dict)
    for f in sorted(glob.glob("results/*probe.csv")):
        if ("mlp" in f) != mlp or "q12" in f or "all_" in f:
            continue
        for r in csv.DictReader(open(f)):
            if r.get("dataset") != "cifar10" or r.get("train_subset_size") != "4000" or r.get("source_lr") != "0.01":
                continue
            s, v = int(r["seed"]), (float(r["probe_test_acc"]), float(r["sat_090_z"]) / 100)
            prev = out[r["source_head"]].get(s)
            assert prev is None or abs(prev[0] - v[0]) < 1e-9, (f, r["source_head"], s, prev, v)
            out[r["source_head"]][s] = v
    return out


def train_acc():
    out = defaultdict(dict)
    for m in glob.glob("runs/protocol_v2/*/*/meta.json"):
        meta = json.loads(Path(m).read_text())
        if meta.get("dataset") != "cifar10" or str(meta.get("train_subset_size")) != "4000" or meta.get("lr") != 0.01:
            continue
        f = Path(m).parent / "final.json"
        if f.exists():
            out[_head_key(meta)].setdefault(meta["seed"], []).append(json.loads(f.read_text())["train_clean_acc"] * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", action="store_true")
    a = ap.parse_args()
    P, T = probes(a.mlp), train_acc()
    print(f"== saturation-clamp control ({'MLP' if a.mlp else 'linear'} probe), seeds 0-9 ==")

    def cell(h):
        d = P[h]
        missing = [s for s in SEEDS if s not in d]
        assert not missing, (h, missing)
        tr = [st.mean(T[h][s]) for s in SEEDS if s in T[h]]
        return ({s: d[s][0] for s in SEEDS}, st.mean(d[s][1] for s in SEEDS), st.mean(tr) if tr else float("nan"), len(tr))

    def diff(x, y):
        d = [x[s] - y[s] for s in SEEDS]
        return st.mean(d), st.stdev(d), sum(v > 0 for v in d), d

    def verdict(r):
        return "H_mediated" if r <= 1 / 3 else "H_direct" if r >= 2 / 3 else "mixed"

    for lay in ("last", "default"):
        h = {"plain": "quantum" + LAYOUT[lay], "esc": "quantum+pi_tanh_esc" + LAYOUT[lay],
             "escclamp": "quantum+pi_tanh_esc" + LAYOUT[lay] + CLAMP[lay]}
        if lay == "default":
            h["plainclamp"] = "quantum" + CLAMP[lay]
        c = {k: cell(v) for k, v in h.items()}
        print(f"\n-- {lay} ring --")
        for k, (acc, S, tr, n) in c.items():
            print(f"  {k:10s} {h[k]:34s} probe {st.mean(acc.values()):6.2f} ± {st.stdev(acc.values()):4.2f}  "
                  f"S {S:.3f}  train {tr:5.1f} (n={n})")
        g = diff(c["esc"][0], c["plain"][0])
        gc = diff(c["escclamp"][0], c["plain"][0])
        cost = diff(c["escclamp"][0], c["esc"][0])
        for lab, x in (("esc - plain (unclamped gain)", g), ("escclamp - plain", gc), ("escclamp - esc", cost)):
            print(f"  {lab:30s} {x[0]:+6.2f} ± {x[1]:4.2f}  pos {x[2]}/10  {[round(v, 1) for v in x[3]]}")
        man = abs(c["escclamp"][1] - c["plain"][1]) <= 0.10
        guard = c["escclamp"][2] >= c["esc"][2] - 10
        R = gc[0] / g[0]
        print(f"  (1) manipulation |S_escclamp - S_plain| = {abs(c['escclamp'][1] - c['plain'][1]):.3f} <= 0.10: {man}; "
              f"guard train {c['escclamp'][2]:.1f} >= {c['esc'][2] - 10:.1f}: {guard}")
        print(f"  (1) R = {R:+.2f} -> {verdict(R) if man and guard else 'INCONCLUSIVE'}")
        if lay == "default":
            gpc = diff(c["escclamp"][0], c["plainclamp"][0])
            pc = diff(c["plainclamp"][0], c["plain"][0])
            guard2 = c["plainclamp"][2] >= c["plain"][2] - 10
            R2 = gpc[0] / g[0]
            for lab, x in (("escclamp - plainclamp", gpc), ("plainclamp - plain", pc)):
                print(f"  {lab:30s} {x[0]:+6.2f} ± {x[1]:4.2f}  pos {x[2]}/10  {[round(v, 1) for v in x[3]]}")
            print(f"  (2) guard plainclamp train {c['plainclamp'][2]:.1f} >= {c['plain'][2] - 10:.1f}: {guard2}; "
                  f"S plainclamp {c['plainclamp'][1]:.3f} vs escclamp {c['escclamp'][1]:.3f}")
            print(f"  (2) R2 = {R2:+.2f} -> {verdict(R2) if guard2 else 'INCONCLUSIVE'}")


if __name__ == "__main__":
    main()
