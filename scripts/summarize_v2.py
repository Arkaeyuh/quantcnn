#!/usr/bin/env python3
"""Summarize the validation-protocol re-runs, frozen-embedding heads and checkpoints.

All selection uses validation accuracy only: for each (cell, head) the learning
rate with the best mean validation accuracy over seeds is chosen, and the test
accuracy of that learning rate is reported (mean +- sample SD over seeds).
Head comparisons are also reported as paired differences over seeds (same seed,
hence same training subset, validation split and, for frozen heads, the same
embedding).

From repository root::

    PYTHONPATH=. python -m scripts.summarize_v2
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _paired(a: dict, b: dict) -> str:
    d = [a[k] - b[k] for k in sorted(a)]
    return f"{st.mean(d):+5.1f} +- {st.stdev(d):3.1f}  per seed {['%+.2f' % x for x in d]}"


def _ms(xs):
    return f"{st.mean(xs):5.1f} +- {st.stdev(xs):4.1f}" if len(xs) > 1 else f"{xs[0]:5.1f}"


def protocol(root: Path):
    cells = defaultdict(lambda: defaultdict(list))
    for f in sorted(root.glob("*/final.json")):
        meta = json.loads((f.parent / "meta.json").read_text())
        fin = json.loads(f.read_text())
        head = meta["head"] + (str(meta["mlp_hidden"]) if meta["head"] == "mlp" else "")
        cells[(int(meta["train_subset_size"]), head)][meta["lr"]].append(
            (meta["seed"], fin["val_acc"] * 100, fin["test_acc"] * 100, fin["train_clean_acc"] * 100, fin["best_epoch"]))
    print(f"\n== {root}  (lr chosen by mean val acc; test reported)")
    sel, shared = {}, {}
    for (n, head), by_lr in sorted(cells.items()):
        for lr, v in by_lr.items():
            assert sorted(x[0] for x in v) == [0, 1, 2], (n, head, lr)
        lr = max(by_lr, key=lambda k: st.mean(x[1] for x in by_lr[k]))
        v = by_lr[lr]
        sel[(n, head)] = {x[0]: x[2] for x in v}
        shared[(n, head)] = {x[0]: x[2] for x in by_lr[1e-3]}
        alt = "  ".join(f"{k:g}:{st.mean(x[2] for x in by_lr[k]):.1f}" for k in sorted(by_lr))
        print(f"{n:5d} {head:8s} lr={lr:<7g} test {_ms([x[2] for x in v])}  val {st.mean(x[1] for x in v):5.1f}"
              f"  clean-train {st.mean(x[3] for x in v):5.1f}  epoch {st.mean(x[4] for x in v):4.1f}   [test by lr: {alt}]")
    print("-- paired test differences, quantum minus other head (mean +- SD over seeds)")
    for n in sorted({k[0] for k in sel}):
        for other in [h for h in ["linear", "mlp8", "mlp7"] if (n, h) in sel]:
            print(f"{n:5d} quantum-{other:6s} selected lr {_paired(sel[(n, 'quantum')], sel[(n, other)])}"
                  f"   shared lr 1e-3 {_paired(shared[(n, 'quantum')], shared[(n, other)])}")


def frozen(path: Path):
    cells = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(path.open()):
        cells[(r["dataset"], r["source_head"], r["head"])][r["lr"]].append(r)
    print(f"\n== {path}  (lr chosen by mean val acc; test reported)")
    sel = {}
    for (ds, src, head), by_lr in sorted(cells.items()):
        lr = max(by_lr, key=lambda k: st.mean(float(x["val_acc"]) for x in by_lr[k]))
        v = by_lr[lr]
        assert sorted(int(x["seed"]) for x in v) == [0, 1, 2]
        sel[(ds, src, head)] = {int(x["seed"]): float(x["test_acc"]) for x in v}
        extra = ""
        if head in ("quantum_std", "quantum_aff"):
            extra = f"  sat>0.9 after affine {st.mean(float(x['sat_090_after']) for x in v):.1f}%"
        print(f"{ds:8s} backbone<-{src:8s} {head:13s} lr={lr:<5s} test {_ms([float(x['test_acc']) for x in v])}"
              f"  train {st.mean(float(x['train_acc']) for x in v):5.1f}  src-sat {st.mean(float(x['sat_090_source']) for x in v):5.1f}{extra}")
    print("-- paired test differences on identical embeddings, other head minus quantum (mean +- SD over seeds)")
    for ds, src in sorted({k[:2] for k in sel}):
        for other in ["linear", "mlp", "fourier", "quantum_std", "quantum_aff", "quantum_long"]:
            if (ds, src, other) in sel:
                print(f"{ds:8s} backbone<-{src:8s} {other:13s} {_paired(sel[(ds, src, other)], sel[(ds, src, 'quantum')])}")


def checkpoints(path: Path):
    cells = defaultdict(list)
    for r in csv.DictReader(path.open()):
        cells[(int(r["train_subset_size"]), r["head"] + (":" + r["reupload_mode"] if r["reupload_mode"] else ""))].append(r)
    print(f"\n== {path}  (saved checkpoint, eval mode, clean images)")
    for (n, head), v in sorted(cells.items()):
        f = lambda k: [float(x[k]) for x in v]
        print(f"{n:5d} {head:15s} test {_ms(f('test_acc'))}  clean-train {st.mean(f('clean_train_acc')):5.1f}"
              f"  clean gap {st.mean(f('clean_gap')):5.1f}  sat>0.9 per seed {[x['sat_090'] for x in v]}"
              f"  sat>0.99 {[x['sat_099'] for x in v]}")


def main():
    for g in ["mnist_search", "cifar_val"]:
        protocol(Path("results/protocol_v2") / g)
    frozen(Path("results/frozen_heads.csv"))
    for name in ["checkpoints_mnist", "checkpoints_cifar10"]:
        checkpoints(Path("results") / f"{name}.csv")


if __name__ == "__main__":
    main()
