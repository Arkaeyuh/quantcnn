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


def protocol(*roots: Path):
    cells = defaultdict(lambda: defaultdict(list))
    for f in sorted(x for root in roots for x in root.glob("*/final.json")):
        meta = json.loads((f.parent / "meta.json").read_text())
        fin = json.loads(f.read_text())
        head = meta["head"] + (str(meta["mlp_hidden"]) if meta["head"] == "mlp" else "")
        head += "+bn" if meta.get("pre_norm", "none") == "bn" else ""
        cells[(int(meta["train_subset_size"]), head)][meta["lr"]].append(
            (meta["seed"], fin["val_acc"] * 100, fin["test_acc"] * 100, fin["train_clean_acc"] * 100, fin["best_epoch"]))
    print(f"\n== {' + '.join(map(str, roots))}  (lr chosen by mean val acc; test reported)")
    sel, shared = {}, {}
    for (n, head), by_lr in sorted(cells.items()):
        for lr, v in by_lr.items():
            assert sorted(x[0] for x in v) == [0, 1, 2], (n, head, lr)
        lr = max(by_lr, key=lambda k: st.mean(x[1] for x in by_lr[k]))
        v = by_lr[lr]
        sel[(n, head)] = {x[0]: x[2] for x in v}
        shared[(n, head)] = {x[0]: x[2] for x in by_lr.get(1e-3, [])}
        alt = "  ".join(f"{k:g}:{st.mean(x[2] for x in by_lr[k]):.1f}" for k in sorted(by_lr))
        print(f"{n:5d} {head:10s} lr={lr:<7g} test {_ms([x[2] for x in v])}  val {st.mean(x[1] for x in v):5.1f}"
              f"  clean-train {st.mean(x[3] for x in v):5.1f}  epoch {st.mean(x[4] for x in v):4.1f}   [test by lr: {alt}]")
    print("-- paired test differences, quantum minus other head (mean +- SD over seeds)")
    for n in sorted({k[0] for k in sel}):
        for other in [h for h in ["linear", "mlp8", "mlp7"] if (n, h) in sel]:
            print(f"{n:5d} quantum-{other:6s} selected lr {_paired(sel[(n, 'quantum')], sel[(n, other)])}"
                  f"   shared lr 1e-3 {_paired(shared[(n, 'quantum')], shared[(n, other)])}")
    if any(h.endswith("+bn") for _, h in sel):
        print("-- paired test differences with pre-encoding BatchNorm (selected lr; mean +- SD over seeds)")
        for n in sorted({k[0] for k in sel}):
            for a, b in [("quantum+bn", "quantum"), ("linear+bn", "linear"), ("quantum+bn", "linear+bn"),
                         ("quantum+bn", "linear"), ("quantum+bn", "mlp8")]:
                if (n, a) in sel and (n, b) in sel:
                    print(f"{n:5d} {a:10s} - {b:10s} {_paired(sel[(n, a)], sel[(n, b)])}")


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


def probe(path: Path):
    """Linear probe on the frozen backbones of the validation-selected protocol runs."""
    cells = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        cells[(r["dataset"], int(r["train_subset_size"]), r["source_head"])][int(r["seed"])] = r
    print(f"\n== {path}  (same linear probe on each validation-selected backbone; test reported)")
    for (ds, n, head), by_seed in sorted(cells.items()):
        assert sorted(by_seed) == [0, 1, 2], (ds, n, head)
        f = lambda k: [float(by_seed[i][k]) for i in range(3)]
        print(f"{ds:8s} {n:5d} {head:10s} probe {_ms(f('probe_test_acc'))}  own head {_ms(f('source_test_acc'))}"
              f"  sat>0.9 z {st.mean(f('sat_090_z')):5.1f}%  head input {st.mean(f('sat_090_head_input')):5.1f}%")
    print("-- paired probe differences between backbones (mean +- SD over seeds)")
    probe_acc = {k: {i: float(v[i]["probe_test_acc"]) for i in v} for k, v in cells.items()}
    for ds, n in sorted({k[:2] for k in cells}):
        for a, b in [("quantum", "linear"), ("quantum", "mlp8"), ("quantum+bn", "quantum"),
                     ("quantum+bn", "linear"), ("linear+bn", "linear")]:
            if (ds, n, a) in probe_acc and (ds, n, b) in probe_acc:
                print(f"{ds:8s} {n:5d} {a:10s} - {b:10s} {_paired(probe_acc[(ds, n, a)], probe_acc[(ds, n, b)])}")


def mech(runs: Path, base_runs: Path, probe_path: Path, base_probe_path: Path):
    """CIFAR-10/4000 mechanism audit: backbone lr at fixed head lr, and the frozen circuit."""
    def key(m):
        bb = m["lr"] * m.get("lr_backbone_multiplier", 0.5)
        return ("quantum+frozen" if m.get("freeze_circuit") else m["head"]), round(bb, 6), m["lr"]
    runs_by = defaultdict(dict)
    for root in (base_runs, runs):
        for fin in sorted(root.glob("cifar10_subset4000_*/final.json")):
            m = json.loads((fin.parent / "meta.json").read_text())
            head, bb, lr = key(m)
            if (head, lr) in {("linear", 0.03), ("quantum", 0.01)} or head == "quantum+frozen":
                f = json.loads(fin.read_text())
                runs_by[(head, bb, lr)][int(m["seed"])] = (f["val_acc"] * 100, f["test_acc"] * 100)
    probes = {}
    for path, base in ((base_probe_path, True), (probe_path, False)):
        for r in csv.DictReader(path.open()):
            if r["dataset"] != "cifar10" or r["train_subset_size"] != "4000":
                continue
            h = r["source_head"].split("@")[0]
            if base and h not in ("linear", "quantum"):
                continue
            lr = float(r["source_lr"])
            bb = float(r["source_head"].split("@bblr")[1]) if "@bblr" in r["source_head"] else lr * 0.5
            probes.setdefault((h, round(bb, 6), lr), {})[int(r["seed"])] = (float(r["probe_test_acc"]),
                                                                         float(r["sat_090_z"]))
    print(f"\n== {runs}  (CIFAR-10, 4000; head lr fixed per head; probe = same linear probe as above)")
    for (head, bb, lr), by in sorted(runs_by.items()):
        if sorted(by) != [0, 1, 2]:
            continue
        pr = probes.get((head, bb, lr))
        tail = (f"  probe {_ms([pr[i][0] for i in range(3)])}  sat>0.9 z {st.mean(pr[i][1] for i in range(3)):5.1f}%"
                if pr else "")
        print(f"{head:15s} head lr {lr:<6g} backbone lr {bb:<7g} val {st.mean(by[i][0] for i in range(3)):5.1f}"
              f"  test {_ms([by[i][1] for i in range(3)])}{tail}")
    print("-- paired probe differences, quantum minus linear backbone at matched backbone lr")
    for bb in sorted({k[1] for k in probes if k[0] == "quantum"}):
        q, l = probes.get(("quantum", bb, 0.01)), probes.get(("linear", bb, 0.03))
        if q and l:
            print(f"backbone lr {bb:<7g} {_paired({i: q[i][0] for i in q}, {i: l[i][0] for i in l})}")
    best = {}
    for head, lr in (("linear", 0.03), ("quantum", 0.01)):
        cand = {k: v for k, v in runs_by.items() if k[0] == head and k[2] == lr and sorted(v) == [0, 1, 2]}
        best[head] = max(cand, key=lambda k: st.mean(cand[k][i][0] for i in range(3)))
        print(f"validation-selected backbone lr for {head}: {best[head][1]:g}")
    q, l = probes[best["quantum"]], probes[best["linear"]]
    print(f"probe, quantum minus linear, each at its selected backbone lr {_paired({i: q[i][0] for i in q}, {i: l[i][0] for i in l})}")
    fz = {k: v for k, v in runs_by.items() if k[0] == "quantum+frozen" and sorted(v) == [0, 1, 2]}
    sel = max(fz, key=lambda k: st.mean(fz[k][i][0] for i in range(3)))
    qf, qt = probes[sel], probes[("quantum", 0.005, 0.01)]
    print(f"frozen circuit, validation-selected head lr {sel[2]:g}; probe frozen minus trained circuit "
          f"{_paired({i: qf[i][0] for i in qf}, {i: qt[i][0] for i in qt})}")


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
    root = Path("results/protocol_v2")
    protocol(root / "mnist_search", root / "mnist_bn")
    protocol(root / "cifar_val", root / "cifar_bn")
    frozen(Path("results/frozen_heads.csv"))
    probe(Path("results/backbone_probe.csv"))
    mech(root / "cifar_mech", root / "cifar_val", Path("results/mech_probe.csv"), Path("results/backbone_probe.csv"))
    for name in ["checkpoints_mnist", "checkpoints_cifar10"]:
        checkpoints(Path("results") / f"{name}.csv")


if __name__ == "__main__":
    main()
