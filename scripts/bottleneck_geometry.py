#!/usr/bin/env python3
"""Bottleneck geometry of trained backbones, as candidate mediators of the representation cost.

For each cell (CIFAR-10, 4,000 images, small CNN, selected head lr; all seeds with a probe row),
on the test embeddings z of each run this records
  S    fraction of coordinates with |tanh z| > 0.9 (saturation),
  PRn  participation ratio (sum l)^2 / sum l^2 of the correlation matrix of z, divided by q,
  CPRn participation ratio of the between-class covariance of standardized z, / min(q, 9),
with q the bottleneck width, next to the run's linear-probe accuracy. Cells and criteria are in
``predictions/prospective_predictions.md`` (mechanism diagnostics).

``--probe_csv`` instead measures every head in one probe file (e.g. the regularized heads of the
mechanism test), at every lr and seed it contains.

From repository root::

    PYTHONPATH=. python -m scripts.bottleneck_geometry --out results/bottleneck_geometry.csv
    PYTHONPATH=. python -m scripts.bottleneck_geometry --probe_csv results/mech2_probe.csv \
        --out results/mech2_geometry.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed
from scripts.probe_backbones import _head_key

ROOT = Path("runs/protocol_v2")
Q12_PROBE = "results/q12_probe.csv"
# cell -> (group, probe head key, order); 12-qubit cells use key suffix "@b12"
CELLS = {
    **{k: ("q8", k, o) for k, o in [("quantum+noent", 1), ("quantum+pairs", 1), ("quantum+pairs_last", 1.5),
                                    ("quantum", 2.5), ("quantum+L2", 5.25), ("quantum+L1", 5.25),
                                    ("quantum+entlast", 5.25), ("quantum+star", 1), ("quantum+chain", 1.5),
                                    ("quantum+star_in_last", 1.875), ("quantum+ladder2_last", 3),
                                    ("quantum+brick", 3.5), ("quantum+chain_last", 4.5)]},
    **{f"quantum+half{s}": ("half", f"quantum+half{s}", o) for s, o in
       [("+noent", 1), ("+pairs_last", 1.5), ("", 2.5), ("+ladder2_last", 3), ("+chain_last", 4.5), ("+entlast", 5.25)]},
    **{f"quantum+phase{s}": ("phase", f"quantum+phase{s}", o) for s, o in
       [("+noent", 1), ("+star", 1), ("+chain", 1.5), ("", 2.5), ("+chain_last", 4.5), ("+entlast", 5.25)]},
    **{f"{k}@b12": ("q12", k, o) for k, o in [("quantum+noent", 1), ("quantum", 47 / 12), ("quantum+entlast", 88 / 12)]},
    **{k: ("classical", k, o) for k, o in [("cos", 1), ("cosprod_pairs_last", 1.5), ("cosprod", 2.5), ("cosprod_last", 5.25),
                                           ("cosprod_win2", 2), ("cosprod_star2", 2), ("cosprod_win3", 3),
                                           ("cosprod_star3", 3), ("cosprod_win4", 4), ("cosprod_gap4", 4)]},
}


def _run_index() -> dict:
    idx = {}
    for m in glob.glob(str(ROOT / "*" / "*" / "meta.json")):
        meta = json.loads(Path(m).read_text())
        if meta["dataset"] != "cifar10" or int(meta["train_subset_size"]) != 4000:
            continue
        if meta.get("backbone") not in (None, "cifar10") or not Path(m).with_name("best_model.pt").exists():
            continue
        q = int(meta.get("bottleneck_dim", 8))
        key = (_head_key(meta) + (f"@b{q}" if q != 8 else ""), f"{meta['lr']:g}", int(meta["seed"]))
        idx[key] = Path(m).parent
    return idx


def _probe_rows() -> dict:
    """cell -> {seed: (lr, probe test acc)} at the lr with the most seeds, seeds 0-4 only (the
    pre-registered cohort; later replications on other seeds are analysed separately)."""
    by = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob("results/*probe.csv"):
        if "mlp" in f or f == Q12_PROBE:
            continue
        for r in csv.DictReader(open(f)):
            if r.get("dataset") == "cifar10" and r.get("train_subset_size") == "4000" and int(r["seed"]) <= 4:
                by[r["source_head"]][r["source_lr"]][int(r["seed"])] = float(r["probe_test_acc"])
    for r in csv.DictReader(open(Q12_PROBE)):
        by[r["source_head"] + "@b12"][r["source_lr"]][int(r["seed"])] = float(r["probe_test_acc"])
    out = {}
    for cell in CELLS:
        lrs = by[cell]
        lr = max(lrs, key=lambda k: len(lrs[k]))
        out[cell] = {s: (lr, v) for s, v in lrs[lr].items()}
    return out


def _pr(m: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(m).clamp_min(0)
    return (ev.sum() ** 2 / (ev**2).sum()).item()


def _stats(z: torch.Tensor, y: torch.Tensor) -> dict:
    q = z.shape[1]
    zs = (z - z.mean(0)) / (z.std(0) + 1e-8)
    means = torch.stack([zs[y == c].mean(0) for c in range(10)])
    return dict(S=(torch.tanh(z).abs() > 0.9).double().mean().item(),
                PRn=_pr(torch.corrcoef(z.T)) / q,
                CPRn=_pr(torch.cov(means.T)) / min(q, 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bottleneck_geometry.csv")
    ap.add_argument("--probe_csv", default=None, help="measure every head/lr/seed in this probe file")
    a = ap.parse_args()
    idx = _run_index()
    if a.probe_csv:
        cells, probes = {}, defaultdict(dict)
        for r in csv.DictReader(open(a.probe_csv)):
            key = f"{r['source_head']}@lr{r['source_lr']}"
            cells[key] = ("probe_csv", r["source_head"], float("nan"))
            probes[key][int(r["seed"])] = (r["source_lr"], float(r["probe_test_acc"]))
    else:
        cells, probes = CELLS, _probe_rows()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for cell, (group, head, order) in cells.items():
        for seed, (lr, probe) in sorted(probes[cell].items()):
            run = idx[(head if a.probe_csv else cell, lr, seed)]
            meta = json.loads((run / "meta.json").read_text())
            model = HybridClassifier(10, int(meta.get("bottleneck_dim", 8)), meta["backbone"], meta["head"], 3,
                                     mlp_hidden=meta.get("mlp_hidden"), n_var_layers=meta.get("n_var_layers", 4),
                                     reupload_mode="none", quantum_sim="torch", pre_norm=meta.get("pre_norm", "none"),
                                     encoding=meta.get("encoding", "pi_tanh"), entangle=meta.get("entangle", True))
            model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
            z, y = _embed(model.to(device), get_splits("cifar10", 4000, seed, 2000)["test"], device)
            rows.append(dict(cell=cell, group=group, order=round(order, 4), seed=seed, lr=lr, run=str(run),
                             probe=probe, **{k: round(v, 5) for k, v in _stats(z.double(), y).items()}))
            print(rows[-1]["cell"], seed, {k: rows[-1][k] for k in ("probe", "S", "PRn", "CPRn")}, flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(Counter(r["group"] for r in rows))


if __name__ == "__main__":
    main()
