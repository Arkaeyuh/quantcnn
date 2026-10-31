#!/usr/bin/env python3
"""Pilot readout for the saturation-clamp factorial (2026-10-31): for each ``cifar_clamp_pilot`` run, the
saturation S (fraction of bottleneck coordinates with |tanh z| > 0.9) on the VALIDATION split at the selected
checkpoint, split into the clamped coordinates (0..k-1) and the rest, and clean training accuracy from
final.json. Deliberately reads no test or probe accuracy. From repository root::

    PYTHONPATH=. .venv/bin/python -m scripts.clamp_pilot_stats --out results/clamp_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed


@torch.no_grad()
def measure(run: Path, device) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, n_var_layers=meta.get("n_var_layers", 4),
                             reupload_mode="none", quantum_sim="torch", encoding=meta.get("encoding", "pi_tanh"),
                             entangle=meta.get("entangle", True))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
    z, _ = _embed(model.to(device), get_splits("cifar10", 4000, meta["seed"], 2000)["val"], device)
    k = int(meta["bottleneck_reg"].removeprefix("satup"))
    sat = (torch.tanh(z).abs() > 0.9).double()
    fin = json.loads((run / "final.json").read_text())
    return dict(layout={True: "default", "last": "last"}[meta.get("entangle", True)], encoding=meta["encoding"],
                k=k, weight=meta["reg_weight"], seed=meta["seed"], S=sat.mean().item(),
                S_clamped=sat[:, :k].mean().item(), S_rest=sat[:, k:].mean().item(),
                train_clean_acc=fin["train_clean_acc"] * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/protocol_v2/cifar_clamp_pilot")
    ap.add_argument("--out", default="results/clamp_pilot.csv")
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = [measure(r, device) for r in sorted(Path(a.root).iterdir()) if (r / "final.json").exists()]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    cells = defaultdict(list)
    for r in rows:
        cells[r["layout"], r["encoding"], r["k"], r["weight"]].append(r)
    print(f"{'layout':8s} {'encoding':12s} {'k':>2s} {'w':>4s} {'n':>2s} {'S':>6s} {'S_clamp':>7s} {'S_rest':>6s} {'train':>6s}")
    for key, rs in sorted(cells.items()):
        m = lambda f: st.mean(r[f] for r in rs)
        print(f"{key[0]:8s} {key[1]:12s} {key[2]:2d} {key[3]:4g} {len(rs):2d} {m('S'):6.3f} {m('S_clamped'):7.3f} "
              f"{m('S_rest'):6.3f} {m('train_clean_acc'):6.1f}")


if __name__ == "__main__":
    main()
