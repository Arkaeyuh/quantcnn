#!/usr/bin/env python3
"""Where trained bottleneck coordinates sit on the encoding (theory track A1, 2026-10-26).

For the archived CIFAR-10/4,000 headline heads (seeds 0-4, selected lr) this records, on the test
embeddings, the saturated fraction S (|tanh z| > 0.9), the near-zero fraction Z0 (|tanh z| < 0.1),
the middle-zone fraction M (0.1 < |tanh z| < 0.9) and E|cos(pi tanh z)|. The signal-budget bound
predicts that M falls and E|cos| rises with product order; it does not say at which end
(z near 0 or saturated) the coordinates gather. Retrospective analysis, not pre-registered. From
repository root::

    PYTHONPATH=. .venv/bin/python -m scripts.pinning_stats --out results/pinning_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed

ROOT = Path("runs/protocol_v2")
CELLS = {  # cell -> (order, slug stem, lr)
    "linear": (float("nan"), "linear", "0.03"),
    "noent": (1.0, "quantum_noent", "0.01"),
    "default": (2.5, "quantum", "0.01"),
    "entlast": (5.25, "quantum_entlast", "0.01"),
    "cosprod": (2.5, "cosprod", "0.01"),
    "cosprod_last": (5.25, "cosprod_last", "0.01"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pinning_stats.csv")
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for cell, (order, stem, lr) in CELLS.items():
        for seed in range(5):
            name = f"cifar10_subset4000_{stem}_lr{lr}_s{seed}"
            run = next(p for p in sorted(ROOT.glob(f"*/{name}")) if (p / "best_model.pt").exists())
            meta = json.loads((run / "meta.json").read_text())
            model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, quantum_sim="torch",
                                     encoding=meta.get("encoding", "pi_tanh"), entangle=meta.get("entangle", True))
            model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
            z, _ = _embed(model.to(device), get_splits("cifar10", 4000, seed, 2000)["test"], device)
            t = torch.tanh(z.double()).abs()
            rows.append(dict(cell=cell, order=order, seed=seed, run=str(run),
                             S=round((t > 0.9).double().mean().item(), 4),
                             Z0=round((t < 0.1).double().mean().item(), 4),
                             M=round(((t >= 0.1) & (t <= 0.9)).double().mean().item(), 4),
                             Ec=round(torch.cos(math.pi * t).abs().mean().item(), 4)))
            print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
