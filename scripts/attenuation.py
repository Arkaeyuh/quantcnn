#!/usr/bin/env python3
"""Cosine-factor attenuation of the zero-rotation readout gradient on trained bottlenecks.

At zero rotations, d<Z_k>/dz_j carries the factor prod_{i in S_k \\ j} cos a_i, a_i = pi tanh z_i
(the proposition in the appendix). For the trained default and last-ring circuits (CIFAR-10,
4,000 images, validation-selected lr, seeds 0-2), on each run's validation embeddings, this
records E[cos^2 a_j] per coordinate and, for every (readout k, input j in S_k) pair, the measured
E[prod_{i in S_k \\ j} cos^2 a_i] next to the product of the per-coordinate means (the value under
independence across coordinates), grouped by the number of other factors |S_k| - 1.

From repository root::

    PYTHONPATH=. python -m scripts.attenuation --out results/attenuation.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import torch

from quantcnn.data import get_splits
from quantcnn.models.classical_heads import cnot_ring_supports
from scripts.frozen_heads import _embed
from scripts.probe_backbones import _selected_runs
from scripts.zero_angle_check import ROOT, SLUG, _load

# name -> (slug suffix, grid, zero-rotation supports)
CASES = {
    "quantum": ("", "cifar_val", cnot_ring_supports(8, 4)),
    "quantum+entlast": ("_entlast", "cifar_entangle", cnot_ring_supports(8, 4, last_only=True)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/attenuation.csv")
    ap.add_argument("--val_size", type=int, default=2000)
    a = ap.parse_args()
    rows = []
    for name, (suffix, grid, supports) in CASES.items():
        lr = _selected_runs(ROOT, [grid])[(4000, name)][0]
        measured, independent, c2_means = defaultdict(list), defaultdict(list), []
        for seed in range(3):
            model = _load(ROOT / grid / SLUG.format(suffix, lr, seed))
            z, _ = _embed(model, get_splits("cifar10", 4000, seed, a.val_size)["val"], torch.device("cpu"))
            c2 = torch.cos(math.pi * torch.tanh(z.double())) ** 2
            c2_means.append(c2.mean(0))
            for s in supports:
                for j in s:
                    others = [i for i in s if i != j]
                    if others:
                        measured[len(others)].append(c2[:, others].prod(1).mean().item())
                        independent[len(others)].append(math.prod(c2[:, i].mean().item() for i in others))
        c2m = torch.stack(c2_means)
        for m in sorted(measured):
            rows.append(dict(head=name, other_factors=m, n_pairs=len(measured[m]),
                             measured=round(sum(measured[m]) / len(measured[m]), 4),
                             independence=round(sum(independent[m]) / len(independent[m]), 4),
                             cos2_mean=round(c2m.mean().item(), 4), cos2_min=round(c2m.min().item(), 4),
                             cos2_max=round(c2m.max().item(), 4)))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
