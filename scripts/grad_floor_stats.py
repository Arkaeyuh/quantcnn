#!/usr/bin/env python3
"""How strongly a gradient floor changes the bottleneck gradient of a trained product head.

For the trained coupled product heads on the default and last-ring supports (CIFAR-10, 4,000
images, lr 1e-2, seeds 0-4), on each run's validation embeddings, this computes the per-sample
loss gradient dL/dz through the true product (floor 0) and through the floored backward of
``_FlooredProduct`` (same weights, same forward), and records the median cosine between them,
the median norm ratio, and the fraction of (readout, input) paths whose other-factor product is
below the floor in magnitude. ``stage=init`` repeats this for a freshly initialized model with the
same seed (backbone and readout at initialization). Measured before the gradient-floor runs, to
size the intervention.

From repository root::

    PYTHONPATH=. python -m scripts.grad_floor_stats --out results/grad_floor_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import math

import torch
import torch.nn.functional as F

from quantcnn.data import get_splits
from quantcnn.models.classical_heads import _FlooredProduct
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed
from scripts.zero_angle_check import ROOT, _load

# head -> (run-name stem, grid for seeds 0-2); seeds 3-4 are in cifar_seeds
CASES = {"cosprod": "cifar_cosprod", "cosprod_last": "cifar_cosprod"}
SLUG = "cifar10_subset4000_{}_lr0.01_s{}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grad_floor_stats.csv")
    ap.add_argument("--val_size", type=int, default=2000)
    a = ap.parse_args()
    rows = []
    for head, grid in CASES.items():
        for seed, stage in [(s, st) for s in range(5) for st in ("trained", "init")]:
            if stage == "trained":
                model = _load(ROOT / (grid if seed < 3 else "cifar_seeds") / SLUG.format(head, seed))
            else:
                torch.manual_seed(seed)
                model = HybridClassifier(10, 8, "cifar10", head, 3).eval()
            z, y = _embed(model, get_splits("cifar10", 4000, seed, a.val_size)["val"], torch.device("cpu"))
            z = z.double()
            h = model.head.double()
            for floor in (0.0, 0.3, 1.0):
                zz = z.clone().requires_grad_(True)
                c = torch.cos(math.pi * torch.tanh(zz))
                loss = F.cross_entropy(h.fc(_FlooredProduct.apply(c, h.mask, floor)), y, reduction="sum")
                (g,) = torch.autograd.grad(loss, zz)
                if floor == 0.0:
                    g_true = g
                    n = c.shape[1]
                    keep = h.mask.bool()[:, None, :] & ~torch.eye(n, dtype=torch.bool)[None]
                    others = torch.where(keep[None], c.detach()[:, None, None, :], torch.ones_like(c)[:, None, None, :]).prod(-1)
                    paths = others[:, h.mask.bool()]  # (B, number of (k, j in S_k) paths)
                    continue
                cos = F.cosine_similarity(g, g_true, dim=1)
                rows.append(dict(head=head, stage=stage, seed=seed, floor=floor,
                                 cos_median=cos.median().item(), cos_mean=cos.mean().item(),
                                 norm_ratio_median=(g.norm(dim=1) / g_true.norm(dim=1)).median().item(),
                                 frac_paths_below=(paths.abs() < floor).double().mean().item()))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
