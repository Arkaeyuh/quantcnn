#!/usr/bin/env python3
"""How strongly a gradient floor changes the bottleneck gradient of a trained product head.

For the trained coupled product heads on the default and last-ring supports (CIFAR-10, 4,000
images, lr 1e-2, seeds 0-4), on each run's validation embeddings, this computes the per-sample
loss gradient dL/dz through the true product (floor 0) and through the floored backward of
``_FlooredProduct`` (same weights, same forward), and records the median cosine between them,
the median norm ratio, and the fraction of (readout, input) paths whose other-factor product is
below the floor in magnitude. ``stage=init`` repeats this for a freshly initialized model with the
same seed (backbone and readout at initialization). Measured before the gradient-floor runs, to
size the intervention. Embeddings are taken in evaluation mode, which at initialization uses
BatchNorm's uncalibrated running statistics; ``--init_train_mode`` (added after the runs) instead
takes the initialization embeddings in training mode with batches of 32, as training computes
them, and records only the init stage.

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
from torch.utils.data import DataLoader
from scripts.frozen_heads import _embed
from scripts.zero_angle_check import ROOT, _load

# head -> (run-name stem, grid for seeds 0-2); seeds 3-4 are in cifar_seeds
CASES = {"cosprod": "cifar_cosprod", "cosprod_last": "cifar_cosprod"}
SLUG = "cifar10_subset4000_{}_lr0.01_s{}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grad_floor_stats.csv")
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--init_train_mode", action="store_true",
                    help="init stage only, embeddings in training mode (BatchNorm batch statistics, batch 32)")
    a = ap.parse_args()
    rows = []
    for head, grid in CASES.items():
        stages = ("init_train",) if a.init_train_mode else ("trained", "init")
        for seed, stage in [(s, st) for s in range(5) for st in stages]:
            if stage == "trained":
                model = _load(ROOT / (grid if seed < 3 else "cifar_seeds") / SLUG.format(head, seed))
            else:
                torch.manual_seed(seed)
                model = HybridClassifier(10, 8, "cifar10", head, 3).eval()
            val = get_splits("cifar10", 4000, seed, a.val_size)["val"]
            if stage == "init_train":
                model.train()
                with torch.no_grad():
                    batches = [(model.backbone(xb), yb) for xb, yb in DataLoader(val, batch_size=32, shuffle=False)]
                z, y = torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])
            else:
                z, y = _embed(model, val, torch.device("cpu"))
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
