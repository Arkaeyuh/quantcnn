#!/usr/bin/env python3
"""Readout sensitivity at initialization against the independence prediction (2026-10-30 predictions).

For each layout and seed, the network is built exactly as ``scripts.train`` builds it (same seed, split and
init). On the validation split, J = mean over samples of sum_k sum_j (d<Z_k>/dz_j)^2 is measured with autograd
through the circuit at its initialized rotations. The prediction treats the readouts as the zero-rotation
products with independent coordinates:
    J_hat = sum_k sum_{j in S_k} pi^2 E[sech^4 z_j sin^2 phi_j] prod_{i in S_k \\ j} c_i,   c_i = E[cos^2 phi_i],
from the empirical marginals of the initial bottleneck. From repository root::

    PYTHONPATH=. .venv/bin/python scripts/init_gradients.py --dataset svhn --seeds 0 1 2 3 4 --out results/init_gradients_svhn.csv
"""

from __future__ import annotations

import argparse
import csv
import math

import torch

from quantcnn.data import dataset_num_classes, get_splits
from quantcnn.models.classical_heads import cnot_ring_supports
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.train import set_seed

# layout -> (entangle argument, supports at zero rotations)
LAYOUTS = {"noent": (False, [[k] for k in range(8)]),
           "pairs": ("pairs", [[k] for k in range(8)]),
           "default": (True, cnot_ring_supports(8, 4)),
           "last": ("last", cnot_ring_supports(8, 4, last_only=True))}


def measure(dataset: str, layout: str, seed: int, val_size: int = 2000, train_mode: bool = False) -> dict:
    set_seed(seed)
    splits = get_splits(dataset, 4000, seed, val_size, data_root="./data")
    ent, supports = LAYOUTS[layout]
    model = HybridClassifier(num_classes=dataset_num_classes(splits["train"]), bottleneck_dim=8, backbone="cifar10",
                             head="quantum", in_channels=3, quantum_sim="torch", entangle=ent)
    # eval mode (the pre-registered measurement) normalizes with the untouched BatchNorm running statistics;
    # train mode uses batch statistics over batches of 32, as the first training steps do
    model.train(train_mode)
    xs = torch.stack([splits["val"][i][0] for i in range(len(splits["val"]))])
    with torch.no_grad():
        z = torch.cat([model.backbone(x) for x in xs.split(32 if train_mode else 256)]).double()
    vqc = model.head.vqc
    zr = z.clone().float().requires_grad_(True)
    r = vqc(zr)  # (n, 8) readouts <Z_k>
    J = sum(torch.autograd.grad(r[:, k].sum(), zr, retain_graph=True)[0].double().pow(2).sum(1) for k in range(8))
    phi = math.pi * torch.tanh(z)
    c = torch.cos(phi).pow(2).mean(0)
    g = (math.pi ** 2 * torch.cosh(z).pow(-4) * torch.sin(phi).pow(2)).mean(0)
    J_hat = sum(g[j] * math.prod(c[i] for i in S if i != j) for S in supports for j in S)
    return dict(dataset=dataset, layout=layout, seed=seed, bn_mode="train" if train_mode else "eval", order=sum(map(len, supports)) / 8, J=J.mean().item(),
                J_hat=float(J_hat), c_mean=c.mean().item(), abs_tanh_mean=torch.tanh(z).abs().mean().item(),
                z_sd=z.std().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="svhn")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_mode", action="store_true", help="BatchNorm batch statistics (post-hoc check)")
    a = ap.parse_args()
    rows = [measure(a.dataset, lay, sd, train_mode=a.train_mode) for lay in LAYOUTS for sd in a.seeds]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
