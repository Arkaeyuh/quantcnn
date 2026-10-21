#!/usr/bin/env python3
"""Effective rank of the class signal the backbone receives through each head.

For CIFAR-10/4000 heads of product order 1 (ring-free), 2.5 (default ring) and 5.25
(last-layer ring), plus the linear head, this computes per-sample cross-entropy gradients
g_i = d loss_i / dz on validation embeddings and reports the effective rank
exp(H(p)) of their second-moment matrix sum_i g_i g_i^T / N, with p its normalized
eigenvalues (8 = gradient energy spread evenly over the bottleneck, 1 = one direction).

* ``final``: the trained run (validation-selected lr, its own backbone);
* ``init``: a freshly initialized model of the same kind built with the run's seed
  (same distribution as the training initialization, not the identical draw).

Also reported: the share of gradient energy in the largest eigenvalue and the median
per-sample gradient norm.

From repository root::

    PYTHONPATH=. python -m scripts.grad_rank --out results/grad_rank.csv
    PYTHONPATH=. python -m scripts.grad_rank --cases family --out results/grad_rank_family.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed
from scripts.probe_backbones import _selected_runs
from scripts.zero_angle_check import _load

ROOT = Path("runs/protocol_v2")
SLUG = "cifar10_subset4000_{}_lr{:g}_s{}"
# name -> (slug head, grid for seeds 0-2, product order, HybridClassifier kwargs for init)
CASES = {
    "linear": ("linear", "cifar_val", None, dict(head="linear")),
    "quantum+noent": ("quantum_noent", "cifar_circuit", 1.0, dict(head="quantum", entangle=False)),
    "quantum": ("quantum", "cifar_val", 2.5, dict(head="quantum", entangle=True)),
    "quantum+entlast": ("quantum_entlast", "cifar_entangle", 5.25, dict(head="quantum", entangle="last")),
}


# Second family of CNOT layouts (--cases family): pattern -> product order
FAMILY = {"star": 1.0, "chain": 1.5, "star_in_last": 1.875, "ladder2_last": 3.0, "brick": 3.5, "chain_last": 4.5}
CASES_FAMILY = {f"quantum+{k}": (f"quantum_{k}", "cifar_family", o, dict(head="quantum", entangle=k))
                for k, o in FAMILY.items()}
EXTRA_SEED_GRID = {"cases": "cifar_seeds", "family": "cifar_family_seeds"}


def _grads(model: HybridClassifier, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = z.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(model.head(model.pre_norm(z)), y, reduction="sum").backward()
    return z.grad.double()


def _stats(g: torch.Tensor) -> dict:
    ev = torch.linalg.eigvalsh(g.T @ g / len(g)).clamp_min(0)
    p = ev / ev.sum()
    h = -(p[p > 0] * p[p > 0].log()).sum()
    return dict(erank=round(h.exp().item(), 4), top_share=round(p.max().item(), 4),
                grad_norm_median=round(g.norm(dim=1).median().item(), 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grad_rank.csv")
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--cases", choices=["cases", "family"], default="cases",
                    help="'family': the second family of CNOT layouts.")
    a = ap.parse_args()
    rows = []
    for name, (slug_head, grid, order, init_kw) in (CASES if a.cases == "cases" else CASES_FAMILY).items():
        lr = _selected_runs(ROOT, [grid])[(4000, name)][0]
        for seed in range(5):
            run = ROOT / (grid if seed < 3 else EXTRA_SEED_GRID[a.cases]) / SLUG.format(slug_head, lr, seed)
            if not (run / "best_model.pt").exists():
                continue
            val = get_splits("cifar10", 4000, seed, a.val_size)["val"]
            trained = _load(run)
            torch.manual_seed(seed)
            fresh = HybridClassifier(10, 8, "cifar10", in_channels=3, quantum_sim="torch", **init_kw).eval()
            for stage, model in (("init", fresh), ("final", trained)):
                z, y = _embed(model, val, torch.device("cpu"))
                row = dict(head=name, order=order if order is not None else "", lr=lr, seed=seed, stage=stage,
                           **_stats(_grads(model, z, y)))
                print(row, flush=True)
                rows.append(row)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
