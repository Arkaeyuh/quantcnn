#!/usr/bin/env python3
"""How much does the VQC head depend on the sign of its input?

With the variational rotations near their small initialization, the circuit is
the ``R_y(pi*tanh z)`` encoding followed by CNOT rings, and each <Z> is a product
of cos(pi*tanh z_j) terms: an even function of every z_j. This script measures
how far trained and frozen-circuit heads (CIFAR-10, 4000 images, head lr 1e-2)
are from that symmetry on test embeddings:

* ``flip_all``: median over samples of |q(-z) - q(z)| / |q(z) - q(z')| (z' a random
  other test sample), so 0 means sign-blind and ~1 means flipping the sign moves
  the output as far as switching to another image;
* ``flip_one``: the same with one coordinate flipped, averaged over coordinates;
* head accuracy on z and on -z.

From repository root::

    PYTHONPATH=. python -m scripts.sign_symmetry --out results/sign_symmetry.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed

CASES = {
    "quantum": "runs/protocol_v2/cifar_val/cifar10_subset4000_quantum_lr0.01_s{seed}",
    "quantum+frozen": "runs/protocol_v2/cifar_mech/cifar10_subset4000_quantum_frozen_lr0.01_s{seed}",
}
# Circuit ablations, each at its validation-selected head lr (filled in by --circuit_lrs).
CIRCUIT = {"quantum+half": "half", "quantum+half_sym": "half_sym", "quantum+noent": "noent"}


def _load(run: Path) -> HybridClassifier:
    meta = json.loads((run / "meta.json").read_text())
    model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, reupload_mode="none",
                             quantum_sim="torch", pre_norm=meta.get("pre_norm", "none"),
                             encoding=meta.get("encoding", "pi_tanh"), entangle=meta.get("entangle", True))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
    return model.eval()


@torch.no_grad()
def _rel_parts(vqc, z: torch.Tensor, zf: torch.Tensor, perm: torch.Tensor) -> tuple[float, float, float]:
    """(median ratio, median raw flip change |q(zf) - q(z)|, median raw between-sample distance)."""
    q = vqc(z)
    num, den = (vqc(zf) - q).norm(dim=1), (q - q[perm]).norm(dim=1)
    return (num / den.clamp_min(1e-9)).median().item(), num.median().item(), den.median().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sign_symmetry.csv")
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--circuit_lrs", nargs="*", default=[], metavar="VARIANT=LR",
                    help="Also test cifar_circuit runs, e.g. quantum+half=0.01.")
    a = ap.parse_args()
    cases = dict(CASES)
    for spec in a.circuit_lrs:
        name, lr = spec.split("=")
        cases[name] = f"runs/protocol_v2/cifar_circuit/cifar10_subset4000_quantum_{CIRCUIT[name]}_lr{float(lr):g}_s{{seed}}"
    rows = []
    for name, pattern in cases.items():
        for seed in range(3):
            model = _load(Path(pattern.format(seed=seed)))
            test = get_splits("cifar10", 4000, seed, a.val_size)["test"]
            z, y = _embed(model, test, torch.device("cpu"))
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(z), generator=g)
            flip_one, flip_one_raw = [], []
            for j in range(z.shape[1]):
                zf = z.clone()
                zf[:, j] = -zf[:, j]
                r_j, raw_j, _ = _rel_parts(model.head.vqc, z, zf, perm)
                flip_one.append(r_j)
                flip_one_raw.append(raw_j)
            with torch.no_grad():
                acc = (model.head(z).argmax(1) == y).float().mean().item()
                acc_neg = (model.head(-z).argmax(1) == y).float().mean().item()
            ratio, raw_flip, raw_between = _rel_parts(model.head.vqc, z, -z, perm)
            row = dict(head=name, seed=seed, flip_all=round(ratio, 4), flip_all_raw_change=round(raw_flip, 4),
                       between_sample_raw_dist=round(raw_between, 4),
                       flip_one=round(sum(flip_one) / len(flip_one), 4),
                       flip_one_raw_change=round(sum(flip_one_raw) / len(flip_one_raw), 4),
                       acc_z=round(acc * 100, 2), acc_neg_z=round(acc_neg * 100, 2))
            print(row, flush=True)
            rows.append(row)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
