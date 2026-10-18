#!/usr/bin/env python3
"""How well does the zero-rotation product-of-cosines model describe trained circuits?

With every variational rotation set to zero, the VQC readout is exactly
<Z_k> = prod_{j in S_k} cos(pi*tanh z_j) (see ``classical_heads.cnot_ring_supports``).
For trained CIFAR-10/4000 heads (default ring, last-ring-only, ring-free, disjoint pairs; validation-selected lr,
seeds 0-4, plus the frozen-at-init circuit as a sanity anchor), this compares the
trained circuit q(z) with the same circuit with its rotations zeroed, q0(z), on the
validation embeddings of the run's own backbone:

* ``out_rel_err``: ||q - q0|| / ||q - mean(q)||, over all samples and readouts;
* ``out_r2``: R^2 of q regressed linearly (with bias) on q0, i.e. how much of the
  trained readout a linear layer could recover from the zero-angle features;
* ``jac_cos_q`` / ``jac_cos_logits``: median over samples of the cosine between the
  flattened input Jacobians dq/dz and dq0/dz, and between d(logits)/dz under the trained
  readout with q versus with q0 (the gradient direction the backbone receives);
* ``acc`` / ``acc_zeroed``: validation accuracy of the head, and of the same head with
  the rotations zeroed (trained readout Linear kept);
* ``mean_abs_ry`` / ``mean_abs_rz``: trained rotation magnitudes (rad).

From repository root::

    PYTHONPATH=. python -m scripts.zero_angle_check --out results/zero_angle_check.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed
from scripts.probe_backbones import _selected_runs

ROOT = Path("runs/protocol_v2")
SLUG = "cifar10_subset4000_quantum{}_lr{:g}_s{}"
# name -> (slug suffix, grid for seeds 0-2, seeds)
CASES = {
    "quantum": ("", "cifar_val", range(5)),
    "quantum+entlast": ("_entlast", "cifar_entangle", range(5)),
    "quantum+noent": ("_noent", "cifar_circuit", range(5)),
    "quantum+frozen": ("_frozen", "cifar_mech", range(3)),
    "quantum+pairs": ("_pairs", "cifar_pairs", range(5)),
    "quantum+pairs_last": ("_pairs_last", "cifar_pairs", range(5)),
}


def _load(run: Path) -> HybridClassifier:
    meta = json.loads((run / "meta.json").read_text())
    model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, n_var_layers=meta.get("n_var_layers", 4),
                             reupload_mode="none", quantum_sim="torch", pre_norm=meta.get("pre_norm", "none"),
                             encoding=meta.get("encoding", "pi_tanh"), entangle=meta.get("entangle", True))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
    return model.eval()


def _jacobian(f, z: torch.Tensor) -> torch.Tensor:
    """Per-sample input Jacobians (B, out, in); samples are independent, so sum-backward works."""
    z = z.clone().requires_grad_(True)
    out = f(z)
    rows = [torch.autograd.grad(out[:, k].sum(), z, retain_graph=True)[0] for k in range(out.shape[1])]
    return torch.stack(rows, 1)


def _median_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(1), b.flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).median().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/zero_angle_check.csv")
    ap.add_argument("--val_size", type=int, default=2000)
    a = ap.parse_args()
    rows = []
    for name, (suffix, grid, seeds) in CASES.items():
        lr = _selected_runs(ROOT, [grid])[(4000, name)][0]  # validation-selected on seeds 0-2
        for seed in seeds:
            run = ROOT / (grid if seed < 3 else "cifar_seeds") / SLUG.format(suffix, lr, seed)
            model = _load(run)
            z, y = _embed(model, get_splits("cifar10", 4000, seed, a.val_size)["val"], torch.device("cpu"))
            vqc, proj = model.head.vqc, model.head.proj
            vqc0 = copy.deepcopy(vqc)
            with torch.no_grad():
                vqc0.qlayer.weights.zero_()
                q, q0 = vqc(z).double(), vqc0(z).double()
                rel = ((q - q0).norm() / (q - q.mean(0)).norm()).item()
                X = torch.cat([q0, torch.ones(len(q0), 1, dtype=q0.dtype)], 1)
                pred = X @ torch.linalg.lstsq(X, q).solution
                r2 = (1 - ((q - pred) ** 2).sum() / ((q - q.mean(0)) ** 2).sum()).item()
                acc = (proj(q.float()).argmax(1) == y).float().mean().item()
                acc0 = (proj(q0.float()).argmax(1) == y).float().mean().item()
            jq, jq0 = _jacobian(vqc, z), _jacobian(vqc0, z)
            w = proj.weight.detach()
            jh, jh0 = w @ jq, w @ jq0
            wts = vqc.qlayer.weights.detach()
            row = dict(head=name, lr=lr, seed=seed, out_rel_err=round(rel, 4), out_r2=round(r2, 4),
                       jac_cos_q=round(_median_cos(jq, jq0), 4), jac_cos_logits=round(_median_cos(jh, jh0), 4),
                       acc=round(acc * 100, 2), acc_zeroed=round(acc0 * 100, 2),
                       mean_abs_ry=round(wts[..., 0].abs().mean().item(), 4),
                       mean_abs_rz=round(wts[..., 1].abs().mean().item(), 4))
            print(row, flush=True)
            rows.append(row)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
