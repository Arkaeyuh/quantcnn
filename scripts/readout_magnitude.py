#!/usr/bin/env python3
"""Readout magnitude and decoder scale of trained VQC heads (decoder-compensation test, 2026-10-30 predictions).

For the 8-qubit pi*tanh runs of ``results/bottleneck_geometry.csv`` (group q8), on the test embeddings z of
each run this records
  A      mean over readouts k of E|<Z_k>|, the trained circuit's actual outputs,
  W      mean over k of the norm of decoder column k (``head.proj.weight[:, k]``),
  L      mean over k of ||w_k|| * sd(<Z_k>), the logit scale each readout contributes,
and, descriptively, A_hat = mean over k of prod_{j in S_k} E|cos phi_j| (zero rotations, independent
factors) and the fraction of readouts of order >= 4 whose signal-budget bound (delta = 0.1, a = their
measured E|<Z_k>|) is below 1.

Added after scoring (post hoc): the proposition's own quantity, the zero-rotation product
r0_k = prod_{j in S_k} cos phi_j per sample. A0 is the mean over k of E|r0_k|; for the readouts of order >= 4,
A0_high_min and A_hat_high_min are the smallest E|r0_k| and prod_j E|cos phi_j|, and delta_crit0_max and
delta_crit_hat_max the largest delta = arccos(a^(1/m)) / pi above which the bound with that a is below 1.
From repository root::

    PYTHONPATH=. .venv/bin/python -m scripts.readout_magnitude --out results/readout_magnitude.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.classical_heads import cnot_ring_supports
from quantcnn.models.hybrid_classifier import HybridClassifier
from quantcnn.models.torch_vqc import entangle_layout
from scripts.frozen_heads import _embed

DELTA = 0.1


def _supports(meta: dict) -> list[list[int]]:
    layout = entangle_layout(meta.get("entangle", True))
    if layout is None:
        return [[k] for k in range(8)]
    pattern, last = layout
    return cnot_ring_supports(8, meta.get("n_var_layers", 4), last_only=last, pattern=pattern)


@torch.no_grad()
def measure(run: Path, seed: int, device) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, n_var_layers=meta.get("n_var_layers", 4),
                             reupload_mode="none", quantum_sim="torch", encoding=meta.get("encoding", "pi_tanh"),
                             entangle=meta.get("entangle", True))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
    z, _ = _embed(model.to(device), get_splits("cifar10", 4000, seed, 2000)["test"], device)
    head = model.head.cpu()
    q = head.vqc(z.float()).double()  # (n, 8) readouts <Z_k>
    w = head.proj.weight.double().norm(dim=0)  # (8,) decoder column norms
    amp = q.abs().mean(0)
    t = torch.tanh(z.double()).abs()
    cabs = torch.cos(math.pi * torch.tanh(z.double())).abs().mean(0)
    sup = _supports(meta)
    a_hat = torch.tensor([math.prod(cabs[j].item() for j in S) for S in sup])
    middle = ((t > DELTA) & (t < 1 - DELTA)).double().mean(0)
    high = [k for k, S in enumerate(sup) if len(S) >= 4]
    bound = [(1 - amp[k].item() ** (1 / len(sup[k]))) / (1 - math.cos(math.pi * DELTA)) for k in high]
    cos = torch.cos(math.pi * torch.tanh(z.double()))
    a0 = torch.stack([cos[:, S].prod(1).abs().mean() for S in sup])
    crit = lambda a, m: math.acos(a ** (1 / m)) / math.pi
    nan = float("nan")
    extra = dict(A0=a0.mean().item(),
                 A0_high_min=min(a0[k].item() for k in high) if high else nan,
                 A_hat_high_min=min(a_hat[k].item() for k in high) if high else nan,
                 delta_crit0_max=max(crit(a0[k].item(), len(sup[k])) for k in high) if high else nan,
                 delta_crit_hat_max=max(crit(a_hat[k].item(), len(sup[k])) for k in high) if high else nan)
    return dict(A=amp.mean().item(), W=w.mean().item(), L=(w * q.std(0)).mean().item(), A_hat=a_hat.mean().item(),
                A_min=amp.min().item(), middle=middle.mean().item(), n_high=len(high),
                bound_nonvacuous=(sum(b < 1 for b in bound) / len(bound)) if high else float("nan"), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/readout_magnitude.csv")
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for r in csv.DictReader(open("results/bottleneck_geometry.csv")):
        if r["group"] != "q8":
            continue
        m = measure(Path(r["run"]), int(r["seed"]), device)
        rows.append(dict(cell=r["cell"], order=float(r["order"]), seed=int(r["seed"]), lr=r["lr"],
                         probe=float(r["probe"]), **{k: round(v, 5) for k, v in m.items()}))
        print(rows[-1]["cell"], rows[-1]["seed"], flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
