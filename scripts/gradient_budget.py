#!/usr/bin/env python3
"""Gradient-budget measurements (predictions/prospective_predictions.md, 2026-11-01).

Per run, on the test embeddings z of the selected checkpoint (or a training checkpoint), with t = |tanh z|:
  own     mean over coordinates of s_j = E[(1 - t_j^2)^2 sin^2(pi t_j)]          (own-slope term)
  att     mean over readouts k and j in S_k of prod_{i in S_k, i != j} E cos^2 phi_i  (cross-factor attenuation)
  J0      (1/8) E||d r0 / d z||_F^2, r0_k = prod_{j in S_k} cos phi_j               (zero-rotation readouts)
  J0_ind  the identity's prediction of J0 under independence
  Jc      (1/8) E||d <Z> / d z||_F^2 of the trained circuit's readouts
  Jh      (1/8) E||d logits / d z||_F^2 (circuit and decoder)
All derivatives are the true derivatives of the forward function: escape-only weights are loaded into the
plain pi*tanh head, which computes the same function. From repository root::

    PYTHONPATH=. .venv/bin/python -m scripts.gradient_budget
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed
from scripts.readout_magnitude import _supports


def _model(meta: dict) -> HybridClassifier:
    enc = meta.get("encoding", "pi_tanh")
    enc = "pi_tanh" if enc.startswith("pi_tanh") else enc  # escape variants: same forward function
    return HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, n_var_layers=meta.get("n_var_layers", 4),
                            reupload_mode="none", quantum_sim="torch", encoding=enc,
                            entangle=meta.get("entangle", True))


def _jac(f, z: torch.Tensor, n_out: int) -> float:
    """(1/8) E ||d f(z) / d z||_F^2 over samples (f maps (n, 8) -> (n, n_out), rows independent)."""
    z = z.clone().requires_grad_(True)
    y = f(z)
    tot = torch.zeros((), dtype=torch.float64)
    for k in range(n_out):
        (g,) = torch.autograd.grad(y[:, k].sum(), z, retain_graph=k < n_out - 1)
        tot = tot + (g.double() ** 2).sum()
    return (tot / (z.shape[0] * z.shape[1])).item()


def measure(run: Path, state: Path, device) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    model = _model(meta)
    model.load_state_dict(torch.load(state, map_location="cpu"))
    z, _ = _embed(model.to(device), get_splits("cifar10", 4000, int(meta["seed"]), 2000)["test"], device)
    head = model.head.cpu().double().eval()
    z = z.double()
    sup = _supports(meta)
    t = torch.tanh(z).abs()
    phi = math.pi * torch.tanh(z)
    s = ((1 - t ** 2) ** 2 * torch.sin(math.pi * t) ** 2).mean(0)
    c = (torch.cos(phi) ** 2).mean(0)
    att = [math.prod(c[i].item() for i in S if i != j) for S in sup for j in S]
    j0_ind = sum(math.pi ** 2 * s[j].item() * math.prod(c[i].item() for i in S if i != j) for S in sup for j in S) / 8
    r0 = lambda x: torch.stack([torch.cos(math.pi * torch.tanh(x[:, S])).prod(1) for S in sup], 1)
    return dict(own=s.mean().item(), att=sum(att) / len(att), J0=_jac(r0, z, len(sup)), J0_ind=j0_ind,
                Jc=_jac(head.vqc, z, 8), Jh=_jac(head, z, 10), S=(t > 0.9).double().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gradient_budget.csv")
    ap.add_argument("--dyn_out", default="results/gradient_budget_dyn.csv")
    ap.add_argument("--esc_out", default="results/gradient_budget_esc.csv")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--part", choices=["all", "E", "D", "X"], default="all")
    ap.add_argument("--dyn_shard", type=int, default=0, help="D only: this process takes dynamics runs i::n")
    ap.add_argument("--dyn_n", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def write(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
        print(f"wrote {len(rows)} rows to {path}", flush=True)

    rows = []
    for r in (csv.DictReader(open("results/bottleneck_geometry.csv")) if a.part in ("all", "E") else []):
        if r["group"] == "q8":
            run = Path(r["run"])
            rows.append(dict(cell=r["cell"], order=float(r["order"]), seed=int(r["seed"]), probe=float(r["probe"]),
                             **measure(run, run / "best_model.pt", device)))
            print("E", r["cell"], r["seed"], flush=True)
    if rows:
        write(a.out, rows)

    rows = []
    dyn = sorted(Path("runs/protocol_v2/cifar_dyn").iterdir())[a.dyn_shard::a.dyn_n] if a.part in ("all", "D") else []
    for run in dyn:
        meta = json.loads((run / "meta.json").read_text())
        gn = {int(m["epoch"]): float(m["backbone_gnorm"]) for m in csv.DictReader(open(run / "metrics.csv"))}
        for ck in sorted(run.glob("ckpt_ep*.pt"), key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1))):
            ep = int(re.search(r"ep(\d+)", ck.name).group(1))
            layout = {False: "noent", True: "default", "last": "last"}[meta.get("entangle", True)]
            rows.append(dict(layout=layout, seed=int(meta["seed"]), epoch=ep, backbone_gnorm=gn.get(ep, float("nan")),
                             **measure(run, ck, device)))
            print("D", layout, meta["seed"], ep, flush=True)
    if rows:
        write(a.dyn_out if a.dyn_n == 1 else a.dyn_out.replace(".csv", f"_{a.dyn_shard}.csv"), rows)

    rows = []
    for m in (sorted(glob.glob("runs/protocol_v2/cifar_esc/*/meta.json")) if a.part in ("all", "X") else []):
        run = Path(m).parent
        meta = json.loads(Path(m).read_text())
        if meta["head"] != "quantum" or meta.get("encoding") != "pi_tanh_esc":
            continue
        layout = {False: "noent", True: "default", "last": "last"}[meta.get("entangle", True)]
        rows.append(dict(layout=layout, seed=int(meta["seed"]), **measure(run, run / "best_model.pt", device)))
        print("X", layout, meta["seed"], flush=True)
    if rows:
        write(a.esc_out, rows)


if __name__ == "__main__":
    main()
