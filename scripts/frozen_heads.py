#!/usr/bin/env python3
"""Train every head on identical frozen embeddings.

Separates the head from the backbone it induces. For each source checkpoint
(a backbone trained jointly with a linear or a quantum head in the reported
sweeps), embeddings are extracted once in ``eval()`` mode on clean images for
the training subset, a validation split drawn from the remaining training pool,
and the test split. Each head below is then trained on those fixed embeddings,
with the epoch and learning rate selected on validation; the test split is read
once per head at the selected point.

Heads:
  linear       Linear(8, 10)
  mlp          8 -> 8 -> 10, ReLU
  fourier      classical: [cos(pi tanh z), sin(pi tanh z)] -> Linear(16, 10)
  quantum      the VQC head (no re-upload) with its fixed R_y(pi tanh z) encoding
  quantum_std  preprocessing control: z standardized with training-subset
               statistics (fixed), then the same VQC
  quantum_aff  encoding intervention: the same standardization followed by a
               learnable per-dimension affine, before the same VQC
  quantum_long optimization intervention: ``quantum`` with 3x the epochs

From repository root::

    PYTHONPATH=. python -m scripts.frozen_heads --out results/frozen_heads.csv

Source checkpoints (``best_model.pt`` + ``meta.json`` of the 12 source runs) are
released under ``results/frozen_sources``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from quantcnn.data import get_splits
from quantcnn.models.classical_heads import LinearHead, MLPHead
from quantcnn.models.hybrid_classifier import HybridClassifier
from quantcnn.models.quantum_head import QuantumClassifierHead

SOURCES = {
    "mnist": ("sweep_20260627_095118", 1000),
    "cifar10": ("sweep_20260628_223253", 4000),
}
LRS = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
# Per-head grid maximum. The grids were extended wherever validation accuracy was
# still rising at the top; 1.0 diverged for the MLP, and quantum_long is 3x the cost.
MAX_LR = {"mlp": 3e-1, "quantum_long": 3e-1, "linear": 1.0, "fourier": 1.0, "quantum_aff": 1.0,
          "quantum": 10.0, "quantum_std": 10.0}
EPOCHS = 60


class FourierHead(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.fc = nn.Linear(2 * d, k)

    def forward(self, z):
        phi = math.pi * torch.tanh(z)
        return self.fc(torch.cat([torch.cos(phi), torch.sin(phi)], 1))


class AffineQuantumHead(nn.Module):
    def __init__(self, mu: torch.Tensor, sd: torch.Tensor, learn_affine: bool = True):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.a = nn.Parameter(torch.ones_like(mu), requires_grad=learn_affine)
        self.b = nn.Parameter(torch.zeros_like(mu), requires_grad=learn_affine)
        self.q = QuantumClassifierHead(8, 10, 4, "none", sim="torch")

    def pre(self, z):
        return self.a * (z - self.mu) / self.sd + self.b

    def forward(self, z):
        return self.q(self.pre(z))


class MLPProbe(nn.Module):
    """Nonlinear probe: fixed standardization, then 8 -> 256 -> 256 -> 10 with ReLU."""

    def __init__(self, mu, sd, width: int = 256) -> None:
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 10))

    def forward(self, z):
        return self.net((z - self.mu) / self.sd)


def _make_head(kind: str, mu, sd) -> nn.Module:
    if kind == "linear":
        return LinearHead(8, 10)
    if kind == "mlp":
        return MLPHead(8, 8, 10)
    if kind == "mlp_probe":
        return MLPProbe(mu, sd)
    if kind == "fourier":
        return FourierHead(8, 10)
    if kind in ("quantum", "quantum_long"):
        return QuantumClassifierHead(8, 10, 4, "none", sim="torch")
    if kind == "quantum_std":
        return AffineQuantumHead(mu, sd, learn_affine=False)
    if kind == "quantum_aff":
        return AffineQuantumHead(mu, sd)
    raise ValueError(kind)


@torch.no_grad()
def _embed(model, ds, device):
    model.eval()
    zs, ys = [], []
    for xb, yb in DataLoader(ds, batch_size=500, shuffle=False):
        zs.append(model.backbone(xb.to(device)).cpu())
        ys.append(yb)
    return torch.cat(zs), torch.cat(ys)


@torch.no_grad()
def _acc(head, z, y):
    head.eval()
    return (head(z).argmax(1) == y).float().mean().item()


def _train_head(kind, emb, seed, lr, epochs):
    (ztr, ytr), (zva, yva), (zte, yte) = emb["train"], emb["val"], emb["test"]
    torch.manual_seed(seed)
    head = _make_head(kind, ztr.mean(0), ztr.std(0) + 1e-6)
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, None, None)
    for ep in range(1, epochs + 1):
        head.train()
        for idx in torch.randperm(len(ytr), generator=g).split(128):
            opt.zero_grad()
            ce(head(ztr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        va = _acc(head, zva, yva)
        if va > best[0]:
            best = (va, ep, {k: v.clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[2])
    out = dict(val_acc=best[0], best_epoch=best[1], train_acc=_acc(head, ztr, ytr), test_acc=_acc(head, zte, yte))
    if kind in ("quantum_std", "quantum_aff"):
        with torch.no_grad():
            out["sat_090_after"] = (torch.tanh(head.pre(zte)).abs() > 0.9).float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/frozen_heads.csv")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"])
    ap.add_argument("--source_heads", nargs="+", default=["linear", "quantum"])
    ap.add_argument("--heads", nargs="+",
                    default=["linear", "mlp", "fourier", "quantum", "quantum_std", "quantum_aff", "quantum_long"])
    ap.add_argument("--source_root", default="results/frozen_sources")
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["dataset"], r["source_head"], r["seed"], r["head"], r["lr"]) for r in csv.DictReader(fh)}
    for ds_name in a.datasets:
        sweep, n = SOURCES[ds_name]
        for src in a.source_heads:
            for seed in [0, 1, 2]:
                run = Path(a.source_root) / sweep / f"{ds_name}_subset{n}_{src}_rupnone_b8_s{seed}"
                meta = json.loads((run / "meta.json").read_text())
                model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 1 if ds_name == "mnist" else 3,
                                         reupload_mode="none", quantum_sim="torch")
                model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
                model.to(device)
                sp = get_splits(ds_name, n, seed, a.val_size)
                emb = {"train": _embed(model, sp["train_clean"], device),
                       "val": _embed(model, sp["val"], device),
                       "test": _embed(model, sp["test"], device)}
                sat = (torch.tanh(emb["test"][0]).abs() > 0.9).float().mean().item()
                for kind in a.heads:
                    if kind == "quantum_long" and ds_name != "cifar10":
                        continue
                    for lr in [x for x in LRS if x <= MAX_LR.get(kind, LRS[-1])]:
                        key = (ds_name, src, str(seed), kind, f"{lr:g}")
                        if key in done:
                            continue
                        epochs = EPOCHS * 3 if kind == "quantum_long" else EPOCHS
                        r = _train_head(kind, emb, seed, lr, epochs)
                        row = dict(dataset=ds_name, train_subset_size=n, source_head=src, seed=seed, head=kind,
                                   lr=f"{lr:g}", epochs=epochs, sat_090_source=round(sat * 100, 2),
                                   **{k: (round(v * 100, 3) if k.endswith(("acc", "after")) else v) for k, v in r.items()})
                        row.setdefault("sat_090_after", "")
                        new = not out.exists()
                        with out.open("a", newline="") as fh:
                            w = csv.DictWriter(fh, fieldnames=list(row))
                            if new:
                                w.writeheader()
                            w.writerow(row)
                        print(row, flush=True)


if __name__ == "__main__":
    main()
