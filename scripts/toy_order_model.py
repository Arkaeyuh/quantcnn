#!/usr/bin/env python3
"""Toy model of the order -> saturation -> probe-cost account (theory track A1, 2026-10-26).

A small MLP backbone (32 -> 64 -> 8) on a 10-class Gaussian mixture is trained end to end through
a classical product head, prod_{j in S_k} cos(pi tanh z_j) followed by Linear, for the supports of
the circuits in the paper (orders 1 to 5.25) and two designed supports. On the frozen bottleneck
it records the linear-probe accuracy, the saturated fraction S (|tanh z| > 0.9), the fraction
in the middle zone M (delta < |tanh z| < 1 - delta) and the mean factor magnitude E|cos|, which
enter the signal-budget bound

    P(middle zone) <= (1 - E|c|) / (1 - cos(pi delta)),   E|c| >= a^(1/m)

for a readout of order m whose mean magnitude is at least a (independent factors). Runs in a
few minutes on one CPU thread. From repository root::

    PYTHONPATH=. .venv/bin/python -m scripts.toy_order_model --out results/toy_order_model.csv
"""

from __future__ import annotations

import argparse
import csv
import math

import torch
import torch.nn as nn

from quantcnn.models.classical_heads import DESIGNED_SUPPORTS, CosProductHead, cnot_ring_supports

DELTA = 0.1
HEADS = {
    "order1": [[k] for k in range(8)],
    "pairs_last": cnot_ring_supports(8, 4, True, "pairs"),
    "default": cnot_ring_supports(8, 4),
    "star3": DESIGNED_SUPPORTS["star3"],
    "win4": DESIGNED_SUPPORTS["win4"],
    "last_ring": cnot_ring_supports(8, 4, True),
}


def _data(seed: int, n: int = 4000, d: int = 32, k: int = 10, sep: float = 0.7):
    g = torch.Generator().manual_seed(1000 + seed)
    means = torch.randn(k, d, generator=g) * sep / math.sqrt(d) * 4
    def draw(m):
        y = torch.randint(0, k, (m,), generator=g)
        return means[y] + torch.randn(m, d, generator=g), y
    return draw(n), draw(n), draw(n)


def _train(model, params, x, y, epochs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
        sched.step()


def _acc(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item() * 100


def run(head: str, seed: int, epochs: int, init_scale: float = 1.0) -> dict:
    (xtr, ytr), (xva, yva), (xte, yte) = _data(seed)
    torch.manual_seed(seed)
    backbone = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    with torch.no_grad():  # larger initial bottleneck scale moves the pinning toward saturation
        backbone[2].weight.mul_(init_scale)
        backbone[2].bias.mul_(init_scale)
    prod = CosProductHead(8, 10, supports=HEADS[head])
    model = nn.Sequential(backbone, prod)
    _train(model, list(model.parameters()), xtr, ytr, epochs, 3e-3, seed)
    with torch.no_grad():
        ztr, zva, zte = backbone(xtr), backbone(xva), backbone(xte)
        t = torch.tanh(zte).abs()
        c = torch.cos(math.pi * torch.tanh(zte))
        feats = torch.where(prod.mask.bool(), c[:, None, :], torch.ones_like(c)[:, None, :]).prod(-1)
    best = None
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-6
    for plr in (1e-2, 3e-2, 1e-1):  # full-batch probe, lr selected on validation as in the paper
        torch.manual_seed(seed)
        probe = nn.Linear(8, 10)
        f = lambda z: probe((z - mu) / sd)  # noqa: E731
        opt = torch.optim.AdamW(probe.parameters(), lr=plr, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            nn.functional.cross_entropy(f(ztr), ytr).backward()
            opt.step()
        with torch.no_grad():
            va = (f(zva).argmax(1) == yva).float().mean().item()
            te = (f(zte).argmax(1) == yte).float().mean().item() * 100
        if best is None or va > best[0]:
            best = (va, te)
    order = sum(len(s) for s in HEADS[head]) / 8
    return dict(head=head, order=order, seed=seed, init_scale=init_scale, train_acc=round(_acc(model, xtr, ytr), 2),
                test_acc=round(_acc(model, xte, yte), 2), probe=round(best[1], 2),
                S=round((t > 0.9).float().mean().item(), 4),
                M=round(((t > DELTA) & (t < 1 - DELTA)).float().mean().item(), 4),
                Ec=round(c.abs().mean().item(), 4), Er=round(feats.abs().mean().item(), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy_order_model.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--init_scale", type=float, default=1.0, help="multiplier on the last backbone layer's init")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for seed in range(a.seeds):
        for head in HEADS:
            rows.append(run(head, seed, a.epochs, a.init_scale))
            print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
