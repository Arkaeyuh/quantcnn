#!/usr/bin/env python3
"""How fast do two implementations of the same head diverge under training?

On the pretrained ResNet-18 (CIFAR-10, 4,000 images, seed 0, head lr 1e-3, backbone lr 1e-4,
AdamW, clip 2.0) three models start from the same weights and see the same batches:

* ``frozen0``: the default ring VQC with every rotation frozen at zero;
* ``cosprod`` and ``cosprod_b``: two copies of the classical head on its zero-rotation
  products, whose forward map and input gradients equal the frozen circuit's
  (tests/test_identities.py).

At selected steps it records each model's training loss and the L2 distance between backbone
parameters (frozen0 vs cosprod, and cosprod vs cosprod_b). The duplicate pair measures
nondeterminism of the training loop itself; the frozen0 / cosprod pair measures how fast
rounding differences between the two implementations grow.

From repository root::

    PYTHONPATH=. python -m scripts.lockstep_divergence --out results/lockstep_divergence.csv
"""

from __future__ import annotations

import argparse
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier

LOG_STEPS = {0, 1, 2, 5, 10, 20, 50, 100, 124}


def _flat(m: HybridClassifier) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.backbone.parameters()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lockstep_divergence.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    models = {"frozen0": HybridClassifier(10, 8, "resnet18_pt", "quantum", 3, quantum_sim="torch"),
              "cosprod": HybridClassifier(10, 8, "resnet18_pt", "cosprod", 3),
              "cosprod_b": HybridClassifier(10, 8, "resnet18_pt", "cosprod", 3)}
    ref = models["frozen0"]
    with torch.no_grad():
        ref.head.vqc.qlayer.weights.zero_()
        ref.head.vqc.qlayer.weights.requires_grad_(False)
        for name in ("cosprod", "cosprod_b"):
            models[name].backbone.load_state_dict(ref.backbone.state_dict())
            models[name].head.fc.load_state_dict(ref.head.proj.state_dict())
    opts = {}
    for name, m in models.items():
        m.to(device).train()
        head = [p for n, p in m.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        opts[name] = torch.optim.AdamW([{"params": list(m.backbone.parameters()), "lr": a.lr * 0.1},
                                        {"params": head, "lr": a.lr}], weight_decay=1e-4)
    loader = DataLoader(get_splits("cifar10", 4000, a.seed, 2000)["train"], batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    ce = nn.CrossEntropyLoss()
    rows = []
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        loss = {}
        for name, m in models.items():
            opts[name].zero_grad(set_to_none=True)
            out = ce(m(xb), yb)
            out.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 2.0)
            opts[name].step()
            loss[name] = out.item()
        if step in LOG_STEPS:
            pa, pb, pc = (_flat(models[n]) for n in ("frozen0", "cosprod", "cosprod_b"))
            row = dict(step=step, loss_frozen0=round(loss["frozen0"], 6), loss_cosprod=round(loss["cosprod"], 6),
                       loss_cosprod_b=round(loss["cosprod_b"], 6),
                       dist_frozen0_cosprod=f"{(pa - pb).norm().item():.3e}",
                       dist_cosprod_duplicate=f"{(pb - pc).norm().item():.3e}",
                       backbone_norm=round(pb.norm().item(), 3))
            print(row, flush=True)
            rows.append(row)
        if step == max(LOG_STEPS):
            break
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
