#!/usr/bin/env python3
"""Lockstep control on the ImageNet-pretrained ResNet-18 (CIFAR-10, 4,000 images).

Three heads are trained in one process from the same initial weights, on the same batches
(each augmented batch is drawn once and fed to all three), with identical optimizer settings:

* ``quantum``: the default ring VQC, rotations trainable (U(-0.1, 0.1) init);
* ``quantum_frozen0``: the same circuit with every rotation frozen at exactly zero, whose
  forward map and input gradients equal the products head's (tests/test_identities.py);
* ``cosprod``: the classical head on the default circuit's zero-rotation products.

The backbone (incl. its new 512->8 layer) and the readout Linear are copied from the first
model into the other two. Each head writes a run directory in the protocol_v2 format
(meta.json, metrics.csv, best_model.pt, final.json) under ``--out_root``, so
``probe_backbones`` can probe them.

From repository root::

    PYTHONPATH=. python -m scripts.lockstep_pretrained --n_workers 5
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from quantcnn.training_utils import RunConfig, accuracy, evaluate
from scripts.train import FIELDNAMES, _provenance

HEADS = {  # name -> (HybridClassifier head, freeze rotations at zero)
    "quantum": ("quantum", False),
    "quantum_frozen0": ("quantum", True),
    "cosprod": ("cosprod", False),
}


def _cfg(head: str, frozen: bool, lr: float, seed: int, epochs: int) -> RunConfig:
    return RunConfig(dataset="cifar10", backbone="resnet18_pt", head=head, bottleneck_dim=8,
                     train_subset_size="4000", seed=seed, epochs=epochs, batch_size=32, lr=lr,
                     lr_backbone_multiplier=0.1, gradient_clip_norm=2.0, weight_decay=1e-4,
                     reupload_mode="none", n_var_layers=4, mlp_hidden=None, lr_scheduler="cosine",
                     val_size=2000, quantum_sim="torch", freeze_circuit=frozen,
                     vqc_init_scale=0.0 if frozen else 0.1)


def _run(lr: float, seed: int, out_root: str, epochs: int, threads: int) -> str:
    torch.set_num_threads(threads)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    models = {name: HybridClassifier(10, 8, "resnet18_pt", head, 3, quantum_sim="torch")
              for name, (head, _) in HEADS.items()}
    ref = models["quantum"]
    with torch.no_grad():
        for name, m in models.items():
            m.backbone.load_state_dict(ref.backbone.state_dict())
            readout = m.head.proj if hasattr(m.head, "proj") else m.head.fc
            readout.load_state_dict(ref.head.proj.state_dict())
            if HEADS[name][1]:
                m.head.vqc.qlayer.weights.zero_()
                m.head.vqc.qlayer.weights.requires_grad_(False)
    splits = get_splits("cifar10", 4000, seed, 2000)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(splits["train"], batch_size=32, shuffle=True, num_workers=0, generator=gen)
    val_loader = DataLoader(splits["val"], batch_size=32, shuffle=False, num_workers=0)
    state = {}
    for name, m in models.items():
        m.to(device)
        head_params = [p for n, p in m.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        opt = torch.optim.AdamW([{"params": list(m.backbone.parameters()), "lr": lr * 0.1},
                                 {"params": head_params, "lr": lr}], weight_decay=1e-4)
        cfg = _cfg(HEADS[name][0], HEADS[name][1], lr, seed, epochs)
        out = Path(out_root) / f"cifar10_subset4000_{name}_lockstep_lr{lr:g}_s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.csv").unlink(missing_ok=True)
        meta = asdict(cfg) | _provenance(splits["train"]) | {"lockstep": True, "lockstep_name": name}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        state[name] = dict(model=m, opt=opt, sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs),
                           out=out, best=-1.0, cfg=cfg)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        sums = {name: [0.0, 0.0, 0] for name in models}
        for m in models.values():
            m.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            for name, s in state.items():
                s["opt"].zero_grad(set_to_none=True)
                logits = s["model"](xb)
                loss = ce(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(s["model"].parameters(), 2.0)
                s["opt"].step()
                sums[name][0] += loss.item() * len(yb)
                sums[name][1] += accuracy(logits, yb).item() * len(yb)
                sums[name][2] += len(yb)
        for name, s in state.items():
            va_loss, va_acc = evaluate(s["model"], val_loader, device)
            s["sched"].step()
            is_best = va_acc > s["best"]
            if is_best:
                s["best"] = va_acc
                torch.save(s["model"].state_dict(), s["out"] / "best_model.pt")
            row = dict(epoch=epoch, train_loss=f"{sums[name][0] / sums[name][2]:.6f}",
                       train_acc=f"{sums[name][1] / sums[name][2]:.6f}", val_loss=f"{va_loss:.6f}",
                       val_acc=f"{va_acc:.6f}", backbone_gnorm="", head_gnorm="", is_best=int(is_best),
                       dataset="cifar10", head=s["cfg"].head, reupload_mode="none", train_subset_size=4000,
                       seed=seed, bottleneck_dim=8)
            path = s["out"] / "metrics.csv"
            new = not path.exists()
            with path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                if new:
                    w.writeheader()
                w.writerow(row)
    for name, s in state.items():
        m = s["model"]
        m.load_state_dict(torch.load(s["out"] / "best_model.pt", map_location=device))
        rows = list(csv.DictReader((s["out"] / "metrics.csv").open()))
        final = {"best_epoch": int([r for r in rows if r["is_best"] == "1"][-1]["epoch"])}
        for split in ("train_clean", "val", "test"):
            loss, acc = evaluate(m, DataLoader(splits[split], batch_size=256, shuffle=False), device)
            final[f"{split}_loss"], final[f"{split}_acc"] = loss, acc
        (s["out"] / "final.json").write_text(json.dumps(final, indent=2))
    return f"lr {lr:g} seed {seed}: " + ", ".join(
        f"{n} val {s['best']:.4f}" for n, s in state.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="runs/protocol_v2/cifar_pretrained_lockstep")
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_workers", type=int, default=5)
    ap.add_argument("--threads_per_worker", type=int, default=2)
    a = ap.parse_args()
    jobs = [(lr, sd) for lr in a.lrs for sd in a.seeds
            if not all((Path(a.out_root) / f"cifar10_subset4000_{n}_lockstep_lr{lr:g}_s{sd}" / "final.json").exists()
                       for n in HEADS)]
    print(f"{len(jobs)} lockstep jobs -> {a.out_root}", flush=True)
    with ProcessPoolExecutor(max_workers=a.n_workers) as pool:
        futs = {pool.submit(_run, lr, sd, a.out_root, a.epochs, a.threads_per_worker): (lr, sd) for lr, sd in jobs}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {futs[f]}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
