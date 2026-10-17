#!/usr/bin/env python3
"""Evaluate saved checkpoints of a sweep on clean train and test images.

For every run directory with a ``best_model.pt`` this reports, for that saved
(best-test-epoch) checkpoint, in ``model.eval()`` mode and with test
preprocessing for both splits:

* clean training-subset accuracy and test accuracy, and their gap;
* bottleneck saturation on the test split: the fraction of embedding values with
  ``|tanh z| > 0.9`` and ``> 0.99`` (the region where the quantum encoding
  ``R_y(pi tanh z)`` is flat). Reported for every head, so quantum-trained
  backbones can be compared with classically trained ones.

Quantum heads are evaluated with the batched torch simulator, which is checked
against PennyLane in ``tests/test_torch_vqc.py``.

From repository root::

    PYTHONPATH=. python -m scripts.analyze_checkpoints runs/logs/sweep_20260628_223253 \\
        --out results/checkpoints_cifar10.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from quantcnn.data import get_cifar10, get_mnist, stratified_indices
from quantcnn.models.hybrid_classifier import HybridClassifier


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _eval(model, ds, device):
    model.eval()
    correct = n = 0
    embs = []
    for xb, yb in DataLoader(ds, batch_size=500, shuffle=False):
        xb, yb = xb.to(device), yb.to(device)
        z = model.backbone(xb)
        logits = model.head(z)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
        embs.append(z.cpu())
    return correct / n, torch.cat(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default="./data")
    a = ap.parse_args()
    device = _device()
    rows = []
    cache = {}
    for run in sorted(Path(a.sweep_dir).iterdir()):
        ckpt = run / "best_model.pt"
        if not ckpt.exists():
            continue
        meta = json.loads((run / "meta.json").read_text())
        ds_name, n, seed = meta["dataset"], int(meta["train_subset_size"]), int(meta["seed"])
        if ds_name not in cache:
            # Test preprocessing for both splits: the train pool is loaded with the test transform.
            loader = get_mnist if ds_name == "mnist" else get_cifar10
            _, test_ds = loader(None, 0, data_root=a.data_root)
            clean_train = type(test_ds)(root=a.data_root, train=True, download=False, transform=test_ds.transform)
            cache[ds_name] = (clean_train, test_ds)
        clean_train, test_ds = cache[ds_name]
        train_sub = Subset(clean_train, stratified_indices(clean_train.targets, n, seed))

        model = HybridClassifier(
            num_classes=10,
            bottleneck_dim=meta["bottleneck_dim"],
            backbone=meta["backbone"],
            head=meta["head"],
            in_channels=1 if ds_name == "mnist" else 3,
            mlp_hidden=meta.get("mlp_hidden"),
            n_var_layers=meta["n_var_layers"],
            reupload_mode=meta["reupload_mode"],
            quantum_sim="torch",
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)
        tr_acc, _ = _eval(model, train_sub, device)
        te_acc, z = _eval(model, test_ds, device)
        t = torch.tanh(z).abs()
        with (run / "metrics.csv").open() as fh:
            logged = [r for r in csv.DictReader(fh)]
        best = [r for r in logged if r["is_best"] == "1"][-1]
        rows.append(
            dict(
                run=run.name,
                dataset=ds_name,
                train_subset_size=n,
                head=meta["head"],
                reupload_mode=meta["reupload_mode"] if meta["head"] == "quantum" else "",
                seed=seed,
                ckpt_epoch=int(best["epoch"]),
                epochs_logged=len(logged),
                logged_test_acc=round(float(best["val_acc"]) * 100, 3),
                clean_train_acc=round(tr_acc * 100, 3),
                test_acc=round(te_acc * 100, 3),
                clean_gap=round((tr_acc - te_acc) * 100, 3),
                logged_aug_train_acc=round(float(best["train_acc"]) * 100, 3),
                sat_090=round((t > 0.9).float().mean().item() * 100, 2),
                sat_099=round((t > 0.99).float().mean().item() * 100, 2),
                emb_abs_mean=round(z.abs().mean().item(), 3),
            )
        )
        print(rows[-1], flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
