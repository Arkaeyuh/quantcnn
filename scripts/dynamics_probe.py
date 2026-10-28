#!/usr/bin/env python3
"""Training dynamics of the bottleneck (plan A5; predictions 2026-10-27 23:20).

For every ``cifar_dyn`` run (no CNOT, default ring, last ring; lr 1e-2, seeds 0-2) and every
checkpoint saved every 5 epochs, embed the clean train / val / test images with the backbone at that
epoch and record S (|tanh z| > 0.9), M (0.1 <= |tanh z| <= 0.9), Z0 (|tanh z| < 0.1) and PRn on
the test embeddings, plus the linear probe of ``probe_backbones`` (probe lr selected on
validation). Resumable: rows already in --out are skipped. From repository root::

    PYTHONPATH=. .venv/bin/python -m scripts.dynamics_probe --out results/dyn_curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.bottleneck_geometry import _pr
from scripts.frozen_heads import _embed, _train_head
from scripts.probe_backbones import PROBE_LRS, _head_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dyn_curves.csv")
    ap.add_argument("--root", default="runs/protocol_v2/cifar_dyn")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["head"], r["seed"], r["epoch"]) for r in csv.DictReader(fh)}
    for run in sorted(Path(a.root).iterdir()):
        meta = json.loads((run / "meta.json").read_text())
        head, seed = _head_key(meta), int(meta["seed"])
        sp = get_splits("cifar10", 4000, seed, 2000)
        ckpts = sorted(run.glob("ckpt_ep*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        for ck in ckpts:
            epoch = int(re.findall(r"\d+", ck.stem)[0])
            if (head, str(seed), str(epoch)) in done:
                continue
            model = HybridClassifier(10, 8, meta["backbone"], meta["head"], 3, reupload_mode="none",
                                     quantum_sim="torch", encoding=meta.get("encoding", "pi_tanh"),
                                     entangle=meta.get("entangle", True))
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model.to(device)
            emb = {k: _embed(model, sp[s], device) for k, s in
                   (("train", "train_clean"), ("val", "val"), ("test", "test"))}
            z = emb["test"][0].double()
            t = torch.tanh(z).abs()
            best = None
            for plr in PROBE_LRS:
                r = _train_head("linear", emb, seed, plr, 60)
                if best is None or r["val_acc"] > best["val_acc"]:
                    best = r
            row = dict(head=head, seed=seed, epoch=epoch, probe_test_acc=round(best["test_acc"] * 100, 3),
                       probe_val_acc=round(best["val_acc"] * 100, 3),
                       S=round((t > 0.9).double().mean().item(), 4), M=round(((t >= 0.1) & (t <= 0.9)).double().mean().item(), 4),
                       Z0=round((t < 0.1).double().mean().item(), 4), PRn=round(_pr(torch.corrcoef(z.T)) / z.shape[1], 4))
            new = not out.exists()
            with out.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
