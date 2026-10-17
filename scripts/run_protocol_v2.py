#!/usr/bin/env python3
"""Re-run the head comparison under a train/validation/test protocol.

Two grids, both using a ``--val_size`` split drawn from the training pool
(disjoint from the training subset) for checkpoint and learning-rate selection,
with the official test split evaluated once on the selected checkpoint:

* ``mnist_search``: linear, MLP (hidden 8, 162 params), MLP (hidden 7, 143
  params) and quantum (no re-upload) heads, each with the same search budget over
  head learning rate {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, 50 epochs.
* ``cifar_val``: linear, MLP and quantum (no re-upload) heads on the recipe of the
  reported CIFAR-10 sweep (80 epochs, clip 2.0), head learning rate {1e-3, 3e-3, 1e-2}.

From repository root::

    PYTHONPATH=. python -m scripts.run_protocol_v2 --grid mnist_search --n_workers 4

Runs (with checkpoints) go to ``runs/protocol_v2/<grid>``; when the grid is done,
each finished run's ``final.json``, ``metrics.csv`` and ``meta.json`` (no
checkpoint) are copied to ``results/protocol_v2/<grid>``, which is what
``scripts.summarize_v2`` reads.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import torch

from scripts.train import default_train_args, run_training

SEEDS = [0, 1, 2]
EDGE_EXTENSION = {
    "mnist_search": [("mnist", n, h, hid, 1e-1) for n in [250, 500, 1000] for h, hid in [("linear", None), ("mlp", 8)]]
                    + [("mnist", 1000, "mlp", 7, 1e-1)],
    "cifar_val": [("cifar10", 2000, "linear", None, 3e-2), ("cifar10", 2000, "quantum", None, 3e-2)]
                 + [("cifar10", 4000, h, hid, 3e-2) for h, hid in [("linear", None), ("mlp", 8), ("quantum", None)]],
}


def _grid(name: str) -> list[dict]:
    items = []
    if name == "mnist_search":
        heads = [("linear", None), ("mlp", 8), ("mlp", 7), ("quantum", None)]
        for n in [250, 500, 1000]:
            for head, hid in heads:
                for lr in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
                    for sd in SEEDS:
                        items.append(dict(dataset="mnist", train_subset_size=n, head=head, mlp_hidden=hid,
                                          lr=lr, seed=sd, epochs=50, gradient_clip_norm=1.0))
    elif name == "cifar_val":
        for n in [1000, 2000, 4000]:
            for head, hid in [("linear", None), ("mlp", 8), ("quantum", None)]:
                for lr in [1e-3, 3e-3, 1e-2]:
                    for sd in SEEDS:
                        items.append(dict(dataset="cifar10", train_subset_size=n, head=head, mlp_hidden=hid,
                                          lr=lr, seed=sd, epochs=80, gradient_clip_norm=2.0))
    else:
        raise ValueError(name)
    # One more step for the cells whose validation-selected lr was the grid maximum.
    for ds, n, head, hid, lr in EDGE_EXTENSION.get(name, []):
        for sd in SEEDS:
            items.append(dict(dataset=ds, train_subset_size=n, head=head, mlp_hidden=hid, lr=lr, seed=sd,
                              epochs=items[0]["epochs"],
                              gradient_clip_norm=items[0]["gradient_clip_norm"]))
    return items


def _slug(it: dict) -> str:
    head = it["head"] + (f"{it['mlp_hidden']}" if it["head"] == "mlp" else "")
    return f"{it['dataset']}_subset{it['train_subset_size']}_{head}_lr{it['lr']:g}_s{it['seed']}"


def _run(it: dict, root: str, val_size: int, threads: int) -> str:
    torch.set_num_threads(threads)
    out = Path(root) / _slug(it)
    if (out / "final.json").exists():
        return f"skip {out.name}"
    args = deepcopy(default_train_args())
    for k, v in it.items():
        setattr(args, k, v)
    args.reupload_mode = "none"
    args.val_size = val_size
    args.quantum_sim = "torch"
    args.weight_decay = 1e-4
    args.out_dir = str(out)
    if (out / "metrics.csv").exists():
        (out / "metrics.csv").unlink()  # restart an interrupted run from scratch
    run_training(args)
    return out.name


def _export(root: Path, dest: Path) -> int:
    n = 0
    for fin in sorted(root.glob("*/final.json")):
        out = dest / fin.parent.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("final.json", "metrics.csv", "meta.json"):
            shutil.copy2(fin.parent / name, out / name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["mnist_search", "cifar_val"], required=True)
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--threads_per_worker", type=int, default=2)
    ap.add_argument("--export_root", default=str(Path("results") / "protocol_v2"))
    a = ap.parse_args()
    root = a.output_root or str(Path("runs") / "protocol_v2" / a.grid)
    items = _grid(a.grid)
    print(f"{len(items)} runs -> {root}", flush=True)
    with ProcessPoolExecutor(max_workers=a.n_workers) as pool:
        futs = {pool.submit(_run, it, root, a.val_size, a.threads_per_worker): it for it in items}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(items)}] {f.result()}", flush=True)
            except Exception as exc:  # report and keep going
                print(f"[{i}/{len(items)}] FAILED {_slug(futs[f])}: {exc!r}", flush=True)
    n = _export(Path(root), Path(a.export_root) / a.grid)
    print(f"done; exported {n} finished runs to {Path(a.export_root) / a.grid}", flush=True)


if __name__ == "__main__":
    main()
