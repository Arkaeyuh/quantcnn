#!/usr/bin/env python3
"""
Train a grid over stratified subset sizes, classical vs quantum heads,
and (for quantum) data re-upload variants.

From repository root::

    PYTHONPATH=. python -m scripts.run_sweep --quick
    PYTHONPATH=. python -m scripts.run_sweep --epochs 25 --n_workers 4
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from scripts.train import build_out_dir_slug, default_train_args, run_training


def parse_sweep_argv() -> argparse.Namespace:
    sp = argparse.ArgumentParser(description="Grid training for hybrid models.")
    sp.add_argument(
        "--output_root",
        default=None,
        help="Sweep output directory. Default ./runs/logs/sweep_<timestamp>",
    )
    sp.add_argument(
        "--quick",
        action="store_true",
        help="Tiny grid (one subset size, fewer seeds/shorter epochs) for smoke-testing.",
    )
    sp.add_argument("--epochs", type=int, default=15)
    sp.add_argument("--batch_size", type=int, default=32)
    sp.add_argument("--bottleneck_dim", type=int, default=8)
    sp.add_argument("--n_var_layers", type=int, default=4)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--lr_backbone_multiplier", type=float, default=0.5)
    sp.add_argument("--gradient_clip_norm", type=float, default=1.0)
    sp.add_argument("--data_root", default="./data")
    sp.add_argument("--lr_scheduler", choices=["cosine", "none"], default="cosine")
    sp.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel training jobs. 0 = use all CPU cores. Default 1 (sequential).",
    )
    return sp.parse_args()


def _build_run_args(swe: argparse.Namespace, root: Path, epochs: int, subsets: list, seeds: list) -> list:
    """Pre-build all training arg namespaces for the sweep grid."""
    grid = []

    for n in subsets:
        for hd in ["linear", "mlp"]:
            for sd in seeds:
                grid.append({"train_subset_size": n, "head": hd, "reupload_mode": "none", "seed": sd})

    for n in subsets:
        for ru in ["none", "light", "strong"]:
            for sd in seeds:
                grid.append({"train_subset_size": n, "head": "quantum", "reupload_mode": ru, "seed": sd})

    all_args = []
    for item in grid:
        args = deepcopy(default_train_args())
        args.dataset = "mnist"
        args.epochs = epochs
        args.batch_size = swe.batch_size
        args.bottleneck_dim = swe.bottleneck_dim
        args.n_var_layers = swe.n_var_layers
        args.lr = swe.lr
        args.lr_backbone_multiplier = swe.lr_backbone_multiplier
        args.gradient_clip_norm = swe.gradient_clip_norm
        args.data_root = swe.data_root
        args.full_train_set = False
        args.out_dir = None
        args.log_csv = None
        args.lr_scheduler = swe.lr_scheduler
        args.train_subset_size = item["train_subset_size"]
        args.head = item["head"]
        args.reupload_mode = item["reupload_mode"]
        args.seed = item["seed"]

        slug = build_out_dir_slug(
            dataset=args.dataset,
            train_subset_desc=str(args.train_subset_size),
            head=args.head,
            reupload_mode=args.reupload_mode,
            seed=int(args.seed),
            bottleneck_dim=int(args.bottleneck_dim),
        )
        args.out_dir = str(root / slug)
        all_args.append(args)

    return all_args


def main():
    swe = parse_sweep_argv()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(swe.output_root) if swe.output_root else Path("runs") / "logs" / f"sweep_{stamp}"

    if swe.quick:
        subsets = [500]
        seeds = [0]
        epochs = min(6, max(3, swe.epochs))
    else:
        subsets = [250, 500, 1000]
        seeds = [0, 1, 2]
        epochs = swe.epochs

    all_args = _build_run_args(swe, root, epochs, subsets, seeds)

    n_workers = swe.n_workers if swe.n_workers != 0 else (os.cpu_count() or 1)

    if n_workers <= 1:
        for args in all_args:
            run_training(args)
    else:
        print(f"Running {len(all_args)} jobs across {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(run_training, args): args for args in all_args}
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    print(f"  [{completed}/{len(all_args)}] done")
                except Exception as exc:
                    args = futures[future]
                    print(f"  FAILED {args.head}/{args.reupload_mode}/s{args.seed}: {exc}")

    print(f"Sweep complete. Logs under: {root}")


if __name__ == "__main__":
    main()
