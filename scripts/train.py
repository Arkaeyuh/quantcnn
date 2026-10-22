"""Train hybrid models with reproducible CSV logging."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quantcnn.data import dataset_num_classes, get_cifar10, get_mnist, get_splits
from quantcnn.models.torch_vqc import ENTANGLE
from quantcnn.models.hybrid_classifier import HybridClassifier
from quantcnn.training_utils import (
    RunConfig,
    append_metric_row,
    evaluate,
    save_run_meta,
    train_one_epoch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


FIELDNAMES = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "backbone_gnorm",
    "head_gnorm",
    "is_best",
    "dataset",
    "head",
    "reupload_mode",
    "train_subset_size",
    "seed",
    "bottleneck_dim",
]


def configure_train_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--backbone", default=None)
    p.add_argument("--head", choices=["linear", "mlp", "quantum", "cos", "cossin", "cosprod", "cosprod_last", "cosprod_pairs_last",
                                       "cosprod_win2", "cosprod_win3", "cosprod_win4", "cosprod_star2", "cosprod_star3",
                                       "cosprod_gap4", "cosprod_sg", "cosprod_last_sg", "cosprod_fl03", "cosprod_fl1",
                                       "cosprod_last_fl03", "cosprod_last_fl1"],
                   default="quantum")
    p.add_argument("--reupload_mode", choices=["none", "light", "strong"], default="none")
    p.add_argument("--bottleneck_dim", type=int, default=8)
    p.add_argument("--n_var_layers", type=int, default=4)
    p.add_argument("--mlp_hidden", type=int, default=None)
    p.add_argument("--train_subset_size", type=int, default=1000)
    p.add_argument("--full_train_set", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_backbone_multiplier", type=float, default=0.5)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="If >0, hold out this many training images (disjoint from the subset) for "
        "checkpoint selection; the test split is then evaluated once, on the selected "
        "checkpoint, and written to final.json. 0 keeps the original protocol.",
    )
    p.add_argument(
        "--quantum_sim",
        choices=["pennylane", "torch"],
        default="pennylane",
        help="Quantum head simulator; 'torch' is a batched, numerically equivalent backend.",
    )
    p.add_argument(
        "--pre_norm",
        choices=["none", "bn"],
        default="none",
        help="Normalization between bottleneck and head; 'bn' is BatchNorm1d with a learned affine.",
    )
    p.add_argument(
        "--freeze_circuit",
        action="store_true",
        help="Quantum head only: keep the VQC rotation angles at their initialization "
        "(the backbone and the readout Linear still train).",
    )
    p.add_argument(
        "--vqc_init_scale",
        type=float,
        default=0.1,
        help="Quantum head only: rotation angles start in uniform(-s, s) (default 0.1, near the identity).",
    )
    p.add_argument(
        "--encoding",
        choices=["pi_tanh", "half", "half_sym", "phase"],
        default="pi_tanh",
        help="Quantum head angle map (torch sim only): pi*tanh z; (pi/2)*tanh z + pi/2 (monotone "
        "under <Z>); (pi/2)*tanh z; or phase: H then RZ(pi*tanh z).",
    )
    p.add_argument("--no_entangle", action="store_true", help="Quantum head without CNOT rings (torch sim only).")
    p.add_argument("--entangle_last_only", action="store_true",
                   help="Quantum head with a CNOT ring after the last variational layer only (torch sim only).")
    p.add_argument("--entangle_pattern", choices=[e for e in ENTANGLE if isinstance(e, str) and e != "last"], default=None,
                   help="Quantum head with disjoint CNOT(0,1), CNOT(2,3), ... after every / the last "
                   "variational layer instead of rings (torch sim only).")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", default=None, help="Directory for meta.json & metrics.csv.")
    p.add_argument(
        "--runs_root",
        default=None,
        help="When out_dir omitted, prepended before auto slug (for sweeps).",
    )
    p.add_argument(
        "--log_csv",
        default=None,
        help="Explicit metrics CSV path (overrides out_dir/metrics.csv).",
    )
    return p


def default_train_args() -> argparse.Namespace:
    """Defaults only (useful for programmatic sweeps)."""
    p = argparse.ArgumentParser(add_help=False)
    configure_train_parser(p)
    return p.parse_args([])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    configure_train_parser(p)
    return p.parse_args(argv)


def _entangle(args) -> bool | str:
    if sum([args.no_entangle, args.entangle_last_only, args.entangle_pattern is not None]) > 1:
        raise ValueError("--no_entangle, --entangle_last_only and --entangle_pattern are exclusive.")
    if args.entangle_pattern is not None:
        return args.entangle_pattern
    return "last" if args.entangle_last_only else not args.no_entangle


def backbone_for_dataset(ds_name: str) -> str:
    return "mnist" if ds_name == "mnist" else "cifar10"


def build_out_dir_slug(
    *,
    dataset: str,
    train_subset_desc: str,
    head: str,
    reupload_mode: str,
    seed: int,
    bottleneck_dim: int,
) -> str:
    slug = (
        f"{dataset}_subset{train_subset_desc}"
        f"_{head}"
        f"_rup{reupload_mode}"
        f"_b{bottleneck_dim}_s{seed}"
    )
    return slug


def run_training(args: argparse.Namespace) -> Path:
    bb = args.backbone or backbone_for_dataset(args.dataset)
    subset_sz = None if args.full_train_set else args.train_subset_size
    subset_desc = str(subset_sz) if subset_sz is not None else "full"

    cfg = RunConfig(
        dataset=args.dataset,
        backbone=bb,
        head=args.head,
        bottleneck_dim=args.bottleneck_dim,
        train_subset_size=subset_desc,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone_multiplier=args.lr_backbone_multiplier,
        gradient_clip_norm=args.gradient_clip_norm,
        weight_decay=args.weight_decay,
        reupload_mode=args.reupload_mode,
        n_var_layers=args.n_var_layers,
        mlp_hidden=args.mlp_hidden,
        lr_scheduler=args.lr_scheduler,
        val_size=int(args.val_size),
        quantum_sim=args.quantum_sim,
        pre_norm=args.pre_norm,
        freeze_circuit=bool(args.freeze_circuit),
        encoding=args.encoding,
        entangle=_entangle(args),
        vqc_init_scale=float(args.vqc_init_scale),
    )

    set_seed(int(args.seed))

    splits = None
    if cfg.val_size > 0:
        if subset_sz is None:
            raise ValueError("--val_size requires a training subset.")
        splits = get_splits(args.dataset, subset_sz, args.seed, cfg.val_size, data_root=args.data_root)
        train_ds, val_ds = splits["train"], splits["val"]
        in_ch = 1 if args.dataset == "mnist" else 3
    elif args.dataset == "mnist":
        train_ds, val_ds = get_mnist(subset_sz, args.seed, data_root=args.data_root)
        in_ch = 1
    else:
        train_ds, val_ds = get_cifar10(subset_sz, args.seed, data_root=args.data_root)
        in_ch = 3

    n_classes = dataset_num_classes(train_ds)

    model = HybridClassifier(
        num_classes=n_classes,
        bottleneck_dim=args.bottleneck_dim,
        backbone=bb,
        head=args.head,
        in_channels=in_ch,
        mlp_hidden=args.mlp_hidden,
        n_var_layers=args.n_var_layers,
        reupload_mode=args.reupload_mode,
        quantum_sim=args.quantum_sim,
        pre_norm=args.pre_norm,
        encoding=args.encoding,
        entangle=_entangle(args),
    )
    if cfg.vqc_init_scale != 0.1:
        if args.head != "quantum":
            raise ValueError("--vqc_init_scale requires --head quantum.")
        with torch.no_grad():
            model.head.vqc.qlayer.weights.uniform_(-cfg.vqc_init_scale, cfg.vqc_init_scale)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    if cfg.freeze_circuit:
        if args.head != "quantum":
            raise ValueError("--freeze_circuit requires --head quantum.")
        model.head.vqc.qlayer.weights.requires_grad_(False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    backbone_params = list(model.backbone.parameters())
    head_params = [
        p
        for n, p in model.named_parameters()
        if not str(n).startswith("backbone.") and p.requires_grad
    ]
    if not head_params:
        raise RuntimeError("No head parameters found.")
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.lr_backbone_multiplier},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    if args.log_csv:
        csv_path = Path(args.log_csv)
        out_dir = csv_path.parent
    elif getattr(args, "out_dir", None):
        out_dir = Path(args.out_dir)
        csv_path = out_dir / "metrics.csv"
    else:
        slug = build_out_dir_slug(
            dataset=cfg.dataset,
            train_subset_desc=subset_desc,
            head=cfg.head,
            reupload_mode=str(cfg.reupload_mode),
            seed=int(cfg.seed),
            bottleneck_dim=cfg.bottleneck_dim,
        )
        base = getattr(args, "runs_root", None)
        root = Path(base) if base is not None else Path("runs") / "logs"
        out_dir = root / slug
        csv_path = out_dir / "metrics.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_meta(out_dir, cfg, extra=_provenance(train_ds))

    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = None

    best_val_acc = -1.0
    checkpoint_path = out_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, bb_gnorm, hd_gnorm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg.gradient_clip_norm,
        )
        va_loss, va_acc = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            torch.save(model.state_dict(), checkpoint_path)

        row = {
            "epoch": epoch,
            "train_loss": f"{tr_loss:.6f}",
            "train_acc": f"{tr_acc:.6f}",
            "val_loss": f"{va_loss:.6f}",
            "val_acc": f"{va_acc:.6f}",
            "backbone_gnorm": f"{bb_gnorm:.6f}",
            "head_gnorm": f"{hd_gnorm:.6f}",
            "is_best": int(is_best),
            "dataset": cfg.dataset,
            "head": cfg.head,
            "reupload_mode": cfg.reupload_mode,
            "train_subset_size": subset_sz if subset_sz is not None else "full",
            "seed": cfg.seed,
            "bottleneck_dim": cfg.bottleneck_dim,
        }
        append_metric_row(csv_path, row, FIELDNAMES)

    if splits is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        best_epoch = _best_epoch(csv_path)
        final = {"best_epoch": best_epoch}
        for name in ("train_clean", "val", "test"):
            loader = DataLoader(splits[name], batch_size=256, shuffle=False, num_workers=0)
            loss, acc = evaluate(model, loader, device)
            final[f"{name}_loss"], final[f"{name}_acc"] = loss, acc
        (out_dir / "final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return out_dir


def _best_epoch(csv_path: Path) -> int:
    """Last epoch flagged is_best (the checkpoint on disk)."""
    import csv

    with csv_path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_best"] == "1"]
    return int(rows[-1]["epoch"])


def _provenance(train_ds) -> dict:
    base = getattr(train_ds, "dataset", train_ds)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain", "--", "quantcnn", "scripts"],
                           capture_output=True, text=True).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "train_transform": repr(getattr(base, "transform", None)),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def main():
    args = parse_args(None)
    run_training(args)


if __name__ == "__main__":
    main()
