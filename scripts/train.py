"""Train hybrid models with reproducible CSV logging."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quantcnn.data import dataset_num_classes, get_cifar10, get_mnist
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
    p.add_argument("--head", choices=["linear", "mlp", "quantum"], default="quantum")
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
    )

    set_seed(int(args.seed))

    if args.dataset == "mnist":
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
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    backbone_params = list(model.backbone.parameters())
    head_params = [
        p for n, p in model.named_parameters() if not str(n).startswith("backbone.")
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
    save_run_meta(out_dir, cfg)

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
    return out_dir


def main():
    args = parse_args(None)
    run_training(args)


if __name__ == "__main__":
    main()
