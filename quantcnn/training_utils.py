"""Training loop helpers and CSV logging."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class RunConfig:
    dataset: str
    backbone: str
    head: str
    bottleneck_dim: int
    train_subset_size: int | str | None
    seed: int
    epochs: int
    batch_size: int
    lr: float
    lr_backbone_multiplier: float
    gradient_clip_norm: float
    weight_decay: float
    reupload_mode: str
    n_var_layers: int
    mlp_hidden: int | None
    lr_scheduler: str
    val_size: int = 0
    quantum_sim: str = "pennylane"
    pre_norm: str = "none"
    freeze_circuit: bool = False
    encoding: str = "pi_tanh"
    entangle: bool | str = True
    vqc_init_scale: float = 0.1
    bottleneck_reg: str = "none"
    reg_weight: float = 0.0


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    return pred.eq(targets).float().mean()


def evaluate(model: nn.Module, loader, device):
    """Return avg loss & accuracy on loader."""
    model.eval()
    total_loss = total_acc = total_n = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb).item()
            acc = accuracy(logits, yb).item()
            bs = len(yb)
            total_loss += loss * bs
            total_acc += acc * bs
            total_n += bs
    if total_n == 0:
        return 0.0, 0.0
    return total_loss / total_n, total_acc / total_n


def _param_grad_norm(params) -> float:
    total = sum(p.grad.detach().norm(2).item() ** 2 for p in params if p.grad is not None)
    return total ** 0.5


def bottleneck_penalty(z: torch.Tensor, kind: str) -> torch.Tensor:
    """Regularizers on the bottleneck z (mechanism test). ``sat``: mean of relu(tanh(z)^2 - 0.81),
    nonzero only where |tanh z| > 0.9, the saturation threshold of the diagnostics. ``decor``: mean
    squared off-diagonal entry of the batch correlation matrix of z (decorrelation)."""
    if kind == "sat":
        return torch.relu(torch.tanh(z) ** 2 - 0.81).mean()
    if kind == "decor":
        zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
        c = zs.T @ zs / (len(z) - 1)
        q = c.shape[0]
        return (c - torch.diag(torch.diag(c))).pow(2).sum() / (q * (q - 1))
    raise ValueError(kind)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device,
    grad_clip: float,
    bottleneck_reg: str = "none",
    reg_weight: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return mean CE loss, accuracy, backbone grad norm, head grad norm. With ``bottleneck_reg``
    the optimized loss adds ``reg_weight * bottleneck_penalty(z)``; the returned loss is CE only."""
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_sum = acc_sum = bb_norm_sum = hd_norm_sum = n = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if bottleneck_reg == "none":
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
        else:
            z = model.backbone(xb)
            logits = model.head(model.pre_norm(z))
            loss = ce(logits, yb)
            (loss + reg_weight * bottleneck_penalty(z, bottleneck_reg)).backward()

        # Pre-clip gradient norms per component. The head norm includes its classical
        # readout, so it monitors trainability but cannot diagnose barren plateaus.
        bb_norm = _param_grad_norm(model.backbone.parameters()) if hasattr(model, "backbone") else 0.0
        hd_norm = _param_grad_norm(model.head.parameters()) if hasattr(model, "head") else 0.0

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = len(yb)
        loss_sum += loss.item() * bs
        acc_sum += accuracy(logits, yb).item() * bs
        bb_norm_sum += bb_norm * bs
        hd_norm_sum += hd_norm * bs
        n += bs
    return loss_sum / n, acc_sum / n, bb_norm_sum / n, hd_norm_sum / n


def append_metric_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def save_run_meta(out_dir: Path, cfg: RunConfig, extra: dict | None = None) -> None:
    meta = Path(out_dir) / "meta.json"
    payload = asdict(cfg)
    if extra:
        payload.update(extra)
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
