#!/usr/bin/env python3
"""Linear-probe the backbones from the validation-protocol runs.

For every (dataset, training size, head) cell of the protocol grids, take the
learning rate selected by mean validation accuracy (as in ``summarize_v2``),
load the three seeds' checkpoints, extract the bottleneck embedding ``z`` (the
backbone output, before any pre-encoding normalization) in ``eval()`` mode on
clean images, and fit a fresh linear probe on it with the same procedure as
``scripts.frozen_heads`` (epoch and probe learning rate selected on validation).
Because the probe is identical for every backbone, differences in probe accuracy
compare the representations the different heads induced in joint training.

Also records the fraction of test embedding values in the flat region of the
encoding, |tanh(.)| > 0.9, both for ``z`` and for what the head actually sees
(after BatchNorm when the run used ``--pre_norm bn``).

From repository root::

    PYTHONPATH=. python -m scripts.probe_backbones --out results/backbone_probe.csv

Mechanism audit (``run_protocol_v2 --grid cifar_mech``; heads are keyed with
``+frozen`` and ``@bblr<absolute backbone lr>`` where they differ from the default)::

    PYTHONPATH=. python -m scripts.probe_backbones --datasets cifar10 --grids cifar_mech \
        --heads all --out results/mech_probe.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed, _train_head

GRIDS = {"mnist": ["mnist_search", "mnist_bn"], "cifar10": ["cifar_val", "cifar_bn"]}
HEADS = ["linear", "mlp8", "quantum", "linear+bn", "quantum+bn"]
PROBE_LRS = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
MLP_PROBE_LRS = [1e-3, 3e-3, 1e-2, 3e-2]


def _head_key(meta: dict) -> str:
    head = meta["head"] + (str(meta["mlp_hidden"]) if meta["head"] == "mlp" else "")
    head += "+bn" if meta.get("pre_norm", "none") == "bn" else ""
    head += "+frozen" if meta.get("freeze_circuit") else ""
    head += f"+{meta['encoding']}" if meta.get("encoding", "pi_tanh") != "pi_tanh" else ""
    head += "+noent" if meta.get("entangle", True) is False else ""
    head += "+entlast" if meta.get("entangle", True) == "last" else ""
    ent = meta.get("entangle", True)
    head += f"+{ent}" if isinstance(ent, str) and ent != "last" else ""
    head += f"+ro{meta['readout']}" if meta.get("readout", "z") != "z" else ""
    head += f"+L{meta['n_var_layers']}" if meta["head"] == "quantum" and meta.get("n_var_layers", 4) != 4 else ""
    head += f"+init{meta['vqc_init_scale']:.3g}" if meta.get("vqc_init_scale", 0.1) != 0.1 else ""
    head += f"+{meta['bottleneck_reg']}{meta['reg_weight']:g}" if meta.get("bottleneck_reg", "none") != "none" else ""
    pretrained = meta.get("backbone") not in (None, "mnist", "cifar10")
    head += f"+ep{meta['epochs']}" if meta["epochs"] != {"mnist": 50, "cifar10": 80, "svhn": 80}[meta["dataset"]] and not pretrained else ""
    head += f"+{meta['backbone']}" if pretrained else ""
    head += f"+ru{meta['reupload_mode']}" if meta.get("reupload_mode", "none") != "none" else ""
    mult = meta.get("lr_backbone_multiplier", 0.5)
    return head + (f"@bblr{meta['lr'] * mult:.3g}" if mult != 0.5 and not pretrained else "")


def _selected_runs(runs_root: Path, grids: list[str]) -> dict[tuple[int, str], tuple[float, list[Path]]]:
    """(size, head) -> (selected lr, run dirs for seeds 0-2), selection on mean val acc."""
    cells = defaultdict(lambda: defaultdict(list))
    for grid in grids:
        for fin in sorted((runs_root / grid).glob("*/final.json")):
            meta = json.loads((fin.parent / "meta.json").read_text())
            cells[(int(meta["train_subset_size"]), _head_key(meta))][meta["lr"]].append(
                (json.loads(fin.read_text())["val_acc"], fin.parent))
    out = {}
    for key, by_lr in cells.items():
        lr = max(by_lr, key=lambda k: st.mean(v for v, _ in by_lr[k]))
        out[key] = (lr, sorted((d for _, d in by_lr[lr]), key=lambda d: d.name))
    return out


def _all_runs(runs_root: Path, grids: list[str]) -> dict[tuple[int, str], list[tuple[float, list[Path]]]]:
    """(size, head) -> [(lr, run dirs)] for every lr (lockstep comparisons pair runs at a fixed lr)."""
    cells = defaultdict(lambda: defaultdict(list))
    for grid in grids:
        for fin in sorted((runs_root / grid).glob("*/final.json")):
            meta = json.loads((fin.parent / "meta.json").read_text())
            cells[(int(meta["train_subset_size"]), _head_key(meta))][meta["lr"]].append(fin.parent)
    return {key: [(lr, sorted(d, key=lambda p: p.name)) for lr, d in sorted(by_lr.items())] for key, by_lr in cells.items()}


@torch.no_grad()
def _head_input(model: HybridClassifier, z: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.pre_norm(z.to(next(model.parameters()).device)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/backbone_probe.csv")
    ap.add_argument("--runs_root", default="runs/protocol_v2")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"])
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grids", nargs="+", default=None, help="Override the grids read (default: GRIDS[dataset]).")
    ap.add_argument("--heads", nargs="+", default=HEADS, help="Head keys to probe, or 'all'.")
    ap.add_argument("--all_lrs", action="store_true",
                    help="Probe every lr of each head instead of the validation-selected one.")
    ap.add_argument("--mlp_lrs", nargs="+", type=float, default=MLP_PROBE_LRS,
                    help="Learning-rate grid of the MLP probe.")
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear",
                    help="linear: the common linear probe; mlp: an 8-256-256-10 ReLU probe on standardized z.")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["dataset"], r["train_subset_size"], r["source_head"], r["seed"])
                    + ((r["source_lr"],) if a.all_lrs else ()) for r in csv.DictReader(fh)}
    for ds in a.datasets:
        grids = a.grids or GRIDS[ds]
        if a.all_lrs:
            cells = [((n, head), lr_runs) for (n, head), v in sorted(_all_runs(Path(a.runs_root), grids).items())
                     for lr_runs in v]
        else:
            cells = sorted(_selected_runs(Path(a.runs_root), grids).items())
        for (n, head), (lr, runs) in cells:
            if a.heads != ["all"] and head not in a.heads:
                continue
            assert a.all_lrs or len(runs) == 3 or a.grids in (["cifar_seeds"], ["night_seeds"], ["cifar_long2"], ["pretrained2_seeds"], ["cifar_family_seeds"], ["cifar_half_seeds"], ["cifar_phase_seeds"], ["cifar_floor_seeds"], ["cifar_q12_seeds"], ["cifar_mech2_seeds"], ["cifar_mech3_seeds"], ["cifar_lastring_retest"], ["cifar_st"], ["cifar_satup"], ["night26_seeds"], ["cifar_esc"], ["cifar_esc_rep"], ["svhn_seeds"], ["cifar_reup_seeds"], ["cifar_mnv3_seeds"], ["cifar_escgate"], ["cifar_escgate_rep"], ["cifar_reup_seeds2"], ["cifar_mnv3_seeds2"]), (ds, n, head, lr)
            for run in runs:
                meta = json.loads((run / "meta.json").read_text())
                if meta["dataset"] != ds:
                    continue  # mixed-dataset grids (night_seeds)
                seed = int(meta["seed"])
                if (ds, str(n), head, str(seed)) + ((f"{lr:g}",) if a.all_lrs else ()) in done:
                    continue
                model = HybridClassifier(10, int(meta.get("bottleneck_dim", 8)), meta["backbone"], meta["head"],
                                         1 if ds == "mnist" else 3,
                                         mlp_hidden=meta.get("mlp_hidden"), n_var_layers=meta.get("n_var_layers", 4),
                                         reupload_mode=meta.get("reupload_mode", "none"),
                                         quantum_sim="torch", pre_norm=meta.get("pre_norm", "none"),
                                         encoding=meta.get("encoding", "pi_tanh"),
                                         entangle=meta.get("entangle", True))
                model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
                model.to(device)
                sp = get_splits(ds, n, seed, a.val_size)
                emb = {"train": _embed(model, sp["train_clean"], device),
                       "val": _embed(model, sp["val"], device),
                       "test": _embed(model, sp["test"], device)}
                z_test = emb["test"][0]
                sat_z = (torch.tanh(z_test).abs() > 0.9).float().mean().item()
                sat_in = (torch.tanh(_head_input(model, z_test)).abs() > 0.9).float().mean().item()
                best = None
                for plr in (PROBE_LRS if a.probe == "linear" else a.mlp_lrs):
                    r = _train_head("linear" if a.probe == "linear" else "mlp_probe", emb, seed, plr, 60)
                    if best is None or r["val_acc"] > best[1]["val_acc"]:
                        best = (plr, r)
                plr, r = best
                row = dict(dataset=ds, train_subset_size=n, source_head=head, source_lr=f"{lr:g}", seed=seed,
                           source_test_acc=round(json.loads((run / "final.json").read_text())["test_acc"] * 100, 3),
                           probe_lr=f"{plr:g}", probe_val_acc=round(r["val_acc"] * 100, 3),
                           probe_train_acc=round(r["train_acc"] * 100, 3), probe_test_acc=round(r["test_acc"] * 100, 3),
                           sat_090_z=round(sat_z * 100, 2), sat_090_head_input=round(sat_in * 100, 2))
                new = not out.exists()
                with out.open("a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(row, flush=True)


if __name__ == "__main__":
    main()
