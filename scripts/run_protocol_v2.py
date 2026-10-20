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
* ``mnist_bn`` / ``cifar_bn``: the quantum head and, as a control, the linear head
  with BatchNorm1d between bottleneck and head (``--pre_norm bn``), on the same
  recipes; the learning-rate grids go one step higher than the plain grids.
* ``cifar_mech``: mechanism audit at CIFAR-10, 4000 images. (a) Backbone learning
  rate swept at a fixed head learning rate (each head's ``cifar_val`` selection:
  linear 3e-2, quantum 1e-2) over absolute backbone rates {2.5e-3, 5e-3, 1.5e-2,
  3e-2, 6e-2}, omitting the rate the ``cifar_val`` run already used; (b) quantum
  head with the circuit frozen at initialization (``--freeze_circuit``), head
  learning rate {3e-3, 1e-2, 3e-2, 1e-1} at the default backbone multiplier.
* ``cifar_circuit``: circuit ablations at CIFAR-10, 4000 images, head learning
  rate {3e-3, 1e-2, 3e-2}: encoding (pi/2)*tanh z + pi/2, monotone under <Z>
  (``half``); its even counterpart (pi/2)*tanh z with the same range width
  (``half_sym``); and the default encoding without CNOT rings (``noent``).
* ``cifar_entangle``: how much entanglement costs, same recipe: a CNOT ring after
  the last variational layer only (``entlast``), and full rings with 1 or 2
  variational layers instead of 4 (``L1``, ``L2``).
* ``cifar_classical``: classical controls for the CNOT-free circuit, same recipe:
  cos(pi*tanh z) -> Linear (``cos``) and [cos, sin](pi*tanh z) -> Linear
  (``cossin``), head learning rate {3e-3, 1e-2, 3e-2, 1e-1}.
* ``cifar_cosprod``: classical heads that reproduce the entangled circuits at zero
  rotation angles, same recipe: products of cos(pi*tanh z_j) over each readout's
  CNOT-ring support, then Linear, for the default circuit (``cosprod``, mean
  support 2.5) and the last-ring-only circuit (``cosprod_last``, 5.25); head
  learning rate {1e-3, 3e-3, 1e-2, 3e-2}.
* ``cifar_pairs``: entangling but low-order circuits, same recipe: disjoint
  CNOT(0,1), CNOT(2,3), ... after every variational layer (``pairs``; zero-rotation
  product order 1) or the last one only (``pairs_last``; order 1.5), head learning rate
  {3e-3, 1e-2, 3e-2}; and the classical product head on the ``pairs_last`` supports
  (``cosprod_pairs_last``), {1e-3, 3e-3, 1e-2, 3e-2}.
* ``cifar_seeds``: seeds 3 and 4 for the headline CIFAR-10/4000 cells (``SEED_CELLS``),
  each at the learning rate validation selected on seeds 0-2 in its source grid.
  Run it after the source grids are done.
* ``mnist_circuit``: the ``cifar_circuit`` variants on MNIST, 500 and 1000 images,
  on the ``mnist_search`` recipe (50 epochs, clip 1.0), head learning rate
  {3e-3, 1e-2, 3e-2}.

From repository root::

    PYTHONPATH=. python -m scripts.run_protocol_v2 --grid mnist_search --n_workers 4

Runs (with checkpoints) go to ``runs/protocol_v2/<grid>``; when the grid is done,
each finished run's ``final.json``, ``metrics.csv`` and ``meta.json`` (no
checkpoint) are copied to ``results/protocol_v2/<grid>``, which is what
``scripts.summarize_v2`` reads.
"""

from __future__ import annotations

import argparse
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import torch

from scripts.train import default_train_args, run_training

SEEDS = [0, 1, 2]
DEFAULT_BB_MULT = 0.5
MECH_HEAD_LR = {"linear": 3e-2, "quantum": 1e-2}
MECH_BACKBONE_LRS = [2.5e-3, 5e-3, 1.5e-2, 3e-2, 6e-2]
EXTRA_SEEDS = [3, 4]
# (source grid, probe head key, run options) for the cells that get seeds 3-4.
DEFAULT_EPOCHS = {"mnist": 50, "cifar10": 80}
# Pretrained ImageNet ResNet-18 fine-tuned at 128 px; backbone lr 0.1x head lr.
_PRETRAINED = dict(dataset="cifar10", train_subset_size=4000, mlp_hidden=None, epochs=30, gradient_clip_norm=2.0,
                   backbone="resnet18_pt", lr_backbone_multiplier=0.1)
_C4000 = dict(dataset="cifar10", train_subset_size=4000, mlp_hidden=None, epochs=80, gradient_clip_norm=2.0)
SEED_CELLS = [
    ("cifar_val", "linear", dict(_C4000, head="linear")),
    ("cifar_val", "mlp8", dict(_C4000, head="mlp", mlp_hidden=8)),
    ("cifar_val", "quantum", dict(_C4000, head="quantum")),
    ("cifar_circuit", "quantum+noent", dict(_C4000, head="quantum", no_entangle=True)),
    ("cifar_entangle", "quantum+entlast", dict(_C4000, head="quantum", entangle_last_only=True)),
    ("cifar_classical", "cos", dict(_C4000, head="cos")),
    ("cifar_cosprod", "cosprod", dict(_C4000, head="cosprod")),
    ("cifar_cosprod", "cosprod_last", dict(_C4000, head="cosprod_last")),
    ("cifar_pairs", "quantum+pairs", dict(_C4000, head="quantum", entangle_pattern="pairs")),
    ("cifar_pairs", "quantum+pairs_last", dict(_C4000, head="quantum", entangle_pattern="pairs_last")),
    ("cifar_pairs", "cosprod_pairs_last", dict(_C4000, head="cosprod_pairs_last")),
]
_M1000 = dict(dataset="mnist", train_subset_size=1000, mlp_hidden=None, epochs=50, gradient_clip_norm=1.0)
# Extra seeds (3-4) for close contrasts and the pretrained cells, at each cell's selected lr.
NIGHT_SEED_CELLS = [
    ("cifar_pretrained", "linear+resnet18_pt", dict(_PRETRAINED, head="linear")),
    ("cifar_pretrained", "quantum+resnet18_pt", dict(_PRETRAINED, head="quantum")),
    ("cifar_pretrained", "quantum+noent+resnet18_pt", dict(_PRETRAINED, head="quantum", no_entangle=True)),
    ("cifar_pretrained", "quantum+entlast+resnet18_pt", dict(_PRETRAINED, head="quantum", entangle_last_only=True)),
    *[("cifar_support", f"cosprod_{s}", dict(_C4000, head=f"cosprod_{s}")) for s in ("star2", "win3", "star3", "win4", "gap4")],
    ("mnist_search", "quantum", dict(_M1000, head="quantum")),
    ("mnist_order", "quantum+pairs_last", dict(_M1000, head="quantum", entangle_pattern="pairs_last")),
]
EDGE_EXTENSION = {
    "mnist_search": [("mnist", n, h, hid, 1e-1) for n in [250, 500, 1000] for h, hid in [("linear", None), ("mlp", 8)]]
                    + [("mnist", 1000, "mlp", 7, 1e-1)],
    "cifar_val": [("cifar10", 2000, "linear", None, 3e-2), ("cifar10", 2000, "quantum", None, 3e-2)]
                 + [("cifar10", 4000, h, hid, 3e-2) for h, hid in [("linear", None), ("mlp", 8), ("quantum", None)]],
    "mnist_bn": [("mnist", n, "quantum", None, 3e-1, "bn") for n in [250, 500, 1000]],
    "cifar_bn": [("cifar10", 1000, "quantum", None, 1e-1, "bn"), ("cifar10", 4000, "quantum", None, 1e-1, "bn"),
                 ("cifar10", 2000, "linear", None, 1e-1, "bn"), ("cifar10", 4000, "linear", None, 1e-1, "bn")],
    "mnist_circuit": [("mnist", n, "quantum", None, 1e-1, dict(encoding="half")) for n in [500, 1000]],
    "mnist_order": [("mnist", 1000, "quantum", None, 1e-1, dict(entangle_pattern="pairs_last")),
                    ("mnist", 1000, "cosprod_last", None, 1e-1, {})],
    # Selected lr at the grid minimum: one step down.
    "cifar_sg": [("cifar10", 4000, "cosprod_sg", None, 1e-3, {}), ("cifar10", 4000, "cosprod_last_sg", None, 1e-3, {})],
}
# Linear head still at the top after the first extension: a second step up.
EDGE_EXTENSION["mnist_search"] += [("mnist", n, "linear", None, 3e-1) for n in [250, 500, 1000]]
EDGE_EXTENSION["cifar_val"] += [("cifar10", n, "linear", None, 1e-1) for n in [2000, 4000]]


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
    elif name in ("mnist_bn", "cifar_bn"):
        mnist = name == "mnist_bn"
        lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1] if mnist else [1e-3, 3e-3, 1e-2, 3e-2]
        for n in ([250, 500, 1000] if mnist else [1000, 2000, 4000]):
            for head in ["linear", "quantum"]:
                for lr in lrs:
                    for sd in SEEDS:
                        items.append(dict(dataset="mnist" if mnist else "cifar10", train_subset_size=n,
                                          head=head, mlp_hidden=None, lr=lr, seed=sd, pre_norm="bn",
                                          epochs=50 if mnist else 80, gradient_clip_norm=1.0 if mnist else 2.0))
    elif name == "cifar_mech":
        common = dict(dataset="cifar10", train_subset_size=4000, mlp_hidden=None, epochs=80, gradient_clip_norm=2.0)
        for head, lr in MECH_HEAD_LR.items():
            for bb_lr in MECH_BACKBONE_LRS:
                mult = round(bb_lr / lr, 6)
                if mult == DEFAULT_BB_MULT:
                    continue  # already in cifar_val
                for sd in SEEDS:
                    items.append(dict(common, head=head, lr=lr, lr_backbone_multiplier=mult, seed=sd))
        for lr in [3e-3, 1e-2, 3e-2, 1e-1]:
            for sd in SEEDS:
                items.append(dict(common, head="quantum", lr=lr, freeze_circuit=True, seed=sd))
    elif name == "cifar_circuit":
        common = dict(dataset="cifar10", train_subset_size=4000, head="quantum", mlp_hidden=None, epochs=80,
                      gradient_clip_norm=2.0)
        for variant in [dict(encoding="half"), dict(encoding="half_sym"), dict(no_entangle=True)]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(common, **variant, lr=lr, seed=sd))
    elif name == "cifar_entangle":
        common = dict(dataset="cifar10", train_subset_size=4000, head="quantum", mlp_hidden=None, epochs=80,
                      gradient_clip_norm=2.0)
        for variant in [dict(entangle_last_only=True), dict(n_var_layers=1), dict(n_var_layers=2)]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(common, **variant, lr=lr, seed=sd))
    elif name == "cifar_classical":
        common = dict(dataset="cifar10", train_subset_size=4000, mlp_hidden=None, epochs=80, gradient_clip_norm=2.0)
        for head in ["cos", "cossin"]:
            for lr in [3e-3, 1e-2, 3e-2, 1e-1]:
                for sd in SEEDS:
                    items.append(dict(common, head=head, lr=lr, seed=sd))
    elif name == "cifar_cosprod":
        for head in ["cosprod", "cosprod_last"]:
            for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(_C4000, head=head, lr=lr, seed=sd))
    elif name == "cifar_pairs":
        for pattern in ["pairs", "pairs_last"]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(_C4000, head="quantum", entangle_pattern=pattern, lr=lr, seed=sd))
        for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
            for sd in SEEDS:
                items.append(dict(_C4000, head="cosprod_pairs_last", lr=lr, seed=sd))
    elif name == "cifar_support":
        for head in ["cosprod_win2", "cosprod_star2", "cosprod_win3", "cosprod_star3", "cosprod_win4", "cosprod_gap4"]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(_C4000, head=head, lr=lr, seed=sd))
    elif name == "cifar_sg":
        for head in ["cosprod_sg", "cosprod_last_sg"]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(_C4000, head=head, lr=lr, seed=sd))
    elif name in ("cifar_pretrained", "cifar_pretrained2"):
        variants = ([dict(head="linear"), dict(head="quantum"), dict(head="quantum", no_entangle=True),
                     dict(head="quantum", entangle_last_only=True)] if name == "cifar_pretrained" else
                    [dict(head="cosprod"), dict(head="quantum", entangle_pattern="pairs")])
        for variant in variants:
            for lr in [1e-3, 3e-3, 1e-2]:
                for sd in SEEDS:
                    items.append(dict(_PRETRAINED, **variant, lr=lr, seed=sd))
    elif name == "cifar_pretrained_edge":
        # One more lr step (into the cifar_pretrained run dir) for cells selected at a grid edge.
        from scripts.probe_backbones import _selected_runs

        grid_lrs = [1e-3, 3e-3, 1e-2]
        sel = _selected_runs(Path("runs") / "protocol_v2", ["cifar_pretrained"])
        for _, key, opts in NIGHT_SEED_CELLS[:4]:
            lr = sel.get((4000, key), (None,))[0]
            step = {grid_lrs[0]: 3e-4, grid_lrs[-1]: 3e-2}.get(lr)
            if step is not None:
                for sd in SEEDS:
                    items.append(dict(opts, lr=step, seed=sd))
    elif name == "cifar_pretrained_10k_lr":
        # Per-size lr selection at 10,000 images (the 10k grid reused the 4,000-image lrs).
        for _, _, opts in NIGHT_SEED_CELLS[:4]:
            for lr in [3e-4, 1e-3, 3e-3]:
                for sd in SEEDS:
                    items.append(dict(opts, train_subset_size=10000, lr=lr, seed=sd))
    elif name == "cifar_init":
        # Rotations start in uniform(-pi, pi) instead of near the identity.
        common = dict(dataset="cifar10", train_subset_size=4000, head="quantum", mlp_hidden=None, epochs=80,
                      gradient_clip_norm=2.0, vqc_init_scale=math.pi)
        for variant in [{}, dict(no_entangle=True)]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(common, **variant, lr=lr, seed=sd))
    elif name == "cifar_pretrained_zero":
        # Default circuit with rotations frozen at exactly zero: the forward map equals the products head.
        for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
            for sd in SEEDS:
                items.append(dict(_PRETRAINED, head="quantum", freeze_circuit=True, vqc_init_scale=0.0, lr=lr, seed=sd))
    elif name == "cifar_pretrained_budget":
        # Same number of optimizer updates as 4,000 images x 30 epochs; each size's selected lr.
        for n, epochs, lrs in [(1000, 120, dict(linear=1e-3, quantum=1e-3, noent=1e-3)),
                               (10000, 12, dict(linear=3e-4, quantum=3e-4, noent=1e-3))]:
            for key, variant in [("linear", dict(head="linear")), ("quantum", dict(head="quantum")),
                                 ("noent", dict(head="quantum", no_entangle=True))]:
                for sd in SEEDS:
                    items.append(dict(_PRETRAINED, **variant, train_subset_size=n, epochs=epochs, lr=lrs[key], seed=sd))
    elif name == "cifar_pretrained_10k_edge":
        # Linear, default and last-ring heads selected 3e-4 (grid bottom) at 10k images: one step down.
        for i in (0, 1, 3):
            for sd in SEEDS:
                items.append(dict(NIGHT_SEED_CELLS[i][2], train_subset_size=10000, lr=1e-4, seed=sd))
    elif name == "cifar_pretrained2_edge":
        # cosprod and pairs selected 1e-3, the bottom of their grid: one more step down.
        for opts in [dict(_PRETRAINED, head="cosprod"), dict(_PRETRAINED, head="quantum", entangle_pattern="pairs")]:
            for sd in SEEDS:
                items.append(dict(opts, lr=3e-4, seed=sd))
    elif name == "cifar_pretrained_edge2":
        # The default and last-ring circuits selected 3e-4 (already one step below the grid): one more step.
        for _, key, opts in NIGHT_SEED_CELLS[1:4:2]:
            for sd in SEEDS:
                items.append(dict(opts, lr=1e-4, seed=sd))
    elif name in ("cifar_pretrained_10k", "cifar_pretrained_1k"):
        # Data-scale check: 10,000 / 1,000 training images, each head at its lr selected on 4,000.
        from scripts.probe_backbones import _selected_runs

        sel = _selected_runs(Path("runs") / "protocol_v2", ["cifar_pretrained"])
        for _, key, opts in NIGHT_SEED_CELLS[:4]:
            if (4000, key) in sel:
                for sd in SEEDS:
                    items.append(dict(opts, train_subset_size=10000 if name.endswith("10k") else 1000,
                                      lr=sel[(4000, key)][0], seed=sd))
    elif name == "cifar_pretrained_1k_lr":
        # Proper per-size lr selection at 1,000 images (the 1k grid reused the 4,000-image lrs).
        for _, _, opts in NIGHT_SEED_CELLS[:4]:
            for lr in [3e-4, 1e-3, 3e-3]:
                for sd in SEEDS:
                    items.append(dict(opts, train_subset_size=1000, lr=lr, seed=sd))
    elif name == "pretrained2_seeds":
        from scripts.probe_backbones import _selected_runs

        sel = _selected_runs(Path("runs") / "protocol_v2", ["cifar_pretrained2"])
        for key, opts in [("cosprod+resnet18_pt", dict(_PRETRAINED, head="cosprod")),
                          ("quantum+pairs+resnet18_pt", dict(_PRETRAINED, head="quantum", entangle_pattern="pairs"))]:
            for sd in EXTRA_SEEDS:
                items.append(dict(opts, lr=sel[(4000, key)][0], seed=sd))
    elif name == "cifar_long2":
        # Linear head at 2x budget (its selected lr), plus seeds 3-4 for the 2x-budget circuits.
        for sd in range(5):
            items.append(dict(_C4000, head="linear", lr=3e-2, seed=sd, epochs=160))
        for variant in [dict(no_entangle=True), dict()]:
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, head="quantum", **variant, lr=1e-2, seed=sd, epochs=160))
    elif name == "night_seeds":
        from scripts.probe_backbones import _selected_runs

        for grid, key, opts in NIGHT_SEED_CELLS:
            sel = _selected_runs(Path("runs") / "protocol_v2", [grid]).get((opts["train_subset_size"], key))
            if sel is None:
                continue  # source grid not finished
            for sd in EXTRA_SEEDS:
                items.append(dict(opts, lr=sel[0], seed=sd))
    elif name == "cifar_long":
        # 2x training budget at the validation-selected head lr (1e-2 for all three circuits).
        for variant in [dict(no_entangle=True), dict(), dict(entangle_last_only=True)]:
            for sd in SEEDS:
                items.append(dict(_C4000, head="quantum", **variant, lr=1e-2, seed=sd, epochs=160))
    elif name == "mnist_order":
        common = dict(dataset="mnist", train_subset_size=1000, mlp_hidden=None, epochs=50, gradient_clip_norm=1.0)
        for variant in [dict(head="quantum", entangle_last_only=True), dict(head="quantum", entangle_pattern="pairs_last"),
                        dict(head="cosprod"), dict(head="cosprod_last")]:
            for lr in [3e-3, 1e-2, 3e-2]:
                for sd in SEEDS:
                    items.append(dict(common, **variant, lr=lr, seed=sd))
    elif name == "cifar_seeds":
        from scripts.probe_backbones import _selected_runs

        for grid, key, opts in SEED_CELLS:
            sel = _selected_runs(Path("runs") / "protocol_v2", [grid]).get((opts["train_subset_size"], key))
            if sel is None:
                raise RuntimeError(f"{grid} has no finished {key} runs to select an lr from")
            for sd in EXTRA_SEEDS:
                items.append(dict(opts, lr=sel[0], seed=sd))
    elif name == "mnist_circuit":
        common = dict(dataset="mnist", head="quantum", mlp_hidden=None, epochs=50, gradient_clip_norm=1.0)
        for n in [500, 1000]:
            for variant in [dict(encoding="half"), dict(encoding="half_sym"), dict(no_entangle=True)]:
                for lr in [3e-3, 1e-2, 3e-2]:
                    for sd in SEEDS:
                        items.append(dict(common, **variant, train_subset_size=n, lr=lr, seed=sd))
    else:
        raise ValueError(name)
    # One more step for the cells whose validation-selected lr was the grid maximum.
    # A trailing str is the pre_norm; a trailing dict holds other run options (e.g. the encoding).
    for ds, n, head, hid, lr, *extra in EDGE_EXTENSION.get(name, []):
        opts = extra[0] if extra and isinstance(extra[0], dict) else dict(pre_norm=extra[0] if extra else "none")
        for sd in SEEDS:
            items.append(dict(dataset=ds, train_subset_size=n, head=head, mlp_hidden=hid, lr=lr, seed=sd,
                              epochs=items[0]["epochs"], gradient_clip_norm=items[0]["gradient_clip_norm"], **opts))
    return items


def _slug(it: dict) -> str:
    head = it["head"] + (f"{it['mlp_hidden']}" if it["head"] == "mlp" else "")
    head += "_bn" if it.get("pre_norm", "none") == "bn" else ""
    head += "_frozen" if it.get("freeze_circuit") else ""
    head += f"_{it['encoding']}" if it.get("encoding", "pi_tanh") != "pi_tanh" else ""
    head += "_noent" if it.get("no_entangle") else ""
    head += "_entlast" if it.get("entangle_last_only") else ""
    head += f"_L{it['n_var_layers']}" if it.get("n_var_layers", 4) != 4 else ""
    head += f"_{it['entangle_pattern']}" if it.get("entangle_pattern") else ""
    head += f"_init{it['vqc_init_scale']:.3g}" if it.get("vqc_init_scale", 0.1) != 0.1 else ""
    head += f"_ep{it['epochs']}" if it["epochs"] != DEFAULT_EPOCHS[it["dataset"]] and not it.get("backbone") else ""
    head += f"_{it['backbone']}" if it.get("backbone") else ""
    mult = it.get("lr_backbone_multiplier", DEFAULT_BB_MULT)
    head += f"_bbm{mult:g}" if mult != DEFAULT_BB_MULT else ""
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
    ap.add_argument("--grid", choices=["mnist_search", "cifar_val", "mnist_bn", "cifar_bn", "cifar_mech", "cifar_circuit",
                                         "cifar_entangle", "cifar_classical", "cifar_cosprod",
                                         "cifar_pairs", "cifar_seeds", "mnist_circuit", "cifar_support",
                                         "cifar_long", "mnist_order", "cifar_sg", "cifar_pretrained",
                                         "cifar_pretrained2", "cifar_long2", "night_seeds",
                                         "cifar_pretrained_10k", "cifar_pretrained_edge",
                                         "cifar_pretrained_1k", "cifar_pretrained_1k_lr",
                                         "pretrained2_seeds", "cifar_pretrained_edge2",
                                         "cifar_pretrained2_edge", "cifar_pretrained_10k_lr",
                                         "cifar_pretrained_10k_edge", "cifar_init",
                                         "cifar_pretrained_zero", "cifar_pretrained_budget"], required=True)
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--n_workers", type=int, default=4)
    ap.add_argument("--threads_per_worker", type=int, default=2)
    ap.add_argument("--export_root", default=str(Path("results") / "protocol_v2"))
    ap.add_argument("--export_only", action="store_true", help="Only export finished runs (no training).")
    a = ap.parse_args()
    root = a.output_root or str(Path("runs") / "protocol_v2" / a.grid)
    if a.export_only:
        n = _export(Path(root), Path(a.export_root) / a.grid)
        print(f"exported {n} finished runs to {Path(a.export_root) / a.grid}", flush=True)
        return
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
