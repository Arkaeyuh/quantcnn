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
import json
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
DEFAULT_EPOCHS = {"mnist": 50, "cifar10": 80, "svhn": 80}
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


# Second circuit family: CNOT layouts outside the ring family, by zero-rotation product order.
FAMILY = ["star", "chain", "star_in_last", "ladder2_last", "brick", "chain_last"]
FAMILY_LRS = [3e-3, 1e-2, 3e-2]


def _family_selected() -> dict[str, tuple[float, list[float]]]:
    """pattern -> (selected lr, lrs tried) from the finished cifar_family runs."""
    from scripts.probe_backbones import _selected_runs

    sel = _selected_runs(Path("runs") / "protocol_v2", ["cifar_family"])
    tried = {}
    for fin in (Path("runs") / "protocol_v2" / "cifar_family").glob("*/final.json"):
        meta = json.loads((fin.parent / "meta.json").read_text())
        tried.setdefault(meta["entangle"], set()).add(meta["lr"])
    return {p: (sel[(4000, f"quantum+{p}")][0], sorted(tried[p])) for p in FAMILY if (4000, f"quantum+{p}") in sel}


# Encoding arms. half: ring-family and second-family layouts spanning orders 1-5.25 under the
# monotone ``half`` encoding (the default ring with ``half`` is already in cifar_circuit). phase:
# H then RZ, where the product identity fails; layouts chosen so that the effective order at
# initialization and the R_y product order disagree.
_LAYOUT_OPTS = {"noent": dict(no_entangle=True), "pairs_last": dict(entangle_pattern="pairs_last"),
                "ladder2_last": dict(entangle_pattern="ladder2_last"), "chain_last": dict(entangle_pattern="chain_last"),
                "entlast": dict(entangle_last_only=True), "chain": dict(entangle_pattern="chain"),
                "star": dict(entangle_pattern="star"), "default": {}}
ENC_ARMS = {"half": ["noent", "pairs_last", "ladder2_last", "chain_last", "entlast"],
            "phase": ["noent", "entlast", "chain_last", "chain", "default", "star"]}
HALF = {k: _LAYOUT_OPTS[k] for k in ENC_ARMS["half"]}


def _enc_selected(encoding: str) -> dict[str, tuple[dict, float, list[float]]]:
    """layout -> (run options, selected lr, lrs tried) from the finished runs of an encoding arm
    (for half, also the default ring from cifar_circuit), selection on mean val acc."""
    root = Path("runs") / "protocol_v2"
    by_layout = {}
    for grid in [f"cifar_{encoding}", f"cifar_{encoding}_ext", "cifar_circuit"]:
        for fin in (root / grid).glob("*/final.json"):
            meta = json.loads((fin.parent / "meta.json").read_text())
            if meta.get("encoding") != encoding or int(meta["train_subset_size"]) != 4000:
                continue
            ent = meta.get("entangle", True)
            name = "noent" if ent is False else "entlast" if ent == "last" else "default" if ent is True else ent
            by_layout.setdefault(name, {}).setdefault(meta["lr"], []).append(json.loads(fin.read_text())["val_acc"])
    return {name: (_LAYOUT_OPTS[name], max(lrs, key=lambda lr: sum(lrs[lr]) / len(lrs[lr])), sorted(lrs))
            for name, lrs in by_layout.items() if name in _LAYOUT_OPTS}


def _half_selected() -> dict[str, tuple[dict, float, list[float]]]:
    return _enc_selected("half")


# Attenuation test: product heads whose backward floors |product of the other factors| (forward
# unchanged), on the default and last-ring supports; the coupled heads (floor 0) are in cifar_cosprod.
FLOOR_HEADS = ["cosprod_fl03", "cosprod_fl1", "cosprod_last_fl03", "cosprod_last_fl1"]
FLOOR_LRS = [1e-3, 3e-3, 1e-2, 3e-2]


def _floor_selected() -> dict[str, tuple[float, list[float]]]:
    """head -> (selected lr, lrs tried) from the finished cifar_floor(_ext) runs."""
    from scripts.probe_backbones import _selected_runs

    root = Path("runs") / "protocol_v2"
    sel = _selected_runs(root, ["cifar_floor", "cifar_floor_ext"])
    tried = {}
    for g in ["cifar_floor", "cifar_floor_ext"]:
        for fin in (root / g).glob("*/final.json"):
            meta = json.loads((fin.parent / "meta.json").read_text())
            tried.setdefault(meta["head"], set()).add(meta["lr"])
    return {h: (sel[(4000, h)][0], sorted(tried[h])) for h in FLOOR_HEADS if (4000, h) in sel}


# Width replication: 12-dimensional bottleneck, 12 qubits (zero-rotation orders 1, 3.92, 7.33).
Q12 = {"linear": (dict(head="linear"), [1e-3, 3e-3, 1e-2, 3e-2]),
       "noent": (dict(head="quantum", no_entangle=True), [3e-3, 1e-2, 3e-2]),
       "default": (dict(head="quantum"), [3e-3, 1e-2, 3e-2]),
       "entlast": (dict(head="quantum", entangle_last_only=True), [3e-3, 1e-2, 3e-2])}


def _q12_selected() -> dict[str, tuple[float, list[float]]]:
    """Q12 key -> (selected lr, lrs tried) from the finished cifar_q12(_ext) runs."""
    from scripts.probe_backbones import _head_key, _selected_runs

    root = Path("runs") / "protocol_v2"
    sel = _selected_runs(root, ["cifar_q12", "cifar_q12_ext"])
    out = {}
    for key, (opts, _) in Q12.items():
        hk = _head_key(dict(opts, mlp_hidden=None, epochs=80, dataset="cifar10", lr=0,
                            entangle=False if opts.get("no_entangle") else "last" if opts.get("entangle_last_only") else True))
        tried = sorted({json.loads((f.parent / "meta.json").read_text())["lr"]
                        for g in ["cifar_q12", "cifar_q12_ext"] for f in (root / g).glob("*/final.json")
                        if _head_key(json.loads((f.parent / "meta.json").read_text())) == hk})
        if (4000, hk) in sel:
            out[key] = (sel[(4000, hk)][0], tried)
    return out


# Mechanism test: product heads (order 1, 2.5, 5.25) with a bottleneck regularizer, lr fixed at the
# unregularized heads' selected 1e-2; the penalty weight is selected on validation per (head, reg).
MECH2_HEADS = ["cos", "cosprod", "cosprod_last"]
MECH2_REGS = {"sat": [3.0, 10.0], "decor": [1.0, 3.0]}


def _mech2_selected() -> dict[tuple[str, str], float]:
    """(head, reg) -> reg weight with the best mean validation accuracy over seeds 0-2."""
    root = Path("runs") / "protocol_v2" / "cifar_mech2"
    vals = {}
    for fin in root.glob("*/final.json"):
        meta = json.loads((fin.parent / "meta.json").read_text())
        vals.setdefault((meta["head"], meta["bottleneck_reg"], meta["reg_weight"]), []).append(
            json.loads(fin.read_text())["val_acc"])
    best = {}
    for (head, reg, w), v in vals.items():
        if (head, reg) not in best or sum(v) / len(v) > best[(head, reg)][1]:
            best[(head, reg)] = (w, sum(v) / len(v))
    return {k: w for k, (w, _) in best.items()}


# Mechanism confirmation on circuits: 8-qubit VQC heads (order 1, 2.5, 5.25) with the
# anti-saturation penalty, lr fixed at their selected 1e-2, weight selected on validation per layout.
MECH3_LAYOUTS = {"noent": dict(head="quantum", no_entangle=True), "default": dict(head="quantum"),
                 "entlast": dict(head="quantum", entangle_last_only=True)}
MECH3_WEIGHTS = [3.0, 10.0]


def _mech3_selected() -> dict[str, float]:
    """layout -> sat weight with the best mean validation accuracy over seeds 0-2."""
    vals = {}
    for fin in (Path("runs") / "protocol_v2" / "cifar_mech3").glob("*/final.json"):
        meta = json.loads((fin.parent / "meta.json").read_text())
        ent = meta.get("entangle", True)
        name = "noent" if ent is False else "entlast" if ent == "last" else "default"
        vals.setdefault((name, meta["reg_weight"]), []).append(json.loads(fin.read_text())["val_acc"])
    best = {}
    for (name, w), v in vals.items():
        if name not in best or sum(v) / len(v) > best[name][1]:
            best[name] = (w, sum(v) / len(v))
    return {k: w for k, (w, _) in best.items()}


# Last-ring replication of the saturation test on fresh seeds 5-9, both arms, penalty fixed at 10.
RETEST_SEEDS = [5, 6, 7, 8, 9]
RETEST_CELLS = [dict(head="quantum", entangle_last_only=True), dict(head="cosprod_last")]


# Mechanism and generality tests, 2026-10-26 (predictions in predictions/prospective_predictions.md).
ALL_SEEDS = [0, 1, 2, 3, 4]
NIGHT26_LRS = [3e-3, 1e-2, 3e-2]
# Straight-through tanh (forward identical, sech^2 removed backward), lr fixed at the selected 1e-2.
ST_CELLS = [dict(head="quantum", encoding="pi_tanh_st"),
            dict(head="quantum", encoding="pi_tanh_st", entangle_last_only=True),
            dict(head="quantum", encoding="pi_tanh_st", no_entangle=True),
            dict(head="cosprod_st"), dict(head="cosprod_last_st")]
# New architectures, each tuned over NIGHT26_LRS on seeds 0-2.
NIGHT26_ARCH = {
    "cifar_cz": [dict(head="quantum", entangle_pattern="cz")],
    "cifar_lin": [dict(head="quantum", encoding="lin", **lay) for lay in
                  (dict(no_entangle=True), dict(entangle_pattern="pairs_last"), {}, dict(entangle_last_only=True))],
    "cifar_readout": [dict(head="quantum", no_entangle=True, readout=r) for r in ("zz", "zzz")],
}
SATUP_WEIGHT = 10.0
# SVHN (2026-10-29, predictions in predictions/prospective_predictions.md): the headline heads at 4,000 images.
_S4000 = dict(_C4000, dataset="svhn")
LR_STEPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SVHN_CELLS = [(dict(head="linear"), [1e-2, 3e-2, 1e-1]),
              (dict(head="quantum", no_entangle=True), NIGHT26_LRS),
              (dict(head="quantum", entangle_pattern="pairs"), NIGHT26_LRS),
              (dict(head="quantum"), NIGHT26_LRS),
              (dict(head="quantum", entangle_last_only=True), NIGHT26_LRS),
              (dict(head="cosprod"), NIGHT26_LRS)]


def _val_means(grid: str, base: dict, opts: dict) -> dict[float, float]:
    """lr -> mean validation accuracy over seeds 0-2, for every lr of this cell of ``grid`` with all three seeds done."""
    out = {}
    for lr in LR_STEPS:
        v = []
        for sd in SEEDS:
            fin = Path("runs") / "protocol_v2" / grid / _slug(dict(base, **opts, lr=lr, seed=sd)) / "final.json"
            if fin.exists():
                v.append(json.loads(fin.read_text())["val_acc"])
        if len(v) == len(SEEDS):
            out[lr] = sum(v) / len(v)
    return out


def _svhn_val_means(opts: dict) -> dict[float, float]:
    return _val_means("svhn", _S4000, opts)


def _edge_steps(grid: str, base: dict, cells: list[dict]) -> list[dict]:
    """One half-decade step beyond the lrs run (seeds 0-2, into ``grid``) for every cell whose best lr is at an edge."""
    items = []
    for opts in cells:
        means = _val_means(grid, base, opts)
        if not means:
            continue
        best, tried = max(means, key=means.get), sorted(means)
        i = LR_STEPS.index(best)
        for lr in (LR_STEPS[i - 1] if best == tried[0] and i > 0 else None,
                   LR_STEPS[i + 1] if best == tried[-1] and i + 1 < len(LR_STEPS) else None):
            if lr is not None:
                items += [dict(base, **opts, lr=lr, seed=sd, _grid=grid) for sd in SEEDS]
    return items


def _selected_seeds(grid: str, base: dict, cells: list[dict], seeds: list[int], dest: str) -> list[dict]:
    """``seeds`` at each cell's selected lr (best mean validation accuracy over seeds 0-2 in ``grid``), into ``dest``."""
    items = []
    for opts in cells:
        means = _val_means(grid, base, opts)
        if means:
            items += [dict(base, **opts, lr=max(means, key=means.get), seed=sd, _grid=dest) for sd in seeds]
    return items


# Night of 2026-10-29 (predictions in predictions/prospective_predictions.md).
# SVHN: lrs below the first grid for every head, then one more edge step, then seeds 3-9 at the selected lr.
SVHN_EXT = {"linear": [3e-3], "noent": [3e-4], "pairs": [3e-4], "default": [3e-4], "entlast": [1e-3, 3e-4],
            "cosprod": [3e-4]}
# Re-uploading ("strong": the encoding before every variational layer) on the four ring-family layouts.
_LAYOUTS = [dict(no_entangle=True), dict(entangle_pattern="pairs_last"), {}, dict(entangle_last_only=True)]
REUP_CELLS = [dict(head="quantum", reupload_mode="strong", **lay) for lay in _LAYOUTS]
# Second pretrained backbone: ImageNet MobileNetV3-Small, the cifar_pretrained heads plus pairs.
_MNV3 = dict(_PRETRAINED, backbone="mobilenetv3s_pt")
MNV3_CELLS = [dict(head="linear"), dict(head="quantum", no_entangle=True), dict(head="quantum", entangle_pattern="pairs"),
              dict(head="quantum"), dict(head="quantum", entangle_last_only=True)]
MNV3_LRS = [1e-3, 3e-3, 1e-2]
# Escape-only straight-through (2026-10-27): replaces the diverged pi_tanh_st circuit cells; same design.
ESC_CELLS = [dict(head="quantum", encoding="pi_tanh_esc"),
             dict(head="quantum", encoding="pi_tanh_esc", entangle_last_only=True),
             dict(head="quantum", encoding="pi_tanh_esc", no_entangle=True),
             dict(head="cosprod_esc"), dict(head="cosprod_last_esc")]


def _night26_selected() -> list[tuple[dict, float]]:
    """(run options, lr with the best mean validation accuracy over seeds 0-2) per NIGHT26_ARCH cell."""
    out = []
    for grid, cells in NIGHT26_ARCH.items():
        for opts in cells:
            vals = {}
            for lr in NIGHT26_LRS:
                v = []
                for sd in SEEDS:
                    fin = Path("runs") / "protocol_v2" / grid / _slug(dict(_C4000, **opts, lr=lr, seed=sd)) / "final.json"
                    if fin.exists():
                        v.append(json.loads(fin.read_text())["val_acc"])
                if len(v) == len(SEEDS):
                    vals[lr] = sum(v) / len(v)
            if vals:
                out.append((opts, max(vals, key=vals.get)))
    return out


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
    elif name == "cifar_pretrained_budget_lr":
        # Matched-update budgets with the head lr re-tuned: one half-decade step each way.
        for n, epochs, lrs in [(1000, 120, dict(linear=[3e-4, 3e-3], quantum=[3e-4, 3e-3], noent=[3e-4, 3e-3])),
                               (10000, 12, dict(linear=[1e-4, 1e-3], quantum=[1e-4, 1e-3], noent=[3e-4, 3e-3]))]:
            for key, variant in [("linear", dict(head="linear")), ("quantum", dict(head="quantum")),
                                 ("noent", dict(head="quantum", no_entangle=True))]:
                for lr in lrs[key]:
                    for sd in SEEDS:
                        items.append(dict(_PRETRAINED, **variant, train_subset_size=n, epochs=epochs, lr=lr, seed=sd))
    elif name == "cifar_family":
        for pattern in FAMILY:
            for lr in FAMILY_LRS:
                for sd in SEEDS:
                    items.append(dict(_C4000, head="quantum", entangle_pattern=pattern, lr=lr, seed=sd))
    elif name == "cifar_family_ext":
        # One step beyond the grid where the selected lr sits on its boundary.
        ladder = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
        for pattern, (lr, tried) in _family_selected().items():
            i = min(range(len(ladder)), key=lambda j: abs(ladder[j] - lr))
            step = ladder[i - 1] if lr == min(tried) else ladder[i + 1] if lr == max(tried) else None
            if step is not None and step not in tried:
                for sd in SEEDS:
                    items.append(dict(_C4000, head="quantum", entangle_pattern=pattern, lr=step, seed=sd))
    elif name == "cifar_family_seeds":
        for pattern, (lr, _) in _family_selected().items():
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, head="quantum", entangle_pattern=pattern, lr=lr, seed=sd))
    elif name in ("cifar_half", "cifar_phase"):
        enc = name.split("_")[1]
        for key in ENC_ARMS[enc]:
            for lr in FAMILY_LRS:
                for sd in SEEDS:
                    items.append(dict(_C4000, head="quantum", encoding=enc, **_LAYOUT_OPTS[key], lr=lr, seed=sd))
    elif name in ("cifar_half_ext", "cifar_phase_ext"):
        enc = name.split("_")[1]
        ladder = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
        for opts, lr, tried in _enc_selected(enc).values():
            i = min(range(len(ladder)), key=lambda j: abs(ladder[j] - lr))
            step = ladder[i - 1] if lr == min(tried) else ladder[i + 1] if lr == max(tried) else None
            if step is not None and step not in tried:
                for sd in SEEDS:
                    items.append(dict(_C4000, head="quantum", encoding=enc, **opts, lr=step, seed=sd))
    elif name in ("cifar_half_seeds", "cifar_phase_seeds"):
        enc = name.split("_")[1]
        for opts, lr, _ in _enc_selected(enc).values():
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, head="quantum", encoding=enc, **opts, lr=lr, seed=sd))
    elif name == "cifar_floor":
        for head in FLOOR_HEADS:
            for lr in FLOOR_LRS:
                for sd in SEEDS:
                    items.append(dict(_C4000, head=head, lr=lr, seed=sd))
    elif name == "cifar_floor_ext":
        ladder = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
        for head, (lr, tried) in _floor_selected().items():
            i = ladder.index(min(ladder, key=lambda v: abs(v - lr)))
            step = ladder[i - 1] if lr == min(tried) else ladder[i + 1] if lr == max(tried) else None
            if step is not None and step not in tried:
                for sd in SEEDS:
                    items.append(dict(_C4000, head=head, lr=step, seed=sd))
    elif name == "cifar_floor_seeds":
        for head, (lr, _) in _floor_selected().items():
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, head=head, lr=lr, seed=sd))
    elif name == "cifar_q12":
        for opts, lrs in Q12.values():
            for lr in lrs:
                for sd in SEEDS:
                    items.append(dict(_C4000, **opts, bottleneck_dim=12, lr=lr, seed=sd))
    elif name == "cifar_q12_ext":
        ladder = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
        for key, (lr, tried) in _q12_selected().items():
            i = ladder.index(min(ladder, key=lambda v: abs(v - lr)))
            step = ladder[i - 1] if lr == min(tried) else ladder[i + 1] if lr == max(tried) else None
            if step is not None and step not in tried:
                for sd in SEEDS:
                    items.append(dict(_C4000, **Q12[key][0], bottleneck_dim=12, lr=step, seed=sd))
    elif name == "cifar_q12_seeds":
        for key, (lr, _) in _q12_selected().items():
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, **Q12[key][0], bottleneck_dim=12, lr=lr, seed=sd))
    elif name == "cifar_mech2":
        for head in MECH2_HEADS:
            for reg, weights in MECH2_REGS.items():
                for w in weights:
                    for sd in SEEDS:
                        items.append(dict(_C4000, head=head, bottleneck_reg=reg, reg_weight=w, lr=1e-2, seed=sd))
    elif name == "cifar_mech2_seeds":
        for (head, reg), w in sorted(_mech2_selected().items()):
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, head=head, bottleneck_reg=reg, reg_weight=w, lr=1e-2, seed=sd))
    elif name == "cifar_mech3":
        for opts in MECH3_LAYOUTS.values():
            for w in MECH3_WEIGHTS:
                for sd in SEEDS:
                    items.append(dict(_C4000, **opts, bottleneck_reg="sat", reg_weight=w, lr=1e-2, seed=sd))
    elif name == "cifar_mech3_seeds":
        for layout, w in sorted(_mech3_selected().items()):
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, **MECH3_LAYOUTS[layout], bottleneck_reg="sat", reg_weight=w, lr=1e-2, seed=sd))
    elif name == "cifar_lastring_retest":
        for opts in RETEST_CELLS:
            for reg in ({}, dict(bottleneck_reg="sat", reg_weight=10.0)):
                for sd in RETEST_SEEDS:
                    items.append(dict(_C4000, **opts, **reg, lr=1e-2, seed=sd))
    elif name == "cifar_st":
        for opts in ST_CELLS:
            for sd in ALL_SEEDS:
                items.append(dict(_C4000, **opts, lr=1e-2, seed=sd))
    elif name == "cifar_esc":
        for opts in ESC_CELLS:
            for sd in ALL_SEEDS:
                items.append(dict(_C4000, **opts, lr=1e-2, seed=sd))
    elif name == "cifar_escgate":
        # Saturation-gated escape-only control (2026-10-30 predictions): the inward boost only on saturated
        # (|tanh z| > 0.9) or only on unsaturated coordinates, three ring-family layouts, seeds 0-4.
        for enc in ("pi_tanh_escsat", "pi_tanh_escmid"):
            for lay in (dict(no_entangle=True), {}, dict(entangle_last_only=True)):
                for sd in ALL_SEEDS:
                    items.append(dict(_C4000, head="quantum", encoding=enc, **lay, lr=1e-2, seed=sd))
    elif name == "cifar_escgate_rep":
        # Fresh-seed replication of cifar_escgate (2026-10-31 predictions), same design on seeds 5-9.
        for enc in ("pi_tanh_escsat", "pi_tanh_escmid"):
            for lay in (dict(no_entangle=True), {}, dict(entangle_last_only=True)):
                for sd in RETEST_SEEDS:
                    items.append(dict(_C4000, head="quantum", encoding=enc, **lay, lr=1e-2, seed=sd))
    elif name == "night31":
        # Night of 2026-10-31: escgate replication, then re-uploading and MobileNet seeds 5-9 at the selected lr.
        items += [dict(it, _grid="cifar_escgate_rep") for it in _grid("cifar_escgate_rep")]
        items += _selected_seeds("cifar_reup", _C4000, REUP_CELLS, RETEST_SEEDS, "cifar_reup_seeds2")
        items += _selected_seeds("cifar_mnv3", _MNV3, MNV3_CELLS, RETEST_SEEDS, "cifar_mnv3_seeds2")
    elif name == "cifar_dyn":
        # Training dynamics (plan A5): the three ring-family baselines, checkpoints every 5 epochs.
        for opts in (dict(no_entangle=True), {}, dict(entangle_last_only=True)):
            for sd in SEEDS:
                items.append(dict(_C4000, head="quantum", **opts, lr=1e-2, seed=sd, save_every=5))
    elif name == "cifar_esc_rep":
        # Fresh-seed replication of the escape-only test (2026-10-28): both arms on seeds 5-9, except the
        # unregularized last ring, which cifar_lastring_retest already ran on these seeds.
        for opts in (dict(head="quantum"), dict(head="quantum", encoding="pi_tanh_esc"),
                     dict(head="quantum", no_entangle=True), dict(head="quantum", encoding="pi_tanh_esc", no_entangle=True),
                     dict(head="quantum", encoding="pi_tanh_esc", entangle_last_only=True)):
            for sd in RETEST_SEEDS:
                items.append(dict(_C4000, **opts, lr=1e-2, seed=sd))
    elif name == "cifar_dyn2":
        # cifar_dyn with a checkpoint after every one of the first 10 epochs (same seeds, deterministic).
        for opts in (dict(no_entangle=True), {}, dict(entangle_last_only=True)):
            for sd in SEEDS:
                items.append(dict(_C4000, head="quantum", **opts, lr=1e-2, seed=sd, save_every=5, save_dense_until=10))
    elif name == "svhn":
        for opts, lrs in SVHN_CELLS:
            for lr in lrs:
                for sd in SEEDS:
                    items.append(dict(_S4000, **opts, lr=lr, seed=sd))
    elif name == "svhn_edge":
        # One half-decade step beyond the grid for every cell whose best lr (seeds 0-2) is at an edge of the lrs run.
        for opts, _ in SVHN_CELLS:
            means = _svhn_val_means(opts)
            if not means:
                continue
            best, tried = max(means, key=means.get), sorted(means)
            i = LR_STEPS.index(best)
            step = (LR_STEPS[i - 1] if best == tried[0] and i > 0 else None,
                    LR_STEPS[i + 1] if best == tried[-1] and i + 1 < len(LR_STEPS) else None)
            for lr in step:
                if lr is not None:
                    for sd in SEEDS:
                        items.append(dict(_S4000, **opts, lr=lr, seed=sd, _grid="svhn"))
    elif name == "svhn_seeds":
        for opts, _ in SVHN_CELLS:
            means = _svhn_val_means(opts)
            if means:
                for sd in EXTRA_SEEDS:
                    items.append(dict(_S4000, **opts, lr=max(means, key=means.get), seed=sd))
    elif name == "svhn_ext":
        for (opts, _), key in zip(SVHN_CELLS, ["linear", "noent", "pairs", "default", "entlast", "cosprod"]):
            for lr in SVHN_EXT[key]:
                items += [dict(_S4000, **opts, lr=lr, seed=sd, _grid="svhn") for sd in SEEDS]
    elif name == "svhn_collapse":
        # Collapse by order at an lr not run before (2026-10-30, predictions in predictions/prospective_predictions.md).
        items += [dict(_S4000, **opts, lr=5e-3, seed=sd) for opts, _ in SVHN_CELLS[1:5] for sd in range(6)]
    elif name == "cifar_reup":
        items += [dict(_C4000, **opts, lr=lr, seed=sd) for opts in REUP_CELLS for lr in NIGHT26_LRS for sd in SEEDS]
    elif name == "cifar_mnv3":
        items += [dict(_MNV3, **opts, lr=lr, seed=sd) for opts in MNV3_CELLS for lr in MNV3_LRS for sd in SEEDS]
    elif name == "night29a":
        for g in ("svhn_ext", "cifar_reup", "cifar_mnv3"):
            items += [dict(it, _grid=it.get("_grid", g)) for it in _grid(g)]
    elif name == "night29b":
        items += _edge_steps("svhn", _S4000, [o for o, _ in SVHN_CELLS])
        items += _edge_steps("cifar_reup", _C4000, REUP_CELLS)
        items += _edge_steps("cifar_mnv3", _MNV3, MNV3_CELLS)
    elif name == "night29c":
        items += _selected_seeds("svhn", _S4000, [o for o, _ in SVHN_CELLS], list(range(3, 10)), "svhn_seeds")
        items += _selected_seeds("cifar_reup", _C4000, REUP_CELLS, EXTRA_SEEDS, "cifar_reup_seeds")
        items += _selected_seeds("cifar_mnv3", _MNV3, MNV3_CELLS, EXTRA_SEEDS, "cifar_mnv3_seeds")
    elif name == "night28":
        for g in ("cifar_esc_rep", "cifar_dyn2"):
            items += [dict(it, _grid=g) for it in _grid(g)]
    elif name == "night27":
        for g in ("cifar_esc", "cifar_dyn"):
            items += [dict(it, _grid=g) for it in _grid(g)]
    elif name in NIGHT26_ARCH:
        for opts in NIGHT26_ARCH[name]:
            for lr in NIGHT26_LRS:
                for sd in SEEDS:
                    items.append(dict(_C4000, **opts, lr=lr, seed=sd))
    elif name == "cifar_satup":
        for k in (2, 4):
            for sd in ALL_SEEDS:
                items.append(dict(_C4000, head="quantum", no_entangle=True, bottleneck_reg=f"satup{k}",
                                  reg_weight=SATUP_WEIGHT, lr=1e-2, seed=sd))
    elif name == "night26":
        # One priority-ordered queue over the night's grids so no worker idles between small grids;
        # each run still goes to its own grid's directory (``_grid``).
        for g in ("cifar_st", "cifar_cz", "cifar_readout", "cifar_lin", "cifar_satup"):
            items += [dict(it, _grid=g) for it in _grid(g)]
    elif name == "night26_seeds":
        for opts, lr in _night26_selected():
            for sd in EXTRA_SEEDS:
                items.append(dict(_C4000, **opts, lr=lr, seed=sd))
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
    head += f"_ro{it['readout']}" if it.get("readout", "z") != "z" else ""
    head += f"_init{it['vqc_init_scale']:.3g}" if it.get("vqc_init_scale", 0.1) != 0.1 else ""
    head += f"_b{it['bottleneck_dim']}" if it.get("bottleneck_dim", 8) != 8 else ""
    head += f"_{it['bottleneck_reg']}{it['reg_weight']:g}" if it.get("bottleneck_reg", "none") != "none" else ""
    head += f"_ep{it['epochs']}" if it["epochs"] != DEFAULT_EPOCHS[it["dataset"]] and not it.get("backbone") else ""
    head += f"_{it['backbone']}" if it.get("backbone") else ""
    head += f"_ru{it['reupload_mode']}" if it.get("reupload_mode", "none") != "none" else ""
    mult = it.get("lr_backbone_multiplier", DEFAULT_BB_MULT)
    head += f"_bbm{mult:g}" if mult != DEFAULT_BB_MULT else ""
    return f"{it['dataset']}_subset{it['train_subset_size']}_{head}_lr{it['lr']:g}_s{it['seed']}"


def _run(it: dict, root: str, val_size: int, threads: int) -> str:
    torch.set_num_threads(threads)
    it = dict(it)
    if "_grid" in it:
        root = str(Path(root).parent / it.pop("_grid"))
    out = Path(root) / _slug(it)
    if (out / "final.json").exists():
        return f"skip {out.name}"
    args = deepcopy(default_train_args())
    for k, v in it.items():
        setattr(args, k, v)
    args.reupload_mode = it.get("reupload_mode", "none")
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
                                         "cifar_pretrained_zero", "cifar_pretrained_budget",
                                         "cifar_family", "cifar_family_ext", "cifar_family_seeds",
                                         "cifar_pretrained_budget_lr", "cifar_half", "cifar_half_ext",
                                         "cifar_half_seeds", "cifar_phase", "cifar_phase_ext",
                                         "cifar_phase_seeds", "cifar_floor", "cifar_floor_ext",
                                         "cifar_floor_seeds", "cifar_q12", "cifar_q12_ext",
                                         "cifar_q12_seeds", "cifar_mech2", "cifar_mech2_seeds", "cifar_mech3",
                                         "cifar_mech3_seeds", "cifar_lastring_retest", "cifar_st", "cifar_cz",
                                         "cifar_lin", "cifar_readout", "cifar_satup", "night26", "night26_seeds",
                                         "cifar_esc", "cifar_dyn", "night27",
                                         "cifar_esc_rep", "cifar_dyn2", "night28", "cifar_escgate",
                                         "svhn", "svhn_edge", "svhn_seeds",
                                         "svhn_ext", "cifar_reup", "cifar_mnv3", "night29a", "night29b",
                                         "night29c", "cifar_reup_seeds", "cifar_mnv3_seeds", "svhn_collapse", "cifar_escgate_rep", "night31", "cifar_reup_seeds2", "cifar_mnv3_seeds2"], required=True)
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
    for g in sorted({it.get("_grid", a.grid) for it in items}):
        r = Path(root).parent / g if g != a.grid else Path(root)
        n = _export(r, Path(a.export_root) / g)
        print(f"done; exported {n} finished runs to {Path(a.export_root) / g}", flush=True)


if __name__ == "__main__":
    main()
