#!/usr/bin/env python3
"""Deadline-aware overnight queue (2026-10-26/27): mechanism and generality tests.

Same scheme as ``night_orchestrator.py``: stages run in priority order, a stage starts only if its
estimated duration ends before SOFT_DEADLINE, and a stage still running at HARD_DEADLINE is killed
with its process group. Predictions: ``predictions/prospective_predictions.md`` (night of
2026-10-26). From repository root::

    PYTHONPATH=. .venv/bin/python -u scripts/night26_orchestrator.py
"""

from __future__ import annotations

import datetime as dt
import os

from scripts.night_orchestrator import log, run

SOFT_DEADLINE = dt.datetime(2026, 10, 27, 6, 40)
HARD_DEADLINE = dt.datetime(2026, 10, 27, 6, 55)
PY = ".venv/bin/python"
GRID = [PY, "-u", "-m", "scripts.run_protocol_v2", "--n_workers", "6", "--threads_per_worker", "2", "--grid"]
PROBE = [PY, "-u", "-m", "scripts.probe_backbones", "--heads", "all", "--datasets", "cifar10"]
EXPORT = [PY, "-u", "-m", "scripts.run_protocol_v2", "--export_only", "--grid"]
GRIDS = ["cifar_st", "cifar_cz", "cifar_readout", "cifar_lin", "cifar_satup", "night26_seeds"]


def _probe(grid: str, kind: str) -> tuple:
    tag = grid.removeprefix("cifar_")
    out = f"results/{tag}_probe.csv" if kind == "linear" else f"results/{tag}_mlp_probe.csv"
    return (f"{grid} {kind} probe", 10 if kind == "linear" else 15,
            PROBE + ["--probe", kind, "--grids", grid, "--out", out], f"{grid}_{kind}_probe.log")


# (name, estimated minutes, command, log file)
STAGES = [
    ("night26 queue (98 runs)", 330, GRID + ["night26"], "night26.log"),
    ("night26 seeds 3-4", 60, GRID + ["night26_seeds"], "night26_seeds.log"),
    *[_probe(g, "linear") for g in GRIDS],
    *[_probe(g, "mlp") for g in GRIDS],
]


def main():
    os.environ["PYTHONPATH"] = "."
    log(f"queue start; soft deadline {SOFT_DEADLINE:%H:%M}, hard deadline {HARD_DEADLINE:%H:%M}")
    stopped = False
    for name, est, cmd, logfile in STAGES:
        end = dt.datetime.now() + dt.timedelta(minutes=est)
        if end > SOFT_DEADLINE:
            log(f"skip '{name}' (est {est} min would end {end:%H:%M})")
            continue
        log(f"start '{name}' (est {est} min)")
        status = run(cmd, logfile, HARD_DEADLINE)
        log(f"end '{name}': {status}")
        if status.startswith("killed"):
            stopped = True
            break
    for grid in GRIDS:
        if os.path.isdir(f"runs/protocol_v2/{grid}"):
            log(f"export {grid}: {run(EXPORT + [grid], 'final_export.log', HARD_DEADLINE + dt.timedelta(minutes=4))}")
    log("queue done" + (" (stopped at hard deadline)" if stopped else ""))


if __name__ == "__main__":
    main()
