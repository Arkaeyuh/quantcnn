#!/usr/bin/env python3
"""Queue for the escape-only replication and dense early dynamics (predictions 2026-10-28).

Same scheme as ``night26_orchestrator.py``; deadlines are given on the command line because the
launch time depends on when the laptop is on power. From repository root, detached::

    PYTHONPATH=. nohup caffeinate -i .venv/bin/python -u scripts/night28_orchestrator.py \
        --soft "2026-10-29 06:40" --hard "2026-10-29 06:55" > runs/protocol_v2/night28_orchestrator.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

from scripts.night26_orchestrator import EXPORT, GRID, PY, _probe
from scripts.night_orchestrator import log, run

GRIDS = ["cifar_esc_rep", "cifar_dyn2"]
STAGES = [
    ("night28 queue (34 runs)", 110, GRID + ["night28"], "night28.log"),
    _probe("cifar_esc_rep", "linear"),
    ("dense dynamics probes", 30, [PY, "-u", "-m", "scripts.dynamics_probe", "--root", "runs/protocol_v2/cifar_dyn2",
                                   "--out", "results/dyn2_curves.csv"], "dyn2_curves.log"),
    _probe("cifar_esc_rep", "mlp"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft", required=True, help="no stage starts that would end after this (YYYY-MM-DD HH:MM)")
    ap.add_argument("--hard", required=True, help="a stage still running at this time is killed")
    a = ap.parse_args()
    soft, hard = (dt.datetime.strptime(t, "%Y-%m-%d %H:%M") for t in (a.soft, a.hard))
    os.environ["PYTHONPATH"] = "."
    log(f"queue start; soft deadline {soft:%F %H:%M}, hard deadline {hard:%F %H:%M}")
    for name, est, cmd, logfile in STAGES:
        end = dt.datetime.now() + dt.timedelta(minutes=est)
        if end > soft:
            log(f"skip '{name}' (est {est} min would end {end:%H:%M})")
            continue
        log(f"start '{name}' (est {est} min)")
        status = run(cmd, logfile, hard)
        log(f"end '{name}': {status}")
        if status.startswith("killed"):
            break
    for grid in GRIDS:
        if os.path.isdir(f"runs/protocol_v2/{grid}"):
            log(f"export {grid}: {run(EXPORT + [grid], 'final_export28.log', hard + dt.timedelta(minutes=4))}")
    log("queue done")


if __name__ == "__main__":
    main()
