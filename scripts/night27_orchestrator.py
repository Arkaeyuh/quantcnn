#!/usr/bin/env python3
"""Follow-on queue for the night of 2026-10-27/28: escape-only straight-through and training dynamics.

Waits for the night26 orchestrator (pid given on the command line) to exit, then runs the
night27 grids and their probes under its own deadlines (the user leaves at 09:15). Predictions:
``predictions/prospective_predictions.md`` (2026-10-27 23:20). From repository root::

    PYTHONPATH=. .venv/bin/python -u scripts/night27_orchestrator.py <night26 pid>
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time

from scripts.night26_orchestrator import EXPORT, GRID, _probe
from scripts.night_orchestrator import log, run

SOFT_DEADLINE = dt.datetime(2026, 10, 28, 8, 30)
HARD_DEADLINE = dt.datetime(2026, 10, 28, 8, 45)
GRIDS = ["cifar_esc", "cifar_dyn"]
STAGES = [
    ("night27 queue (34 runs)", 110, GRID + ["night27"], "night27.log"),
    _probe("cifar_esc", "linear"),
    _probe("cifar_dyn", "linear"),
    _probe("cifar_esc", "mlp"),
]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    os.environ["PYTHONPATH"] = "."
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        log(f"waiting for night26 orchestrator (pid {pid})")
        while _alive(pid) and dt.datetime.now() < SOFT_DEADLINE:
            time.sleep(30)
    log(f"queue start; soft deadline {SOFT_DEADLINE:%H:%M}, hard deadline {HARD_DEADLINE:%H:%M}")
    for name, est, cmd, logfile in STAGES:
        end = dt.datetime.now() + dt.timedelta(minutes=est)
        if end > SOFT_DEADLINE:
            log(f"skip '{name}' (est {est} min would end {end:%H:%M})")
            continue
        log(f"start '{name}' (est {est} min)")
        status = run(cmd, logfile, HARD_DEADLINE)
        log(f"end '{name}': {status}")
        if status.startswith("killed"):
            break
    for grid in GRIDS:
        if os.path.isdir(f"runs/protocol_v2/{grid}"):
            log(f"export {grid}: {run(EXPORT + [grid], 'final_export27.log', HARD_DEADLINE + dt.timedelta(minutes=4))}")
    log("queue done")


if __name__ == "__main__":
    main()
