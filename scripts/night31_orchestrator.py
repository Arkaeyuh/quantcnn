#!/usr/bin/env python3
"""Queue for the night of 2026-10-31 (predictions in predictions/prospective_predictions.md).

Waits for the cifar_escgate chain to finish, then trains ``night31`` (the escgate replication on seeds 5-9 and
seeds 5-9 of re-uploading and MobileNetV3-Small at the selected lr), then probes and exports. Same scheme as
``night29_orchestrator.py``. From repository root, detached::

    PYTHONPATH=. nohup caffeinate -dimsu .venv/bin/python -u scripts/night31_orchestrator.py \
        --soft "2026-10-31 11:30" --hard "2026-10-31 12:00" > runs/protocol_v2/night31_orchestrator.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path

from scripts.night26_orchestrator import EXPORT, GRID, _probe
from scripts.night_orchestrator import log, run

GRIDS = ["cifar_escgate_rep", "cifar_reup_seeds2", "cifar_mnv3_seeds2"]
STAGES = [
    ("night31 training (75 runs)", 170, GRID + ["night31"], "night31.log"),
    *[_probe(g, "linear") for g in GRIDS],
    *[_probe(g, "mlp") for g in GRIDS[1:]],
]
CHAIN_OUT = Path("runs/protocol_v2/escgate_chain.out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft", required=True, help="no stage starts that would end after this (YYYY-MM-DD HH:MM)")
    ap.add_argument("--hard", required=True, help="a stage still running at this time is killed")
    a = ap.parse_args()
    soft, hard = (dt.datetime.strptime(t, "%Y-%m-%d %H:%M") for t in (a.soft, a.hard))
    os.environ["PYTHONPATH"] = "."
    log("waiting for the cifar_escgate chain")
    while not (CHAIN_OUT.exists() and "CHAIN_DONE" in CHAIN_OUT.read_text()):
        time.sleep(60)
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
            log(f"export {grid}: {run(EXPORT + [grid], 'night31_export.log', hard + dt.timedelta(minutes=4))}")
    log("queue done")


if __name__ == "__main__":
    main()
