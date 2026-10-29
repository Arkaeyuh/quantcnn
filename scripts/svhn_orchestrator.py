#!/usr/bin/env python3
"""SVHN queue (predictions 2026-10-29): tune, extend any edge-selected lr, add seeds 3-4, probe, export.

Same scheme as ``night28_orchestrator.py``. From repository root, detached::

    PYTHONPATH=. nohup caffeinate -i .venv/bin/python -u scripts/svhn_orchestrator.py \
        --soft "2026-10-29 16:00" --hard "2026-10-29 16:15" > runs/protocol_v2/svhn_orchestrator.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

from scripts.night26_orchestrator import EXPORT, GRID, PY
from scripts.night_orchestrator import log, run

PROBE = [PY, "-u", "-m", "scripts.probe_backbones", "--heads", "all", "--datasets", "svhn"]
STAGES = [
    ("svhn tuning (54 runs)", 160, GRID + ["svhn"], "svhn.log"),
    ("svhn edge steps", 60, GRID + ["svhn_edge"], "svhn_edge.log"),
    ("svhn seeds 3-4", 45, GRID + ["svhn_seeds"], "svhn_seeds.log"),
    ("svhn linear probe", 10, PROBE + ["--grids", "svhn", "--out", "results/svhn_probe.csv"], "svhn_probe.log"),
    ("svhn seeds linear probe", 8, PROBE + ["--grids", "svhn_seeds", "--out", "results/svhn_seeds_probe.csv"],
     "svhn_seeds_probe.log"),
    ("svhn mlp probe", 15, PROBE + ["--probe", "mlp", "--grids", "svhn", "--out", "results/svhn_mlp_probe.csv"],
     "svhn_mlp_probe.log"),
    ("svhn seeds mlp probe", 10, PROBE + ["--probe", "mlp", "--grids", "svhn_seeds",
                                          "--out", "results/svhn_seeds_mlp_probe.csv"], "svhn_seeds_mlp_probe.log"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft", required=True)
    ap.add_argument("--hard", required=True)
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
    for grid in ("svhn", "svhn_seeds"):
        if os.path.isdir(f"runs/protocol_v2/{grid}"):
            log(f"export {grid}: {run(EXPORT + [grid], 'svhn_export.log', hard + dt.timedelta(minutes=4))}")
    log("queue done")


if __name__ == "__main__":
    main()
