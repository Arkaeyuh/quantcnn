#!/usr/bin/env python3
"""Queue for the night of 2026-10-29 (predictions in predictions/prospective_predictions.md).

SVHN lr extension and seeds 3-9, re-uploading on CIFAR-10, and a MobileNetV3-Small backbone. The three
grids share each stage so the six workers stay busy: tuning (``night29a``), one edge step
(``night29b``), seeds at the selected lr (``night29c``), then probes and exports. Same scheme as
``svhn_orchestrator.py``. From repository root, detached::

    PYTHONPATH=. nohup caffeinate -dimsu .venv/bin/python -u scripts/night29_orchestrator.py \
        --soft "2026-10-30 08:15" --hard "2026-10-30 08:30" > runs/protocol_v2/night29_orchestrator.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

from scripts.night26_orchestrator import EXPORT, GRID, PROBE, PY, _probe
from scripts.night_orchestrator import log, run

SVHN_PROBE = [PY, "-u", "-m", "scripts.probe_backbones", "--heads", "all", "--datasets", "svhn", "--all_lrs"]
CIFAR_GRIDS = ["cifar_reup", "cifar_reup_seeds", "cifar_mnv3", "cifar_mnv3_seeds"]
STAGES = [
    ("night29 tuning (102 runs)", 270, GRID + ["night29a"], "night29a.log"),
    ("night29 edge steps", 100, GRID + ["night29b"], "night29b.log"),
    ("night29 seeds", 180, GRID + ["night29c"], "night29c.log"),
    *[_probe(g, "linear") for g in CIFAR_GRIDS],
    *[("svhn all-lr linear probe", 25, SVHN_PROBE + ["--grids", g, "--out", f"results/{g}_all_probe.csv"],
       f"{g}_all_probe.log") for g in ("svhn", "svhn_seeds")],
    *[_probe(g, "mlp") for g in CIFAR_GRIDS],
    *[("svhn all-lr mlp probe", 30, SVHN_PROBE + ["--probe", "mlp", "--grids", g,
                                                  "--out", f"results/{g}_all_mlp_probe.csv"],
       f"{g}_all_mlp_probe.log") for g in ("svhn", "svhn_seeds")],
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
    for grid in ["svhn", "svhn_seeds"] + CIFAR_GRIDS:
        if os.path.isdir(f"runs/protocol_v2/{grid}"):
            log(f"export {grid}: {run(EXPORT + [grid], 'night29_export.log', hard + dt.timedelta(minutes=4))}")
    log("queue done")


if __name__ == "__main__":
    main()
