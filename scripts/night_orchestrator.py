#!/usr/bin/env python3
"""Deadline-aware overnight queue (2026-10-18/19).

Runs stages in priority order. A stage starts only if its estimated duration ends before
SOFT_DEADLINE; any stage still running at HARD_DEADLINE is killed (with its process group)
and the queue stops. Finished runs are exported at the end regardless. From repository root::

    PYTHONPATH=. .venv/bin/python -u scripts/night_orchestrator.py
"""

from __future__ import annotations

import datetime as dt
import os
import signal
import subprocess
import time

SOFT_DEADLINE = dt.datetime(2026, 10, 19, 10, 15)
HARD_DEADLINE = dt.datetime(2026, 10, 19, 10, 25)
PY = ".venv/bin/python"
GRID = [PY, "-u", "-m", "scripts.run_protocol_v2", "--n_workers", "6", "--threads_per_worker", "2", "--grid"]
PROBE = [PY, "-u", "-m", "scripts.probe_backbones", "--heads", "all"]
EXPORT = [PY, "-u", "-m", "scripts.run_protocol_v2", "--export_only", "--grid"]

# (name, estimated minutes, command, log file)
STAGES = [
    ("pretrained grid", 130, GRID + ["cifar_pretrained"], "cifar_pretrained.log"),
    ("pretrained edge steps", 25, GRID + ["cifar_pretrained_edge", "--output_root", "runs/protocol_v2/cifar_pretrained"],
     "cifar_pretrained_edge.log"),
    ("pretrained export", 1, EXPORT + ["cifar_pretrained"], "cifar_pretrained_export.log"),
    ("pretrained linear probe", 10, PROBE + ["--datasets", "cifar10", "--grids", "cifar_pretrained",
                                             "--out", "results/pretrained_probe.csv"], "pretrained_probe.log"),
    ("pretrained mlp probe", 12, PROBE + ["--probe", "mlp", "--datasets", "cifar10", "--grids", "cifar_pretrained",
                                          "--out", "results/pretrained_mlp_probe.csv"], "pretrained_mlp_probe.log"),
    ("night seeds", 80, GRID + ["night_seeds"], "night_seeds.log"),
    ("night seeds linear probe", 10, PROBE + ["--datasets", "cifar10", "mnist", "--grids", "night_seeds",
                                              "--out", "results/night_seeds_probe.csv"], "night_seeds_probe.log"),
    ("pretrained grid 2", 70, GRID + ["cifar_pretrained2"], "cifar_pretrained2.log"),
    ("pretrained 2 probes", 12, PROBE + ["--datasets", "cifar10", "--grids", "cifar_pretrained2",
                                         "--out", "results/pretrained2_probe.csv"], "pretrained2_probe.log"),
    ("2x budget part 2", 60, GRID + ["cifar_long2"], "cifar_long2.log"),
    ("2x budget part 2 probe", 8, PROBE + ["--datasets", "cifar10", "--grids", "cifar_long2",
                                           "--out", "results/long2_probe.csv"], "long2_probe.log"),
    ("pretrained 10k", 110, GRID + ["cifar_pretrained_10k"], "cifar_pretrained_10k.log"),
    ("pretrained 10k probe", 15, PROBE + ["--datasets", "cifar10", "--grids", "cifar_pretrained_10k",
                                          "--out", "results/pretrained_10k_probe.csv"], "pretrained_10k_probe.log"),
    ("mlp probes, newer grids", 25, PROBE + ["--probe", "mlp", "--datasets", "cifar10", "--grids", "cifar_pretrained2",
                                             "--out", "results/pretrained2_mlp_probe.csv"], "pretrained2_mlp_probe.log"),
    ("pretrained 1k", 35, GRID + ["cifar_pretrained_1k"], "cifar_pretrained_1k.log"),
    ("pretrained 1k probe", 8, PROBE + ["--datasets", "cifar10", "--grids", "cifar_pretrained_1k",
                                        "--out", "results/pretrained_1k_probe.csv"], "pretrained_1k_probe.log"),
    ("pretrained 10k mlp probe", 20, PROBE + ["--probe", "mlp", "--datasets", "cifar10", "--grids",
                                              "cifar_pretrained_10k", "--out", "results/pretrained_10k_mlp_probe.csv"],
     "pretrained_10k_mlp_probe.log"),
]
FINAL_EXPORTS = ["cifar_pretrained", "night_seeds", "cifar_pretrained2", "cifar_long2", "cifar_pretrained_10k",
                 "cifar_pretrained_1k"]


def log(msg: str):
    print(f"{dt.datetime.now():%F %T} {msg}", flush=True)


def run(cmd: list[str], logfile: str, kill_at: dt.datetime) -> str:
    with open(f"runs/protocol_v2/{logfile}", "a") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            if dt.datetime.now() >= kill_at:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(20)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                return "killed at hard deadline"
            time.sleep(15)
    return f"exit {proc.returncode}"


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
    for grid in FINAL_EXPORTS:
        if os.path.isdir(f"runs/protocol_v2/{grid}"):
            log(f"export {grid}: {run(EXPORT + [grid], 'final_export.log', HARD_DEADLINE + dt.timedelta(minutes=4))}")
    log("queue done" + (" (stopped at hard deadline)" if stopped else ""))


if __name__ == "__main__":
    main()
