#!/usr/bin/env python3
"""Reproduce every row of every probe CSV in results/ from the saved models (provenance, 2026-10-31).

For each row (dataset, size, source head, source lr, seed) the matching run directories are found by the same
head key as ``scripts.probe_backbones``; the bottleneck is re-embedded and the probe re-fit with the same code
(``scripts.frozen_heads._train_head``), probe kind (MLP if the file name contains ``mlp``) and learning-rate grid
(linear: PROBE_LRS; MLP: MLP_PROBE_LRS, or 1e-4..3e-2 for the files in EXT_FILES). A row reproduces if some
matching run gives the same selected probe lr, probe test accuracy and saturation. Every candidate is logged.
Sharded so several processes can run at once::

    PYTHONPATH=. .venv/bin/python -m scripts.reproduce_probes --shard 0 --n_shards 4
    PYTHONPATH=. .venv/bin/python -m scripts.reproduce_probes --summarize
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import torch

from quantcnn.data import get_splits
from quantcnn.models.hybrid_classifier import HybridClassifier
from scripts.frozen_heads import _embed, _train_head
from scripts.probe_backbones import MLP_PROBE_LRS, PROBE_LRS, _head_input, _head_key

OUT = Path("results/provenance/reproduce")
EXT_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
EXT_FILES = ("mlp_probe_ext.csv", "family_mlp_probe.csv")  # probed with --mlp_lrs extended down to 1e-4


def grid(fname: str) -> tuple[str, list[float]]:
    if "mlp" not in fname:
        return "linear", PROBE_LRS
    return "mlp", EXT_GRID if fname in EXT_FILES else MLP_PROBE_LRS


def run_index() -> dict:
    idx = defaultdict(list)
    for m in sorted(glob.glob("runs/protocol_v2/*/*/meta.json")):
        d = Path(m).parent
        if not (d / "best_model.pt").exists():
            continue
        meta = json.loads(Path(m).read_text())
        idx[(meta["dataset"], str(meta["train_subset_size"]), _head_key(meta), f"{float(meta['lr']):g}",
             str(meta["seed"]))].append(d)
    return idx


def tasks() -> list[tuple]:
    out = []
    for f in sorted(glob.glob("results/*probe*.csv")):
        for i, r in enumerate(csv.DictReader(open(f))):
            out.append((Path(f).name, i, r))
    return out


@torch.no_grad()
def embed(run: Path, device):
    meta = json.loads((run / "meta.json").read_text())
    ds = meta["dataset"]
    model = HybridClassifier(10, int(meta.get("bottleneck_dim", 8)), meta["backbone"], meta["head"],
                             1 if ds == "mnist" else 3, mlp_hidden=meta.get("mlp_hidden"),
                             n_var_layers=meta.get("n_var_layers", 4), reupload_mode=meta.get("reupload_mode", "none"),
                             quantum_sim="torch", pre_norm=meta.get("pre_norm", "none"),
                             encoding=meta.get("encoding", "pi_tanh"), entangle=meta.get("entangle", True))
    model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu"))
    model.to(device)
    sp = get_splits(ds, int(meta["train_subset_size"]), int(meta["seed"]), 2000)
    emb = {"train": _embed(model, sp["train_clean"], device), "val": _embed(model, sp["val"], device),
           "test": _embed(model, sp["test"], device)}
    z = emb["test"][0]
    sat_z = round((torch.tanh(z).abs() > 0.9).float().mean().item() * 100, 2)
    sat_in = round((torch.tanh(_head_input(model, z)).abs() > 0.9).float().mean().item() * 100, 2)
    return emb, sat_z, sat_in, int(meta["seed"])


def reproduce(shard: int, n_shards: int):
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    idx = run_index()
    by_key = defaultdict(list)
    for t in tasks():
        r = t[2]
        by_key[(r["dataset"], r["train_subset_size"], r["source_head"], r["source_lr"], r["seed"])].append(t)
    keys = sorted(by_key)[shard::n_shards]
    rows_out, cand_out = OUT / f"rows_{shard}.csv", OUT / f"candidates_{shard}.csv"
    done = set()
    if rows_out.exists():
        done = {(r["file"], r["row"]) for r in csv.DictReader(open(rows_out))}
    for n_key, key in enumerate(keys):
        pending = [t for t in by_key[key] if (t[0], str(t[1])) not in done]
        if not pending:
            continue
        runs = idx.get(key, [])
        results = {t[:2]: [] for t in pending}
        for run in runs:
            emb, sat_z, sat_in, seed = embed(run, device)
            cache = {}
            for fname, i, r in pending:
                kind, lrs = grid(fname)
                if (kind, tuple(lrs)) not in cache:
                    cands = []
                    for plr in lrs:
                        res = _train_head("linear" if kind == "linear" else "mlp_probe", emb, seed, plr, 60)
                        cands.append((plr, res))
                        with open(cand_out, "a", newline="") as fh:
                            w = csv.writer(fh)
                            if fh.tell() == 0:
                                w.writerow(["run", "probe", "grid", "probe_lr", "best_epoch", "probe_val_acc", "probe_test_acc"])
                            w.writerow([str(run), kind, "/".join(f"{x:g}" for x in lrs), f"{plr:g}", res["best_epoch"],
                                        round(res["val_acc"] * 100, 3), round(res["test_acc"] * 100, 3)])
                    best = None
                    for plr, res in cands:
                        if best is None or res["val_acc"] > best[1]["val_acc"]:
                            best = (plr, res)
                    cache[(kind, tuple(lrs))] = best
                plr, res = cache[(kind, tuple(lrs))]
                got = dict(probe_lr=f"{plr:g}", probe_test_acc=round(res["test_acc"] * 100, 3), sat_090_z=sat_z,
                           sat_090_head_input=sat_in)
                ok = (got["probe_lr"] == r["probe_lr"] and abs(got["probe_test_acc"] - float(r["probe_test_acc"])) < 1e-6
                      and abs(sat_z - float(r["sat_090_z"])) < 1e-6)
                results[(fname, i)].append((run, got, ok))
        with open(rows_out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "row", "dataset", "train_subset_size", "source_head", "source_lr",
                                               "seed", "n_runs", "status", "run", "orig_probe_lr", "repro_probe_lr",
                                               "orig_probe_test_acc", "repro_probe_test_acc", "orig_sat_090_z",
                                               "repro_sat_090_z"])
            if fh.tell() == 0:
                w.writeheader()
            for fname, i, r in pending:
                res = results[(fname, i)]
                hit = next((x for x in res if x[2]), None)
                pick = hit or (res[0] if res else None)
                status = "exact" if hit else ("no_run" if not res else "differs")
                w.writerow(dict(file=fname, row=i, dataset=key[0], train_subset_size=key[1], source_head=key[2],
                                source_lr=key[3], seed=key[4], n_runs=len(runs), status=status,
                                run=str(pick[0]) if pick else "", orig_probe_lr=r["probe_lr"],
                                repro_probe_lr=pick[1]["probe_lr"] if pick else "",
                                orig_probe_test_acc=r["probe_test_acc"],
                                repro_probe_test_acc=pick[1]["probe_test_acc"] if pick else "",
                                orig_sat_090_z=r["sat_090_z"], repro_sat_090_z=pick[1]["sat_090_z"] if pick else ""))
        print(f"[shard {shard}] {n_key + 1}/{len(keys)} {key} runs={len(runs)}", flush=True)


def summarize():
    rows = [r for f in sorted(OUT.glob("rows_*.csv")) for r in csv.DictReader(open(f))]
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by[r["file"]][r["status"]] += 1
    total = defaultdict(int)
    for f, d in sorted(by.items()):
        for k, v in d.items():
            total[k] += v
        if set(d) != {"exact"}:
            print(f"  {f}: {dict(d)}")
    print("total:", dict(total), "of", sum(total.values()), "rows;", len(tasks()), "rows in results/*probe*.csv")
    bad = [r for r in rows if r["status"] != "exact"]
    if bad:
        with open(OUT / "not_exact.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(bad[0]))
            w.writeheader()
            w.writerows(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    torch.set_num_threads(a.threads)
    reproduce(a.shard, a.n_shards)


if __name__ == "__main__":
    main()
