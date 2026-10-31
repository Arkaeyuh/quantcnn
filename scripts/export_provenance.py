#!/usr/bin/env python3
"""Export configuration and test provenance that the result CSVs do not record (2026-10-31).

Everything here is read from the code that produced the results (or, where the code has since changed,
from the git revision that produced them) or measured by running it. Writes ``results/provenance/``:

  probes.json        linear and MLP probe: layer shapes of the instantiated probes, and optimizer,
                     schedule, batch size, epochs and learning-rate grids captured by running the probe
                     trainer on synthetic data; git history of the probe code vs the probe CSVs
  models.csv         every backbone: conv layers, parameter counts, input resolution seen by the first
                     convolution (forward hook), bottleneck width
  tests.csv          every test in tests/: outcome, time, and the tolerances it asserts
  environment.json   host, chip, library versions (recorded now; the runs' meta.json has no host field)
  measurements.json  settings of measurements whose CSVs lack them (init-gradient batch size, toy model,
                     the committed effective-order cutoff), each with the revision it is read from

From repository root: ``PYTHONPATH=. .venv/bin/python -m scripts.export_provenance``.
"""

from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import math
import platform
import re
import subprocess
import time
import traceback
from pathlib import Path

import torch
from torch import nn

OUT = Path("results/provenance")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _layers(m: nn.Module) -> list[str]:
    return [f"{type(x).__name__}({x.in_features}->{x.out_features})" if isinstance(x, nn.Linear) else type(x).__name__
            for x in m.modules() if not list(x.children())]


def _call_arg(src: str, func: str, index: int):
    """Constant value of positional argument ``index`` in every call to ``func`` in ``src``."""
    return sorted({ast.literal_eval(c.args[index]) for c in ast.walk(ast.parse(src))
                   if isinstance(c, ast.Call) and getattr(c.func, "id", None) == func and len(c.args) > index
                   and isinstance(c.args[index], ast.Constant)})


def _argparse_default(src: str, flag: str):
    for c in ast.walk(ast.parse(src)):
        if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "add_argument" and c.args \
                and isinstance(c.args[0], ast.Constant) and c.args[0].value == flag:
            for kw in c.keywords:
                if kw.arg == "default":
                    return ast.unparse(kw.value)


def probes() -> dict:
    import scripts.frozen_heads as fh
    import scripts.probe_backbones as pb

    src = Path("scripts/probe_backbones.py").read_text()
    out = {"source": "scripts/probe_backbones.py -> scripts/frozen_heads._train_head", "commit": git("rev-parse", "HEAD")}
    mu, sd = torch.zeros(8), torch.ones(8)
    for kind, name in [("linear", "linear"), ("mlp_probe", "mlp")]:
        head = fh._make_head(kind, mu, sd)
        rec = {"class": type(head).__name__, "layers": _layers(head),
               "parameters": sum(p.numel() for p in head.parameters()),
               "standardizes_input": hasattr(head, "mu")}
        # capture optimizer, schedule and batch size by running the real trainer on synthetic embeddings
        seen, real_adamw, real_cos, real_make = {}, torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR, fh._make_head

        def adamw(params, **kw):
            seen["optimizer"] = {"type": "AdamW", **{k: float(v) for k, v in kw.items()}}
            return real_adamw(params, **kw)

        def cos(opt, **kw):
            seen["schedule"] = {"type": "CosineAnnealingLR", **kw}
            return real_cos(opt, **kw)

        sizes = set()

        def make(k, m, s):
            h = real_make(k, m, s)
            h.register_forward_pre_hook(lambda mod, a: sizes.add(a[0].shape[0]) if mod.training else None)
            return h

        torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR, fh._make_head = adamw, cos, make
        try:
            g = torch.Generator().manual_seed(0)
            emb = {s: (torch.randn(n, 8, generator=g), torch.randint(0, 10, (n,), generator=g))
                   for s, n in [("train", 1000), ("val", 200), ("test", 200)]}
            epochs = _call_arg(src, "_train_head", 4)
            assert len(epochs) == 1, epochs
            fh._train_head(kind, emb, 0, 1e-2, epochs[0])
        finally:
            torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR, fh._make_head = real_adamw, real_cos, real_make
        rec.update(seen, epochs=epochs[0], batch_size=max(sizes), last_partial_batch=sorted(sizes)[:-1],
                   selection="epoch and probe learning rate with the highest validation accuracy")
        out[name] = rec
    out["linear"]["lr_grid"] = pb.PROBE_LRS
    out["mlp"]["lr_grid_default"] = ast.literal_eval(_argparse_default(src, "--mlp_lrs").replace("MLP_PROBE_LRS",
                                                                                                 repr(pb.MLP_PROBE_LRS)))
    out["mlp"]["lr_grid_note"] = "--mlp_lrs overrides the default; results/mlp_probe_ext.csv used the grid extended down to 1e-4"
    # history: every change to the probe code, and when each probe CSV was first committed
    hist = git("log", "--format=%h %ad %s", "--date=iso-strict", "--", "scripts/frozen_heads.py", "scripts/probe_backbones.py")
    out["code_history"] = hist.splitlines()
    first = {}
    for f in sorted(Path("results").glob("*probe*.csv")):
        log = git("log", "--diff-filter=A", "--format=%h %ad", "--date=iso-strict", "--", str(f)).splitlines()
        first[f.name] = log[-1] if log else "untracked"
    out["probe_csv_first_commit"] = first
    out["note"] = ("_train_head, MLPProbe and PROBE_LRS last changed in 96bb892 (2026-10-23), which only made the MLP "
                   "probe's input width follow the bottleneck (8 -> d); every probe CSV with an 8-dimensional "
                   "bottleneck is unaffected. Earlier changes predate the probe CSVs they could affect.")
    return out


def models() -> list[dict]:
    from quantcnn.models.hybrid_classifier import HybridClassifier

    rows = []
    for ds, bk, shape in [("mnist", "mnist", (2, 1, 28, 28)), ("cifar10", "cifar10", (2, 3, 32, 32)),
                          ("svhn", "cifar10", (2, 3, 32, 32)), ("cifar10", "resnet18_pt", (2, 3, 32, 32)),
                          ("cifar10", "mobilenetv3s_pt", (2, 3, 32, 32))]:
        m = HybridClassifier(10, 8, bk, "quantum", shape[1], n_var_layers=4, reupload_mode="none", quantum_sim="torch")
        convs = [x for x in m.backbone.modules() if isinstance(x, nn.Conv2d)]
        res = {}
        convs[0].register_forward_pre_hook(lambda mod, a: res.setdefault("hw", tuple(a[0].shape[-2:])) and None)
        with torch.no_grad():
            m.eval()
            z = m.backbone(torch.zeros(shape))
        rows.append(dict(dataset=ds, backbone=bk, backbone_class=type(m.backbone).__name__, conv_layers=len(convs),
                         backbone_params=sum(p.numel() for p in m.backbone.parameters()),
                         head_params=sum(p.numel() for p in m.head.parameters()), data_input=f"{shape[2]}x{shape[3]}",
                         first_conv_input=f"{res['hw'][0]}x{res['hw'][1]}", bottleneck_dim=z.shape[1]))
    return rows


def _tolerances(fn) -> list[str]:
    tree = ast.parse(inspect.getsource(fn).lstrip())
    return sorted({f"{kw.arg}={ast.unparse(kw.value)}" for c in ast.walk(tree) if isinstance(c, ast.Call)
                   for kw in c.keywords if kw.arg in ("atol", "rtol")})


def tests() -> list[dict]:
    rows = []
    for path in sorted(Path("tests").glob("test_*.py")):
        mod = importlib.import_module(f"tests.{path.stem}") if Path("tests/__init__.py").exists() else None
        if mod is None:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            t0, status, err = time.time(), "pass", ""
            try:
                fn()
            except Exception:  # noqa: BLE001 - record any failure
                status, err = "FAIL", traceback.format_exc(limit=1).strip().splitlines()[-1]
            rows.append(dict(file=str(path), test=name, status=status, seconds=round(time.time() - t0, 2),
                             tolerances=" ".join(_tolerances(fn)) or "-",
                             note="torch.allclose default rtol=1e-05 unless rtol is given", error=err))
            print(f"  {status:4s} {path.stem}.{name}", flush=True)
    return rows


def environment() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    import torchvision
    try:
        import pennylane
        pl = pennylane.__version__
    except ImportError:
        pl = "not installed"
    return {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host_os": platform.platform(),
            "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]), "memory_bytes": sh(["sysctl", "-n", "hw.memsize"]),
            "cpu_cores": sh(["sysctl", "-n", "hw.ncpu"]), "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "pennylane": pl, "mps_available": torch.backends.mps.is_available(),
            "runs_on_this_disk": sum(1 for _ in Path("runs/protocol_v2").glob("*/*/final.json")),
            "note": ("Recorded post hoc. Run metadata has no host field. Author attestation, not measured: every run directory counted above was "
                     "written on this machine (runs/ is not synchronized from elsewhere), which is the basis for "
                     "the paper's statement that experiments ran on one Apple M3 Max.")}


def measurements() -> dict:
    out = {}
    rev = git("log", "-1", "--format=%h", "--", "results/init_gradients_svhn_trainbn.csv")
    src = git("show", f"{rev}:scripts/init_gradients.py")
    m = re.search(r"split\((\d+) if train_mode else (\d+)\)", src)
    out["init_gradients_svhn_trainbn"] = {"csv_commit": rev, "source": f"{rev}:scripts/init_gradients.py",
                                          "train_mode_batch": int(m.group(1)), "eval_mode_batch": int(m.group(2)),
                                          "line": m.group(0)}
    rev = git("log", "-1", "--format=%h", "--", "results/toy_order_model.csv")
    src = git("show", f"{rev}:scripts/toy_order_model.py")
    lin = re.search(r"backbone = nn\.Sequential\((.+)\)", src).group(1)
    data = re.search(r"def _data\((.+)\):", src).group(1)
    out["toy_order_model"] = {"csv_commit": rev, "source": f"{rev}:scripts/toy_order_model.py", "backbone": lin,
                              "data_signature": data, "optimizer_steps_per_batch": src.split("def _train")[1]
                              .split("def _acc")[0].count("opt.step()")}
    fix = "3d25933"
    old = git("show", f"{fix}^:scripts/effective_order.py")
    line = next(x.strip() for x in old.splitlines() if "order = " in x and "vis > tol" in x)
    tol = re.search(r"def readout_stats\(.*tol: float = ([0-9e.-]+)", old).group(1)
    out["effective_order_committed_cutoff"] = {"source": f"{fix}^:scripts/effective_order.py", "line": line,
                                               "tol_default": float(tol), "fixed_in": fix,
                                               "csv": "results/effective_order_init_phase_committed.csv"}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    (OUT / "probes.json").write_text(json.dumps(probes(), indent=1))
    rows = models()
    with open(OUT / "models.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "environment.json").write_text(json.dumps(environment(), indent=1))
    (OUT / "measurements.json").write_text(json.dumps(measurements(), indent=1))
    rows = tests()
    with open(OUT / "tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"tests: {sum(r['status'] == 'pass' for r in rows)}/{len(rows)} pass; wrote {OUT}/")


if __name__ == "__main__":
    main()
