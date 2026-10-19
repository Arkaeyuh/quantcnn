#!/usr/bin/env python3
"""Table 4: the representation cost on an ImageNet-pretrained ResNet-18 fine-tuned through
each head, across training-set sizes (CIFAR-10). Probe accuracy of the frozen bottleneck and
paired differences to the linear head. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_table4_pretrained.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

from quantcnn.models.classical_heads import cnot_ring_supports  # noqa: E402
from scripts.probe_backbones import _head_key  # noqa: E402
from scripts.summarize_v2 import circuit_data, select_lr  # noqa: E402

ROOT = Path("results/protocol_v2")
SUF = "+resnet18_pt"
# key (without suffix) -> (label, CNOT structure for the order, or None for no CNOTs)
HEADS = {
    "linear": ("linear", "---"),
    "quantum+noent": ("VQC, no CNOT", None),
    "quantum+pairs": ("VQC, pairs", (4, False, "pairs")),
    "cosprod": ("cos-prod.\\ (default supports)", (4, False, "ring")),
    "quantum": ("VQC, ring (default)", (4, False, "ring")),
    "quantum+entlast": ("VQC, ring, last", (4, True, "ring")),
}
# training size -> (main grids, extra-seed grids, probe files, MLP-probe files)
SIZES = {
    1000: (["cifar_pretrained_1k"], [], ["pretrained_1k_probe"], []),
    4000: (["cifar_pretrained", "cifar_pretrained2"], ["night_seeds", "pretrained2_seeds"],
           ["pretrained_probe", "pretrained2_probe", "night_seeds_probe", "pretrained2_seeds_probe"],
           ["pretrained_mlp_probe", "pretrained2_mlp_probe"]),
    10000: (["cifar_pretrained_10k"], [], ["pretrained_10k_probe"], ["pretrained_10k_mlp_probe"]),
}


def _probes(names: list[str], n: int) -> dict:
    out = {}
    for name in names:
        for r in csv.DictReader(Path(f"results/{name}.csv").open()):
            if r["dataset"] == "cifar10" and r["train_subset_size"] == str(n):
                out.setdefault(r["source_head"], {})[int(r["seed"])] = float(r["probe_test_acc"])
    return out


def _test(grids: list[str], extra: list[str], n: int) -> dict:
    out = {}
    for g in grids:
        cells, _ = circuit_data(ROOT / g, [])
        for (ds, nn, key), by in cells.items():
            if nn == n:
                lr = select_lr(by)
                out[key] = (lr, {s: v[1] for s, v in by[lr].items()})
    for g in extra:
        for fin in (ROOT / g).glob(f"cifar10_subset{n}_*/final.json"):
            m = json.loads((fin.parent / "meta.json").read_text())
            key = _head_key(m)
            if key in out and m["lr"] == out[key][0]:
                out[key][1][int(m["seed"])] = json.loads(fin.read_text())["test_acc"] * 100
    return out


def _order(structure) -> str:
    if structure == "---":
        return "---"
    return "1" if structure is None else f"{st.mean(len(s) for s in cnot_ring_supports(8, *structure)):g}"


def main():
    data = {n: (_test(g, e, n), _probes(p, n), _probes(mlp, n)) for n, (g, e, p, mlp) in SIZES.items()}
    lines = []
    for key, (label, structure) in HEADS.items():
        k = key + SUF
        cells = [label, _order(structure)]
        for n in SIZES:
            test, probe, mlp = data[n]
            if k not in probe:
                cells += ["", ""] if n == 4000 else [""]
                continue
            lin = probe["linear" + SUF]
            delta = "---" if key == "linear" else paired(probe[k], {s: lin[s] for s in probe[k]})
            if n == 4000:
                cells += [f"{ms(probe[k].values())} ({len(probe[k])})", delta]
            else:
                cells.append(f"{ms(probe[k].values())}" if key == "linear" else delta)
        test, _, mlp = data[4000]
        lm = mlp["linear" + SUF]
        cells.append("---" if key == "linear" else paired(mlp[k], {s: lm[s] for s in mlp[k]}))
        lines.append(" & ".join(cells) + " \\\\")
    lr4 = {k: data[4000][0][k + SUF][0] for k in HEADS}
    body = r"""\begin{table*}[t]
\centering
\caption{The representation cost on an ImageNet-pretrained ResNet-18 fine-tuned through each
head (CIFAR-10; backbone learning rate $0.1\times$ the head's; 30 epochs at $128$\,px). Probe:
test accuracy (\%) of the common linear probe on the frozen 8-dimensional bottleneck, and its
paired difference to the linear head in percentage points (mean $\pm$ SD; 4,000 images: number
of seeds in parentheses; 1,000 and 10,000 images: 3 seeds). Head learning rates are selected
on validation at 1,000 and 4,000 images and reused at 10,000. At 4,000 images the default and
last-ring circuits select the lowest rate tried ($3\times10^{-4}$, one step below the initial
grid). The last column repeats the 4,000-image difference with a two-layer MLP probe (3 seeds).}
\label{tab:pretrained}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
& & 1,000 images & \multicolumn{2}{c}{4,000 images} & 10,000 images & 4,000, MLP probe \\
\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}
Head & Order & $-$ linear & probe & $-$ linear & $-$ linear & $-$ linear \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_table("table4_pretrained", body)
    print("selected lrs at 4000:", lr4)


if __name__ == "__main__":
    main()
