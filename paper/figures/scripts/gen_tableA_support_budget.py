#!/usr/bin/env python3
"""Appendix tables: (i) classical product heads with designed supports that separate product
order from support overlap, and (ii) the 2x training-budget check. CIFAR-10, 4000 images,
seeds 0-2. From repository root::

    PYTHONPATH=. .venv/bin/python paper/figures/scripts/gen_tableA_support_budget.py
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import ms, paired, write_table  # noqa: E402

from quantcnn.models.classical_heads import DESIGNED_SUPPORTS, cnot_ring_supports  # noqa: E402
from scripts.summarize_v2 import BASE_PROBES, circuit_data, select_lr  # noqa: E402

ROOT = Path("results/protocol_v2")
SEEDS = (0, 1, 2)
# key -> (label, supports, grid holding its runs)
SUPPORT_CELLS = {
    "cos": ("cos (order 1)", [[k] for k in range(8)], "cifar_classical"),
    "cosprod_win2": ("window 2", DESIGNED_SUPPORTS["win2"], "cifar_support"),
    "cosprod_star2": ("star 2", DESIGNED_SUPPORTS["star2"], "cifar_support"),
    "cosprod": ("default ring products", cnot_ring_supports(8, 4), "cifar_cosprod"),
    "cosprod_win3": ("window 3", DESIGNED_SUPPORTS["win3"], "cifar_support"),
    "cosprod_star3": ("star 3", DESIGNED_SUPPORTS["star3"], "cifar_support"),
    "cosprod_win4": ("window 4", DESIGNED_SUPPORTS["win4"], "cifar_support"),
    "cosprod_gap4": ("gapped window 4", DESIGNED_SUPPORTS["gap4"], "cifar_support"),
}
CONTRASTS = [("cosprod_star3", "cosprod_gap4", "equal overlap, both rank 8"),
             ("cosprod_star2", "cosprod_win3", "approximately equal overlap, both rank 8"),
             ("cosprod_star3", "cosprod_win4", "equal overlap, rank 8 vs 5"),
             ("cosprod_star2", "cosprod_win2", "equal order, rank 8 vs 7"),
             ("cosprod_star3", "cosprod_win3", "equal order, both rank 8")]
PROBES = BASE_PROBES + [Path(f"results/{n}_probe.csv") for n in ("classical", "cosprod", "support", "long")]


def _rank(sup) -> int:
    import torch

    m = torch.zeros(8, 8)
    for k, s in enumerate(sup):
        m[k, s] = 1.0
    return int(torch.linalg.matrix_rank(m))


def _overlap(sup) -> float:
    return st.mean(len(set(a) & set(b)) for a, b in itertools.permutations(sup, 2))


def support_table():
    test, probe = {}, {}
    for key, (_, _, grid) in SUPPORT_CELLS.items():
        cells, probes = circuit_data(ROOT / grid, PROBES)
        by_lr = cells[("cifar10", 4000, key)]
        test[key] = {s: v[1] for s, v in by_lr[select_lr(by_lr)].items() if s in SEEDS}
        probe[key] = {s: v for s, v in probes[("cifar10", 4000, key)].items() if s in SEEDS}
    lines = [f"{label} & {st.mean(len(s) for s in sup):g} & {_overlap(sup):.2f} & {_rank(sup)} & {ms(test[k].values())} & "
             f"{ms(probe[k].values())} \\\\" for k, (label, sup, _) in SUPPORT_CELLS.items()]
    contrasts = "; ".join(f"{SUPPORT_CELLS[a][0]} $-$ {SUPPORT_CELLS[b][0]} ({why}): {paired(probe[a], probe[b])}"
                          for a, b, why in CONTRASTS)
    body = r"""\begin{table}[h]
\centering
\caption{Classical product heads whose supports separate product order from overlap (mean
pairwise $|S_k\cap S_l|$), trained end to end like every other head (CIFAR-10, 4,000 training
images; 3 seeds; each at its validation-selected learning rate). Rank: rank of the $8\times8$
support matrix, which bounds the number of independent input directions the products can
respond to. Window $w$: $S_k=\{k,\dots,k{+}w{-}1\}$; gapped window 4: $\{k,k{+}1,k{+}2,k{+}4\}$
(mod 8). Star 2: $\{1,2\},\{0,1\},\{0,2\},\dots,\{0,7\}$. Star 3: $\{1,2,3\},
\{0,1,2\},\{0,1,3\},\dots,\{0,1,7\},\{0,2,3\}$. Paired probe differences: """ + contrasts + r""".}
\label{tab:support}
\small
\begin{tabular}{lccccc}
\toprule
Supports & Order & Overlap & Rank & Test & Probe \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_support", body)


def _runs(grid: str, match) -> dict:
    out = {}
    for fin in sorted((ROOT / grid).glob("cifar10_subset4000_*/final.json")):
        m = json.loads((fin.parent / "meta.json").read_text())
        if match(m) and m["lr"] == 1e-2 and int(m["seed"]) in SEEDS:
            f = json.loads(fin.read_text())
            rows = list(csv.DictReader((fin.parent / "metrics.csv").open()))
            out[int(m["seed"])] = (f["test_acc"] * 100, f["train_clean_acc"] * 100, float(rows[-1]["train_loss"]))
    return out


def budget_table():
    probes = {}
    for path in PROBES + [Path("results/entangle_probe.csv")]:
        for r in csv.DictReader(path.open()):
            if r["dataset"] == "cifar10" and r["train_subset_size"] == "4000" and int(r["seed"]) in SEEDS:
                probes.setdefault(r["source_head"], {})[int(r["seed"])] = float(r["probe_test_acc"])
    cells = {  # key -> (label, 80-epoch grid, run matcher)
        "quantum+noent": ("no CNOT", "cifar_circuit", lambda m: m.get("entangle", True) is False and m.get("encoding", "pi_tanh") == "pi_tanh"),
        "quantum": ("ring (default)", "cifar_val", lambda m: m.get("entangle", True) is True and m.get("encoding", "pi_tanh") == "pi_tanh"
                    and m.get("pre_norm", "none") == "none" and m.get("n_var_layers", 4) == 4),
        "quantum+entlast": ("ring, last", "cifar_entangle", lambda m: m.get("entangle", True) == "last"),
    }
    lines, long_probe = [], {}
    for key, (label, grid, match) in cells.items():
        for ep, g in ((80, grid), (160, "cifar_long")):
            r = _runs(g, lambda m, ep=ep: match(m) and m["epochs"] == ep and m["head"] == "quantum")
            pk = key if ep == 80 else key + "+ep160"
            if ep == 160:
                long_probe[key] = probes[pk]
            lines.append(f"{label if ep == 80 else ''} & {ep} & {ms(x[1] for x in r.values())} & "
                         f"{st.mean(x[2] for x in r.values()):.2f} & {ms(x[0] for x in r.values())} & "
                         f"{ms(probes[pk].values())} \\\\")
    gap = paired(long_probe["quantum"], long_probe["quantum+noent"])
    body = r"""\begin{table}[h]
\centering
\caption{Twice the training budget (160 instead of 80 epochs, same head learning rate
$10^{-2}$ and cosine schedule; CIFAR-10, 4,000 training images; 3 seeds). Clean-train:
accuracy on the unaugmented training subset at the selected checkpoint; loss: final-epoch
training loss on augmented batches. At 160 epochs, default $-$ no-CNOT probe accuracy is
""" + gap + r""" (paired).}
\label{tab:budget}
\small
\begin{tabular}{lccccc}
\toprule
Circuit & Epochs & Clean-train & Train loss & Test & Probe \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_table("tableA_budget", body)


if __name__ == "__main__":
    support_table()
    budget_table()
