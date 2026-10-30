"""Score the night29 pre-registration (predictions/prospective_predictions.md). From repository root."""
import csv, json, glob, re, statistics as st
from collections import defaultdict

def finals(grids, pat):
    val = defaultdict(dict)
    for g in grids:
        for f in glob.glob(f"results/protocol_v2/{g}/*/final.json"):
            m = re.search(pat, f)
            val[(m.group(1), float(m.group(2)))][int(m.group(3))] = json.load(open(f))["val_acc"] * 100
    return val

def select(val, name):
    sel = {}
    print(f"== {name}: selection (mean val, seeds 0-2) ==")
    for h in sorted({h for h, _ in val}):
        row = {lr: st.mean(v for s, v in d.items() if s < 3) for (hh, lr), d in val.items() if hh == h}
        best = max(row, key=row.get); sel[h] = best
        print(f"  {h:22s}", {k: round(v, 2) for k, v in sorted(row.items())}, "->", best, "EDGE" if best in (min(row), max(row)) else "interior")
    return sel

def probes(paths):
    out = defaultdict(dict)
    for p in paths:
        for r in csv.DictReader(open(p)):
            out[r["source_head"]][(float(r["source_lr"]), int(r["seed"]))] = (float(r["probe_test_acc"]), float(r["sat_090_z"]) / 100)
    return out

def table(P, sel, keys, seeds):
    ph, sat = {}, {}
    for k, (csvh, valh) in keys.items():
        d = P[csvh]; lr = sel[valh]
        got = {s: d[(lr, s)] for s in seeds if (lr, s) in d}
        assert len(got) == len(seeds), (k, lr, sorted(got))
        ph[k] = {s: x[0] for s, x in got.items()}; sat[k] = st.mean(x[1] for x in got.values())
        print(f"  {k:8s} lr={lr:<7} probe {st.mean(ph[k].values()):6.2f}±{st.stdev(ph[k].values()):5.2f}  S={sat[k]:.2f}  {[round(ph[k][s],1) for s in seeds]}")
    return ph, sat

def pair(ph, a, b, seeds, sub=None):
    ss = [s for s in seeds if sub is None or s in sub]
    d = [ph[a][s] - ph[b][s] for s in ss]
    m = st.mean(d) if d else float("nan")
    print(f"  {a}-{b}: {m:+.2f}  pos {sum(x>0 for x in d)}/{len(d)} neg {sum(x<0 for x in d)}/{len(d)}  {[round(x,1) for x in d]}")
    return m, sum(x > 0 for x in d), sum(x < 0 for x in d), len(d)

def verdict(name, ok): print(f"  >> {name}: {'PASS' if ok else 'FAIL'}")

for probe in ("", "_mlp"):
    print(f"\n######## {'LINEAR' if not probe else 'MLP'} probe ########")
    # ---- (A) SVHN
    val = finals(["svhn", "svhn_seeds"], r"svhn_subset4000_(.+)_lr([\d.e-]+)_s(\d)/")
    sel = select({k: v for k, v in val.items()}, "SVHN")
    P = probes([f"results/svhn_all{probe}_probe.csv", f"results/svhn_seeds_all{probe}_probe.csv"])
    keys = {"linear": ("linear", "linear"), "noent": ("quantum+noent", "quantum_noent"), "pairs": ("quantum+pairs", "quantum_pairs"),
            "default": ("quantum", "quantum"), "last": ("quantum+entlast", "quantum_entlast"), "prod": ("cosprod", "cosprod")}
    S10 = list(range(10)); ph, _ = table(P, sel, keys, S10)
    col = {k: {s for s in S10 if val[(keys[k][1], sel[keys[k][1]])][s] < 25} for k in keys}
    print("  collapse (head val<25%):", {k: f"{len(v)} {sorted(v)}" for k, v in col.items()})
    a = pair(ph, "default", "noent", S10); b = pair(ph, "last", "default", S10)
    verdict("H_order-svhn2", a[0] <= -1 and a[2] >= 8 and b[0] <= -1 and b[2] >= 8)
    c = pair(ph, "pairs", "default", S10); d = pair(ph, "pairs", "noent", S10)
    verdict("H_pairs-svhn2", c[0] >= 1 and c[1] >= 8 and abs(d[0]) <= 1.5)
    e = pair(ph, "linear", "default", S10); verdict("H_gap-svhn2", e[0] >= 1 and e[1] >= 8)
    n = {k: len(v) for k, v in col.items()}
    verdict("H_collapse", n["noent"] <= n["default"] <= n["last"] and n["pairs"] <= n["default"] and n["last"] - n["noent"] >= 3)
    for x, y in [("default", "noent"), ("last", "default")]:
        surv = set(S10) - col[x] - col[y]
        r = pair(ph, x, y, S10, surv)
        print(f"  >> H_graded {x}-{y}: n={r[3]} " + ("INCONCLUSIVE (<5)" if r[3] < 5 else ("PASS" if r[0] <= -1 else "FAIL")))
    f = pair(ph, "prod", "default", S10); print(f"  >> secondary products within 2: {'PASS' if abs(f[0]) <= 2 else 'MISS'}")
    print(f"  >> refutation: {'MET' if a[0] > -.5 and b[0] > -.5 else 'not met'}")
    # ---- (B) reup
    val = finals(["cifar_reup", "cifar_reup_seeds"], r"cifar10_subset4000_(.+)_rustrong_lr([\d.e-]+)_s(\d)/")
    sel = select(val, "REUP")
    P = probes([f"results/reup{probe}_probe.csv", f"results/reup_seeds{probe}_probe.csv"])
    keys = {"noent": ("quantum+noent+rustrong", "quantum_noent"), "pairs": ("quantum+pairs_last+rustrong", "quantum_pairs_last"),
            "default": ("quantum+rustrong", "quantum"), "last": ("quantum+entlast+rustrong", "quantum_entlast")}
    S5 = list(range(5)); ph, _ = table(P, sel, keys, S5)
    a = pair(ph, "default", "noent", S5); b = pair(ph, "last", "default", S5)
    verdict("H_reup-order", a[0] <= -1 and a[2] >= 4 and b[0] <= -1 and b[2] >= 4)
    order = {"noent": 1, "pairs": 1.5, "default": 2.5, "last": 5.25}
    means = {k: st.mean(ph[k].values()) for k in keys}
    rk = lambda xs: {k: i for i, k in enumerate(sorted(xs, key=xs.get))}
    r1, r2 = rk(means), rk(order); n4 = 4
    rho = 1 - 6 * sum((r1[k] - r2[k]) ** 2 for k in keys) / (n4 * (n4 * n4 - 1))
    print(f"  Spearman(mean probe, order) = {rho:+.2f}"); verdict("H_reup-rank", rho <= -0.8)
    print(f"  >> refutation: {'MET' if a[0] > -.5 and b[0] > -.5 else 'not met'}")
    # ---- (C) mnv3
    val = finals(["cifar_mnv3", "cifar_mnv3_seeds"], r"cifar10_subset4000_(.+)_mobilenetv3s_pt_bbm0.1_lr([\d.e-]+)_s(\d)/")
    sel = select(val, "MNV3")
    P = probes([f"results/mnv3{probe}_probe.csv", f"results/mnv3_seeds{probe}_probe.csv"])
    keys = {"linear": ("linear+mobilenetv3s_pt", "linear"), "noent": ("quantum+noent+mobilenetv3s_pt", "quantum_noent"),
            "pairs": ("quantum+pairs+mobilenetv3s_pt", "quantum_pairs"), "default": ("quantum+mobilenetv3s_pt", "quantum"),
            "last": ("quantum+entlast+mobilenetv3s_pt", "quantum_entlast")}
    ph, _ = table(P, sel, keys, S5)
    a = pair(ph, "default", "noent", S5); b = pair(ph, "last", "default", S5)
    verdict("H_order-mnv3", a[0] <= -1 and a[2] >= 4 and b[0] <= -1 and b[2] >= 4)
    c = pair(ph, "pairs", "default", S5); d = pair(ph, "pairs", "noent", S5)
    verdict("H_pairs-mnv3", c[0] >= 1 and c[1] >= 4 and abs(d[0]) <= 1.5)
    e = pair(ph, "linear", "default", S5); g = pair(ph, "linear", "noent", S5)
    verdict("H_gap-mnv3", e[0] >= 1 and e[1] >= 4 and abs(g[0]) <= 1.0)
    print(f"  >> refutation: {'MET' if a[0] > -.5 and b[0] > -.5 else 'not met'}")
