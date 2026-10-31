"""Score the night31 seeds 5-9 replications (predictions/prospective_predictions.md, 2026-10-31).
Each verdict uses seeds 5-9 only; 10-seed pooled numbers (with seeds 0-4) are printed descriptively.
From repository root: ``PYTHONPATH=. .venv/bin/python scripts/score_night31.py``."""
import csv
import statistics as st

R5, ALL = list(range(5, 10)), list(range(10))


def load(*paths):
    """(head, seed) -> (probe test acc, S%); each file holds one (selected) lr per head."""
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            key = (r["source_head"], int(r["seed"]))
            assert key not in out, (p, key)
            out[key] = (float(r["probe_test_acc"]), float(r["sat_090_z"]))
    return out


def pair(P, a, b, seeds, label):
    d = [P[(a, s)][0] - P[(b, s)][0] for s in seeds]
    m = st.mean(d)
    print(f"  {label:26s} {m:+6.2f} ± {st.stdev(d):4.2f}  pos {sum(x > 0 for x in d)}/{len(d)}  "
          f"neg {sum(x < 0 for x in d)}/{len(d)}  {[round(x, 1) for x in d]}")
    return m, sum(x > 0 for x in d), sum(x < 0 for x in d)


def mean(P, h, seeds):
    v = [P[(h, s)][0] for s in seeds]
    return f"{st.mean(v):.2f} ± {st.stdev(v):.2f} (S% {st.mean(P[(h, s)][1] for s in seeds):.2f})"


def spearman(x, y):
    rk = lambda v: [sorted(v).index(a) for a in v]
    rx, ry = rk(x), rk(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def v(name, ok):
    print(f"  >> {name}: {'PASS' if ok else 'FAIL'}")


# ---- (A) gated escape-only, seeds 5-9
print("== (A) gated escape-only, seeds 5-9 (linear probe) ==")
P = load("results/escgate_rep_probe.csv", "results/esc_rep_probe.csv", "results/retest_probe.csv")
base = {"noent": "quantum+noent", "default": "quantum", "last": "quantum+entlast"}
g = {}
for lay, b in base.items():
    suf = {"noent": "+noent", "default": "", "last": "+entlast"}[lay]
    print(f" {lay}: plain {mean(P, b, R5)}")
    for var in ("esc", "escsat", "escmid"):
        h = f"quantum+pi_tanh_{var}{suf}"
        print(f"   {var:6s} {mean(P, h, R5)}")
        g[lay, var] = pair(P, h, b, R5, f"gain {var}")
sat, mid = g["last", "escsat"], g["last", "escmid"]
satd = {s: P[(f"quantum+pi_tanh_escsat+entlast", s)][0] - P[(f"quantum+pi_tanh_escmid+entlast", s)][0] for s in R5}
print(f"  sat>mid per seed: {sum(x > 0 for x in satd.values())}/5 {[round(x, 1) for x in satd.values()]}")
v("H_sat-route-rep", sat[0] >= 4.5 and sat[1] >= 4)
v("H_mid-small-rep", mid[0] <= 2.3 and sum(x > 0 for x in satd.values()) >= 4)
print(f"  refutation: {'MET' if mid[0] >= sat[0] or sat[0] <= 2.3 else 'not met'}")
print(f"  secondary default gain_sat >= 0.75: {g['default', 'escsat'][0] >= 0.75}; "
      f"no CNOT both <= 1.5: {max(g['noent', 'escsat'][0], g['noent', 'escmid'][0]) <= 1.5}")
P0 = load("results/escgate_probe.csv")
Pb = {}
from scripts.bottleneck_geometry import _probe_rows  # seeds 0-4 baselines at their selected lr
for lay, b in base.items():
    for s, (_, acc) in _probe_rows()[b].items():
        Pb[(b, s)] = (acc, 0.0)
PP = {**P, **P0, **Pb}
for var in ("escsat", "escmid"):
    pair(PP, f"quantum+pi_tanh_{var}+entlast", "quantum+entlast", ALL, f"pooled 10-seed last {var}")

# ---- (B) re-uploading, seeds 5-9
for probe in ("", "_mlp"):
    print(f"\n== (B) re-uploading, seeds 5-9 ({'linear' if not probe else 'MLP'} probe) ==")
    P = load(f"results/reup_seeds2{probe}_probe.csv")
    keys = ["quantum+noent+rustrong", "quantum+pairs_last+rustrong", "quantum+rustrong", "quantum+entlast+rustrong"]
    for k in keys:
        print(f"  {k:28s} {mean(P, k, R5)}")
    a = pair(P, keys[2], keys[0], R5, "default - no CNOT")
    b = pair(P, keys[3], keys[2], R5, "last - default")
    rho = spearman([st.mean(P[(k, s)][0] for s in R5) for k in keys], [1, 1.5, 2.5, 5.25])
    print(f"  Spearman rho {rho:+.2f}")
    v("H_reup-order-rep", a[0] <= -1 and a[2] >= 4 and b[0] <= -1 and b[2] >= 4)
    v("H_reup-rank-rep", rho <= -0.8)
    print(f"  refutation: {'MET' if a[0] > -.5 and b[0] > -.5 else 'not met'}")
    Q = {**load(f"results/reup{probe}_probe.csv", f"results/reup_seeds{probe}_probe.csv"), **P}
    try:
        for k in keys:
            print(f"  pooled {k:28s} {mean(Q, k, ALL)}")
        pair(Q, keys[0], keys[1], ALL, "pooled no CNOT - pairs")
        pair(Q, keys[2], keys[0], ALL, "pooled default - no CNOT")
        pair(Q, keys[3], keys[2], ALL, "pooled last - default")
    except KeyError as e:
        print("  pooled: missing", e)

# ---- (C) MobileNetV3-Small, seeds 5-9
for probe in ("", "_mlp"):
    print(f"\n== (C) MobileNetV3-Small, seeds 5-9 ({'linear' if not probe else 'MLP'} probe) ==")
    P = load(f"results/mnv3_seeds2{probe}_probe.csv")
    t = "+mobilenetv3s_pt"
    L, N, Pa, D, E = (f"linear{t}", f"quantum+noent{t}", f"quantum+pairs{t}", f"quantum{t}", f"quantum+entlast{t}")
    for k in (L, N, Pa, D, E):
        print(f"  {k:34s} {mean(P, k, R5)}")
    a = pair(P, D, N, R5, "default - no CNOT")
    b = pair(P, E, D, R5, "last - default")
    v("H_order-mnv3-rep", a[0] <= -1 and a[2] >= 4 and b[0] <= -1 and b[2] >= 4)
    c = pair(P, Pa, D, R5, "pairs - default")
    d = pair(P, Pa, N, R5, "pairs - no CNOT")
    v("H_pairs-mnv3-rep", c[0] >= 1 and c[1] >= 4 and abs(d[0]) <= 1.5)
    e = pair(P, L, D, R5, "linear - default")
    f = pair(P, L, N, R5, "linear - no CNOT")
    v("H_gap-mnv3-rep", e[0] >= 1 and e[1] >= 4 and abs(f[0]) <= 1.0)
    print(f"  refutation: {'MET' if a[0] > -.5 and b[0] > -.5 else 'not met'}")
    Q = {**load(f"results/mnv3{probe}_probe.csv", f"results/mnv3_seeds{probe}_probe.csv"), **P}
    try:
        for k in (L, N, Pa, D, E):
            print(f"  pooled {k:34s} {mean(Q, k, ALL)}")
        for x, y, lab in [(N, L, "no CNOT - linear"), (Pa, L, "pairs - linear"), (D, L, "default - linear"),
                          (E, L, "last - linear"), (D, N, "default - no CNOT"), (E, D, "last - default")]:
            pair(Q, x, y, ALL, "pooled " + lab)
    except KeyError as e:
        print("  pooled: missing", e)
