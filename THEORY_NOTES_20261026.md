# Theory track A1: the signal budget of a product readout (draft, 2026-10-26)

These are working notes toward a main-text proposition. They are retrospective: nothing here was
pre-registered. Tonight's runs are the pre-registered tests (`predictions/prospective_predictions.md`).

## Statement

Let a readout be r = ∏_{j∈S} c_j with c_j = cos(π tanh z_j) and |S| = m. Write t_j = |tanh z_j|, and
call the *middle zone* δ < t_j < 1 − δ.

**Proposition (signal budget).** Suppose the factors are independent and E|r| ≥ a > 0. Then
some j ∈ S has E|c_j| ≥ a^{1/m}, and for that coordinate

    P(δ < t_j < 1 − δ) ≤ (1 − a^{1/m}) / (1 − cos πδ).

If the factors are identically distributed, this holds for every j ∈ S.

*Proof.* Independence gives E|r| = ∏ E|c_j| ≥ a, so the largest E|c_j| is at least a^{1/m}. On
the middle zone |cos πt| ≤ cos πδ, and |c| ≤ 1 everywhere, so
E|c_j| ≤ 1 − (1 − cos πδ) · P(mid). Rearranging gives the bound. ∎

As m grows, a^{1/m} → 1, so a readout that keeps a fixed share of its signal forces its inputs
out of the middle zone. They gather at one of two ends: z ≈ 0, where c = +1, or saturation, where
c = −1 and sech² z ≈ 0. A coordinate pinned at one end carries little information, so an informative
coordinate in a high-order support must split its mass between the two ends. It is then close to
binary. That is a loss of graded information, which the linear probe sees.

**Which end is not determined by the proposition.** The toy model below pins near z ≈ 0 at the
default initialization scale and near saturation when the backbone's initial output scale is
tripled. The CNNs pin at saturation. The proposition explains why the *middle zone empties*.
Saturation is how that happens in our backbones, and sech² then makes it hard to leave
(`prop:atten` is the other, cross-factor half).

Two connections to existing results:
- This predicts the star > window result. Concentrating overlap on a few inputs means fewer
  coordinates fall under high-order constraints.
- It also explains why an anti-saturation penalty helps but is not the whole story. The penalty
  moves mass out of one end, but the budget still pushes the coordinates out of the middle.

## Retrospective check on the archived CNNs

CIFAR-10/4,000, seeds 0–4, selected lr; `scripts/pinning_stats.py` → `results/pinning_stats.csv`.
M = P(0.1 ≤ |tanh z| ≤ 0.9), Z0 = P(|tanh z| < 0.1).

| head | order | S | Z0 | M | E\|cos\| |
|---|---|---|---|---|---|
| no CNOT | 1 | 0.013 | 0.147 | 0.840 | 0.607 |
| default ring | 2.5 | 0.295 | 0.075 | 0.630 | 0.734 |
| last ring | 5.25 | 0.494 | 0.049 | 0.457 | 0.817 |
| default products | 2.5 | 0.412 | 0.060 | 0.528 | 0.783 |
| last-ring products | 5.25 | 0.604 | 0.043 | 0.352 | 0.860 |

M falls and E|cos| rises with order, in both the circuits and the classical product heads, as the
proposition predicts. The mass goes to the saturated end (S up, Z0 down). The linear head's S
(0.32) is not comparable, because nothing bounds its z scale.

## Toy model

A 32 → 64 → 8 MLP on a 10-class Gaussian mixture, trained through classical product heads with the
paper's supports. 5 seeds, `scripts/toy_order_model.py` → `results/toy_order_model.csv`.

| head | order | train | probe | S | M | E\|cos\| |
|---|---|---|---|---|---|---|
| order 1 | 1 | 98.3 | 78.2 | 0.22 | 0.68 | 0.71 |
| pairs-last | 1.5 | 97.0 | 77.4 | 0.16 | 0.68 | 0.73 |
| default | 2.5 | 95.5 | 75.8 | 0.09 | 0.58 | 0.79 |
| star3 | 3 | 95.0 | 76.8 | 0.12 | 0.56 | 0.78 |
| win4 | 4 | 92.9 | 76.5 | 0.07 | 0.47 | 0.84 |
| last ring | 5.25 | 87.3 | 74.5 | 0.06 | 0.47 | 0.84 |

What it reproduces:
- the middle-zone and E|cos| signature;
- a probe cost that rises with order;
- star3 above default at higher order.

What it does not reproduce:
- the size of the cost (3.7 points from order 1 to 5.25, against 13.5 in the CNN);
- saturation, since it pins near z ≈ 0 instead;
- monotonicity (win4 is above default).

Seed SDs are 2–4 points, because the data differ by seed; paired differences have not been computed
yet. **Reading:** the budget mechanism alone gives a modest cost. The large CNN cost needs the
pinning to land at saturation, where sech² blocks recovery. That is exactly what tonight's
straight-through test (cifar_st) and induced-saturation test (cifar_satup) measure.

**Paired gaps to order 1 (probe points, 5 seeds), at two initial bottleneck scales**
(`results/toy_order_model_scale3.csv` for ×3):

| head | ×1: gap | ×1: S | ×3: gap | ×3: S |
|---|---|---|---|---|
| pairs-last | −0.7 ± 0.5 | 0.16 | −0.5 ± 0.5 | 0.24 |
| default | −2.4 ± 1.0 | 0.09 | −2.3 ± 0.7 | 0.37 |
| star3 | −1.4 ± 0.7 | 0.12 | −0.9 ± 1.3 | 0.32 |
| win4 | −1.6 ± 1.4 | 0.07 | −8.9 ± 13.3 (one collapsed seed) | 0.50 |
| last ring | −3.7 ± 3.1 | 0.06 | −2.5 ± 2.1 | 0.32 |

At ×3 the toy saturates more with order, as the CNN does. Even so, the default gap is unchanged
(−2.4 vs −2.3), and only win4 gets worse, through one unstable seed. **In the toy, then, landing
the pinning at saturation does not by itself enlarge the cost.** The cost follows the middle-zone
budget at either end. This is a caveat for the saturation account: in the CNN, saturation may be how
the budget shows up rather than an extra cause. The straight-through test (cifar_st) decides this
for the CNN. If removing sech² gives little recovery, the budget framing, not the trap, is the
right main-text story.

## Next
1. Read cifar_st and cifar_satup against this. If both are weak, lead the main text with the
   signal-budget proposition and present saturation as its form in CNNs.
2. A bound without independence, for example via the readouts' joint distribution or a per-readout
   version, since a reviewer will ask about the independence assumption.
3. Make the toy closer to the CNN (BatchNorm before the bottleneck, more classes per readout) only if
   the CNN tests leave the question open.
