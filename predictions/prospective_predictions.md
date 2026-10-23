# Prospective predictions (verbatim from the project log)

The paper (Sec. 5.3, App. F, App. M) states that the predicted ordering for the MNIST
replication and the predictions and pass criteria for the second circuit family were written
before those runs. They were written in the project's working log (`findings.md`), which is not
under version control. This file copies those entries **verbatim and unedited** so they have a
verifiable record; its commit date is the first verifiable timestamp, not the date of writing.
Outcomes are deliberately left out (they are in the paper). Later entries (the two encoding arms, the gradient-floor test, the 12-qubit replication, the mechanism diagnostics and the mechanism test) were
committed and pushed before their runs started, so their commit date is itself the record.

Corroborating, not conclusive: the earliest file in `results/protocol_v2/cifar_family/` was
created 2026-10-20 22:16:55 local time, after the 22:15 time stamped on the second-family entry;
the `mnist_order` runs ran 2026-10-18 16:53–17:10.

---

## Pre-registered predictions for review-loop round 1 experiments (2026-10-18, written before any run)
Context: auto-review-loop run_20261018_8a770126, round 1 (5/10). Overlap = mean over readout pairs k≠l of |S_k ∩ S_l|.

**(A) `cifar_support`: classical product heads with designed supports (CIFAR-10/4000, lr {3e-3,1e-2,3e-2}, 3 seeds).**

| cell | supports | order | overlap |
|---|---|---|---|
| win2 | {k,k+1} | 2 | 0.29 |
| star2 | {1,2},{0,1},…,{0,7} | 2 | 0.82 |
| win3 | {k,k+1,k+2} | 3 | 0.86 |
| star3 | {1,2,3},{0,1,2},{0,1,3},…,{0,1,7},{0,2,3} | 3 | 1.71 |
| win4 | {k,…,k+3} | 4 | 1.71 |

- If order drives the probe: win2 ≈ star2 > win3 ≈ star3 > win4.
- If overlap drives it: win2 > star2 ≈ win3 > star3 ≈ win4.
- Decisive contrasts: star2 vs win3 (equal overlap, order 2 vs 3) and star3 vs win4 (equal overlap, order 3 vs 4) test order. win2 vs star2 and win3 vs star3 (equal order) test overlap.

**(B) `cifar_long`: 160 epochs (2× budget) for no-CNOT, default and last-ring VQC at lr 1e-2, 3 seeds.**
- Prediction, if the cost is representational: the probe ordering noent > default > last-ring persists. The default − noent probe gap stays ≥ 2 points. Clean-train accuracy rises for all three.

**(C) `mnist_order`: MNIST/1000, quantum last-ring, quantum pairs-last, classical cosprod (default products), classical cosprod_last; lr {3e-3,1e-2,3e-2}, 3 seeds.**
- Prediction, ordered by probe: noent (order 1, 96.5) ≥ pairs-last (1.5) ≥ default (2.5, 95.1) > last-ring (5.25).
- cosprod falls within ~1 point of the default VQC; cosprod_last falls at or below last-ring.

---

## Second circuit family (out-of-sample test of product order), predictions written 2026-10-20 22:15 before any run
**Design:** six new CNOT layouts outside the ring family, chosen only by their zero-rotation product order (computed from the gates; tests/test_identities.py::test_second_family_orders). Same protocol as the headline cells: CIFAR-10/4,000, small CNN, 80 epochs, head lr {3e-3, 1e-2, 3e-2} × seeds 0–2, one edge step if a boundary lr is selected, then seeds 3–4 at the selected lr. Common linear probe, plus the MLP probe.

| circuit | gates (after every layer unless "last") | order | overlap | predicted probe |
|---|---|---|---|---|
| star | CNOT(0, q) for every q (cancels at zero rotations) | 1 | 0.00 | 73.5 |
| chain | CNOT(q, q+1), q = 0..6 (open ring) | 1.5 | 0.14 | 71.9 |
| star_in_last | CNOT(q, 0), last layer only (readout 0 has order 8, the rest order 1) | 1.875 | 0.25 | 70.7 |
| ladder2_last | CNOT(q, q+2 mod 8), last layer only | 3 | 0.93 | 67.0 |
| brick | brickwork: pairs from q = 0 after even layers, from q = 1 after odd ones | 3.5 | 1.29 | 65.4 |
| chain_last | open chain, last layer only | 4.5 | 3.00 | 62.2 |

- **Prediction source:** a linear fit of probe on order over the 7 existing VQC circuits (seeds 0–2): probe = 76.69 − 3.22 × order, residual SD 1.69.
- **Primary criterion (pre-registered):** the Spearman ρ between order and mean probe (5 seeds) over the six new circuits is ≤ −0.8.
- **Secondary:** at least 4 of 6 new circuits within ±4 points of the fitted prediction.
- **Kill:** ρ > −0.5, or ≥ 3 circuits off by > 4 points. Then "order transfers beyond the ring family" fails, and the paper says the descriptor is validated only within the ring family.
- **Specific risks I expect:**
  - star_in_last's skew: one order-8 readout plus seven order-1 readouts. The mean order understates its cost if one dead readout matters little, or overstates it. Prediction: it lands within ±4 of 70.7 anyway, slightly above it.
  - star (order 1) behaves like pairs (cancelling, order 1, ~73).

---

## Encoding arm (monotone `half` encoding), predictions written 2026-10-21 14:45 before any run
**Design:** the angle map changes from π·tanh z to (π/2)·tanh z + π/2, so each zero-rotation factor becomes cos φ = −sin((π/2) tanh z): 0 at z = 0, where the π·tanh factor is 1. Every R_y encoding keeps the zero-rotation identity, so product order is unchanged (checked to 1e-15 for all six layouts). Six layouts spanning orders 1–5.25: no CNOT (1), pairs-last (1.5), default ring (2.5; already run in cifar_circuit, seeds 0–2), ladder-2-last (3), chain-last (4.5), ring-last (5.25). Same protocol as the second family: CIFAR-10/4,000, small CNN, 80 epochs, head lr {3e-3, 1e-2, 3e-2} × seeds 0–2 (grid `cifar_half`), one edge step if a boundary lr is selected (`cifar_half_ext`), then seeds 3–4 at the selected lr for all six (`cifar_half_seeds`). Common linear probe, plus the MLP probe.

**Attenuation measured before any new run** (results/attenuation.csv, existing half default-ring checkpoints, seeds 0–2): mean E[cos² φ] = 0.705 under `half` vs 0.632 under π·tanh (default ring). Scaling the ring-family slope (3.22 points per order) by ln 0.705 / ln 0.632 = 0.761 gives 2.45 points per order.

| layout | order | predicted probe (attenuation-scaled slope 2.45) | unscaled slope 3.22 |
|---|---|---|---|
| no CNOT | 1 | 73.9 | 75.0 |
| pairs, last | 1.5 | 72.6 | 73.4 |
| ring (default) | 2.5 | 70.2 (anchor: observed, 3 seeds) | 70.2 |
| ladder-2, last | 3 | 69.0 | 68.6 |
| chain, last | 4.5 | 65.3 | 63.8 |
| ring, last | 5.25 | 63.5 | 61.3 |

- **Primary criterion:** Spearman ρ between order and mean probe (5 seeds) over the six layouts ≤ −0.8.
- **Secondary:** (a) no CNOT − default ring ≥ 2 probe points (paired, 5 seeds; point prediction 3.7); (b) at least 4 of 6 layouts within ±4 points of the attenuation-scaled prediction.
- **Exploratory (weak, stated in advance):** the fitted probe-on-order slope over the six layouts is closer to −2.45 than to −3.22. The two differ by less than the likely slope uncertainty, so this can support but not establish the attenuation account.
- **Kill:** ρ > −0.5. Then the paper says the order ranking is specific to the π·tanh map.
- **Risks I expect:** ladder-2-last (3) vs default (2.5) are close in order and may swap; chain-last (4.5) vs ring-last (5.25) likewise. The `half` factor vanishes at z = 0, so high-order heads may train poorly early (worse than predicted) if the backbone starts near the origin.

---

## Phase-encoding arm (H then RZ(π tanh z)), predictions written 2026-10-21 18:55 before any run
**Design:** each qubit is encoded as H then RZ(φ), φ = π·tanh z, Bloch vector (cos φ, sin φ, 0). Every zero-rotation readout is a Z string and vanishes, so Eq. (1) and the R_y product order do not apply. The simulator matches an independent PennyLane circuit (tests/test_torch_vqc.py::test_phase_encoding_matches_pennylane, 1e-10), and the Pauli expansion generalizes with Z strings invisible and X or Y factors touching an input (tests/test_effective_order.py::test_phase_encoding_expansion). Same protocol as the half arm: CIFAR-10/4,000, small CNN, 80 epochs, head lr {3e-3, 1e-2, 3e-2} × seeds 0–2 (`cifar_phase`), one edge step if a boundary lr is selected (`cifar_phase_ext`), seeds 3–4 at the selected lr (`cifar_phase_seeds`). Linear and MLP probes.

**Pre-run quantity:** the effective order at the initialization distribution θ ~ U(−0.1, 0.1) (50 draws; results/effective_order_init_phase.csv; R_y control results/effective_order_init_ry.csv, where it reproduces the product order). The six layouts were chosen so that it disagrees with the R_y product order (Spearman between the two −0.23 over these layouts):

| layout | effective order at init, phase (mean, SD over draws) | R_y product order | point prediction from init order (76.69 − 3.22 × order) | point prediction from R_y order |
|---|---|---|---|---|
| no CNOT | 1.00 (0.00) | 1 | 73.5 | 73.5 |
| ring, last | 1.51 (0.50) | 5.25 | 71.8 | 59.8 |
| chain, last | 1.64 (0.50) | 4.5 | 71.4 | 62.2 |
| chain | 2.98 (0.39) | 1.5 | 67.1 | 71.9 |
| ring (default) | 4.44 (0.17) | 2.5 | 62.4 | 68.6 |
| star | 4.47 (1.78) | 1 | 62.3 | 73.5 |

- **H_eff (effective order computed for the actual encoding predicts the cost):** Spearman ρ(init effective order, mean probe over 5 seeds) ≤ −0.8.
- **H_layout (the cost is a property of the CNOT layout, as ranked by the R_y order):** ρ(R_y order, mean probe) ≤ −0.8.
- At most one can pass. If neither does (both ρ > −0.8), the effective order at initialization does not carry over to an encoding whose initial readouts vanish, and the paper says so.
- **Secondary:** (a) the no-CNOT head is a per-coordinate sinusoid under any single-qubit encoding, so its probe lands in [72, 75] (point 73.5). (b) Count of layouts within ±4 points of each column's point predictions (reported for both; no threshold).
- **My expectation, stated in advance:** about 99% of the readout weight is invisible at initialization, so training must move the rotations far from the initial regime. I expect the init effective order to be at best a moderate predictor (ρ between −0.5 and −0.8) and H_layout to fail.
- **Planned post hoc analysis (labelled as such):** effective order at the trained rotations under the phase encoding, against probe accuracy.

---

## Gradient-floor test of the attenuation mechanism, predictions written 2026-10-21 23:41 before any run
**Design:** classical product heads (forward = the zero-rotation circuit's cosine products, then Linear) on the default supports (order 2.5) and the last-ring supports (order 5.25). The backward pass replaces each path's derivative d feat_k / d c_j = P_kj (product of the other factors in S_k) by sign(P_kj) · max(|P_kj|, τ): τ = 0 is the true gradient (the existing coupled heads: default 68.8 ± 1.6 probe, 86.0 clean train; last-ring 54.8 ± 3.4, 56.1; 5 seeds, lr 1e-2), τ = 1 removes all magnitude attenuation while keeping each path's sign, τ = 0.3 changes only paths attenuated below 0.3. Unlike the earlier decoupling test (which set P_kj = 1 and lost its sign; training collapsed), the sign is kept. Forward values are unchanged and τ = 0 reproduces the true gradient exactly (tests/test_identities.py::test_grad_floor_head). Heads cosprod_fl03, cosprod_fl1, cosprod_last_fl03, cosprod_last_fl1; CIFAR-10/4,000, small CNN, 80 epochs, head lr {1e-3, 3e-3, 1e-2, 3e-2} × seeds 0–2 (`cifar_floor`), one edge step if a boundary lr is selected (`cifar_floor_ext`), seeds 3–4 at the selected lr (`cifar_floor_seeds`). Linear probe (primary) and MLP probe. Order-1 reference: the cos head, 73.3 probe.

**Intervention strength measured before any run** (results/grad_floor_stats.csv; coupled checkpoints, validation embeddings, per-sample dL/dz): at initialization the floors change nothing (bottleneck near 0, every cosine ≈ 1; mean cosine with the true gradient 1.000). At the end of coupled training, τ = 1 gives mean cosine 0.974 (default) and 0.914 (last-ring), median norm ratio 1.10 and 1.28; τ = 0.3 changes 16% (default) and 34% (last-ring) of paths, mean cosine 0.998 and 0.975. So the manipulation is mild at both ends of the coupled trajectory and can act only in between.

- **H_att (gradient-magnitude attenuation causes the representation cost):** τ = 1 raises the probe over the coupled head, by ≥ 2 points on the default supports and ≥ 4 points on the last-ring supports (paired by seed, 5 seeds; mean difference positive in ≥ 4 of 5 seeds), with τ = 0.3 between τ = 0 and τ = 1.
- **Training guard:** a support's result counts only if τ = 1 keeps clean training accuracy within 10 points of the coupled head (86.0 / 56.1); otherwise it is inconclusive, as the earlier decoupling test was.
- **Refutation:** training intact and τ = 1 − τ = 0 below +1 point on both supports. Then gradient-magnitude attenuation is not the main cause, and the paper says so (with the caveat that the manipulation is mild at the measured states).
- **My expectation, stated in advance:** given the mild measured manipulation, I expect at most a small recovery on the default supports (< 2 points) and a larger but partial one on the last-ring supports, i.e. H_att partly supported at most. I would not be surprised by a null.

---

## Width replication at 12 qubits, predictions written 2026-10-22 20:30 before any run
**Design:** the headline comparison with a 12-dimensional bottleneck and 12 qubits (one per coordinate), everything else as in the CIFAR-10/4,000 small-CNN protocol (80 epochs, 4 variational layers, π·tanh encoding). Heads: linear (lr {1e-3, 3e-3, 1e-2, 3e-2}); VQC without CNOTs, with the default rings, and with a ring after the last layer only (lr {3e-3, 1e-2, 3e-2}); seeds 0–2 (`cifar_q12`), one edge step if a boundary lr is selected (`cifar_q12_ext`), seeds 3–4 at the selected lr (`cifar_q12_seeds`). Linear probe (primary) and MLP probe. At 12 qubits the zero-rotation identity holds (tests/test_identities.py::test_zero_rotation_identity_12_qubits) with product orders 1, 3.92 (supports of size 3–5) and 7.33 (2–12), against 1, 2.5 and 5.25 at 8 qubits.

**8-qubit reference (5 seeds, linear probe):** linear 74.45, no CNOT 73.61, default 69.77, ring-last 60.11; default − no CNOT −3.84, ring-last − default −9.66.

- **Primary:** the mean probe accuracy (5 seeds) is ordered no CNOT > default > ring-last.
- **Secondary:** (a) default − no CNOT ≤ −2 points, paired, negative in ≥ 4 of 5 seeds; (b) default − linear < 0, negative in ≥ 4 of 5 seeds.
- **Point extrapolation (order only):** with the 8-qubit ring-family slope of 3.22 points per unit of order, default − no CNOT ≈ −9.4 and ring-last − no CNOT ≈ −20.4.
- **Kill:** default ≥ no CNOT in mean probe. Then the order effect does not carry over to 12 qubits, and the paper says so.
- **My expectation, stated in advance:** the ordering holds, but the gaps fall short of the order-only extrapolation (default − no CNOT between −2 and −9), because a wider bottleneck gives the backbone more room, and the 8-qubit slope was fitted at one width only.

---

## Mechanism diagnostics: bottleneck geometry as a mediator, predictions written 2026-10-23 10:54 before any computation
**Design (no training):** on the trained checkpoints, test embeddings z of each selected run (all available seeds), compute three bottleneck statistics and correlate their cell means with the cells' mean linear-probe accuracy. Cells (CIFAR-10/4,000, small CNN, selected lr), 38 in total:
- 8-qubit VQCs, π·tanh (13): no CNOT, pairs, pairs-last, default, 2-layer ring, 1-layer ring, ring-last; star, chain, star-in-last, ladder-2-last, brickwork, chain-last.
- half encoding (6) and phase encoding (6): the layouts of the two encoding arms.
- 12 qubits (3): no CNOT, default, ring-last (results/q12_probe.csv).
- classical product heads (10): cos, products of the pairs-last, default and ring-last supports, and the designed supports win2, star2, win3, star3, win4, gap4.

**Statistics (q = bottleneck width):**
- S (saturation): fraction of test coordinates with |tanh z| > 0.9.
- PRn (effective dimensionality): participation ratio (Σλ)²/Σλ² of the correlation matrix of z, divided by q.
- CPRn (class-mean dimensionality): participation ratio of the between-class covariance of standardized z, divided by min(q, 9).

**Hypotheses:**
- **H_dim (dimensional collapse):** Spearman(PRn, probe) ≥ 0.7 over the 28 quantum cells and ≥ 0.7 over the 10 classical cells. Secondary: CPRn behaves the same way; within the 13 8-qubit π·tanh cells, the residual of PRn after its own fit on order correlates ≥ 0.5 (Spearman) with the probe residual from the ring-family order fit (does dimensionality explain what order misses, e.g. brickwork?).
- **H_sat (saturation):** Spearman(S, probe) ≤ −0.7 over the 28 quantum cells and over the 10 classical cells.
- A statistic that fails either set is not a candidate mediator. Whatever passes is only a correlate; a causal intervention would be pre-registered separately.
- **My expectation, stated in advance:** H_dim holds (the bottleneck-gradient rank already falls with order); H_sat is weaker, because the encodings move the saturating region (the half and phase encodings read z differently), so pooled across encodings saturation should not track the probe well.

---

## Mechanism test (phase 2): saturation vs dimensional collapse, predictions written 2026-10-23 11:44 before any run
**Design:** classical product heads on CIFAR-10/4,000 (small CNN, 80 epochs, lr fixed at the unregularized heads' selected 1e-2): cos (order 1), default-support products (2.5), last-ring products (5.25). Each is trained with one regularizer on the bottleneck z added to the loss (quantcnn/training_utils.py::bottleneck_penalty; tested in tests/test_identities.py::test_bottleneck_penalties):
- **sat** (anti-saturation): λ · mean(relu(tanh(z)² − 0.81)), nonzero only where |tanh z| > 0.9; λ ∈ {3, 10}.
- **decor** (anti-collapse): λ · mean squared off-diagonal entry of the batch correlation matrix of z; λ ∈ {1, 3}.
Seeds 0–2 for every (head, reg, λ) (`cifar_mech2`); λ selected per (head, reg) on mean validation accuracy, never on the probe; seeds 3–4 at the selected λ (`cifar_mech2_seeds`). Linear probe (primary) and MLP probe; S and PRn on test embeddings as in the diagnostics. λ sizes were set from the penalties on the existing trained bottlenecks (sat 0.065 / 0.109, decor 0.19 / 0.38 for default / last-ring products, against training cross-entropy of roughly 0.3–1.5).

**Unregularized baselines (5 seeds, lr 1e-2):** cos probe 73.3, S 0.052, PRn 0.666; default products 68.8, S 0.412, PRn 0.476, clean train 86.0; last-ring products 54.8, S 0.604, PRn 0.278, clean train 56.1.

- **Manipulation checks:** sat lowers S of the last-ring products to ≤ 0.42 (−30% relative); decor raises their PRn by ≥ 0.05. A regularizer that fails its check is inconclusive.
- **Training guard:** clean training accuracy within 10 points of the unregularized head; otherwise that (head, reg) is inconclusive.
- **H_sat-causal (saturation causes the cost):** with sat at the selected λ, the probe rises by ≥ 3 points on the last-ring products and ≥ 1.5 on the default products (paired by seed, positive in ≥ 4 of 5 seeds), and the gain on cos is less than half the last-ring gain (specificity).
- **H_dim-causal (collapse causes the cost):** the same criteria with decor.
- **Reading:** both pass → each is sufficient to recover part of the cost; one passes → that one is the operative route; manipulation succeeds but neither passes → both are symptoms. The non-selected λ is reported as secondary.
- **My expectation, stated in advance:** decorrelation raises PRn and recovers a few points on the last-ring products (H_dim partly supported, smaller on default); the saturation penalty conflicts with what products need (factors near ±1) and helps little or hurts fit, so H_sat fails or is inconclusive.
