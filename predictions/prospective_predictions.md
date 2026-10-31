# Prospective predictions (verbatim from the project log)

The paper (Sec. 5.3, App. F, App. M) states that the predicted ordering for the MNIST
replication and the predictions and pass criteria for the second circuit family were written
before those runs. They were written in the project's working log (`findings.md`), which is not
under version control. This file copies those entries **verbatim and unedited** so they have a
verifiable record; its commit date is the first verifiable timestamp, not the date of writing.
Outcomes are deliberately left out (they are in the paper). Later entries (the two encoding arms, the gradient-floor test, the 12-qubit replication, the mechanism diagnostics, the mechanism test, its circuit confirmation and the last-ring replication) were
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

---

## Mechanism confirmation on circuits, predictions written 2026-10-23 13:42 before any run
**Design:** the anti-saturation penalty of the phase-2 test (λ · mean(relu(tanh(z)² − 0.81)) on the bottleneck z), applied to the 8-qubit VQC heads with π·tanh encoding: no CNOT (order 1, specificity control), default rings (2.5), ring after the last layer (5.25). CIFAR-10/4,000, small CNN, 80 epochs, lr fixed at the unregularized heads' selected 1e-2; λ ∈ {3, 10} × seeds 0–2 (`cifar_mech3`), λ selected per layout on mean validation accuracy, seeds 3–4 at the selected λ (`cifar_mech3_seeds`). Linear probe (primary), MLP probe; S and PRn on test embeddings.

**Unregularized baselines (5 seeds, lr 1e-2):** no CNOT probe 73.61, S 0.013, PRn 0.679, clean train 96.6; default 69.77, S 0.295, PRn 0.526, 89.2; ring-last 60.11, S 0.494, PRn 0.338, 65.6.

- **Manipulation check:** the penalty lowers S to ≤ 0.30 on ring-last and ≤ 0.18 on default (−40% relative). Otherwise inconclusive for that layout.
- **Training guard:** clean training accuracy within 10 points of the unregularized head.
- **H_sat-circuit:** at the selected λ the probe rises by ≥ 3 points on ring-last and ≥ 1.5 on default (paired by seed, positive in ≥ 4 of 5 seeds), and the no-CNOT gain is less than half the ring-last gain.
- **Refutation:** manipulation and guard pass, and the gains are below +1 on both default and ring-last. Then the phase-2 result does not transfer from the product heads to the circuits, and the paper says so.
- **My expectation, stated in advance:** the penalty transfers, with gains a little smaller than on the product heads (ring-last about +5 to +10, default about +1.5 to +3), because the trained rotations can partly route around the saturated coordinates.

---

## Last-ring replication of the saturation test, predictions written 2026-10-23 18:55 before any run
**Why:** in the phase-2 and circuit tests the last-ring heads improved clean training accuracy by 14–23 points under the anti-saturation penalty, beyond the committed guard ("within 10 points"), which made those results inconclusive as written; the guard was meant as a limit on lost fit. This replication fixes that wording and uses fresh seeds.

**Design:** CIFAR-10/4,000, small CNN, 80 epochs, lr 1e-2, seeds 5–9 (not used before), both arms on every seed: the 8-qubit VQC with a ring after the last layer only, and the classical head on its products (last-ring supports), each unregularized and with the anti-saturation penalty at the previously selected λ = 10 (no new selection). Grid `cifar_lastring_retest` (20 runs). Linear probe (primary) and MLP probe; S on test embeddings.

- **Manipulation check:** S with the penalty ≤ 0.30 (circuit) and ≤ 0.42 (products), seed-averaged.
- **Training guard (no loss of fit):** the penalized head's mean clean training accuracy is at least the unregularized head's minus 10 points. Gains of any size pass.
- **H_sat-lastring:** paired mean probe gain (penalty − unregularized, same seed) ≥ +3 points on each family, positive in ≥ 4 of 5 seeds.
- **Refutation:** check and guard pass, and the mean gain is below +1 on both families.
- **My expectation, stated in advance:** it replicates, circuit about +5 to +10 and products about +8 to +15 (seeds 0–4 gave +7.9 and +12.6).

---

## Mechanism and generality tests (night of 2026-10-26), predictions written 2026-10-26 22:00 before any run
**Why:** reviewer rounds 6–10 left two limits: the saturation mechanism is not isolated from the fit and dimensionality changes the penalty also causes, and generality is bounded (one gate family, tanh-bounded encodings, order only ever set by CNOT layouts). Plan: `PUSH_PLAN_20261026.md`.

**Common design:** CIFAR-10/4,000, small CNN, 80 epochs, the usual recipe (`_C4000`). Linear probe (primary) and MLP probe; S (fraction of test bottleneck coordinates with |tanh z| > 0.9) and PRn on test embeddings. "Paired" means same seed. **Archived baselines (5 seeds, lr 1e-2, π·tanh encoding):** no CNOT probe 73.61, S 0.013, clean train 96.6; default ring 69.77, S 0.295, 89.2; last ring 60.11, S 0.494, 65.6; default products 68.8, S 0.412; last-ring products 54.8, S 0.604. Pairs-last 71.2 (3 seeds). **Ring-family order fit** (least squares over orders 1, 1.5, 2.5, 5.25): probe = 76.62 − 3.10 · order, which gives 70.4 at order 2 and 67.3 at order 3. Code: `quantcnn/models/torch_vqc.py` (`pi_tanh_st`, `lin`, `cz`, `readout`), `CosProductHead(st_tanh=True)`, `bottleneck_penalty("satup<k>")`; tests in `tests/test_torch_vqc.py` and `tests/test_identities.py` (CZ ring, readouts and linear encoding checked against independent PennyLane circuits; zero-rotation orders; straight-through forward equality and gradient).

### (1) `cifar_st`: straight-through tanh (forward identical, backward d tanh/dz = 1), lr 1e-2, seeds 0–4
Cells: default ring, last ring, no CNOT (all `pi_tanh_st`); classical default products and last-ring products (`cosprod_st`, `cosprod_last_st`). Only the gradient at saturated coordinates changes, so this separates the sech² gradient trap from the saturated forward values.
- **Manipulation check:** S falls by ≥ 30% relative to the baseline on the default and last-ring circuits. Otherwise that cell is inconclusive (the trap was not what kept coordinates saturated).
- **Training guard:** mean clean training accuracy ≥ baseline − 10 points.
- **H_trap (the gradient trap is a route):** paired probe gain ≥ +1.5 on the default ring and ≥ +3 on the last ring (each positive in ≥ 4 of 5 seeds), and the no-CNOT gain is below half the last-ring gain. The product heads are a secondary check with the same thresholds.
- **Refutation:** check and guard pass, and the gains are below +1 on both circuits. Then the cost comes from saturated forward values, not from blocked gradients.
- **My expectation, stated in advance:** partial support. Last ring +3 to +8, default +0.5 to +2. S falls, but less than under the penalty, because the forward products still reward factors at ±1.

### (2) `cifar_cz`: CZ ring after every layer, lr {3e-3, 1e-2, 3e-2} × seeds 0–2, seeds 3–4 at the selected lr (`night26_seeds`)
CZ commutes with Z, so the circuit entangles but has zero-rotation product order **1**.
- **H_order:** CZ probe − default-ring probe ≥ +2 (paired over 5 seeds, ≥ 4 positive), and the CZ probe is within 1.5 points of the no-CNOT head.
- **H_entanglement (the cost needs entanglement, not order):** CZ probe ≤ default + 1.
- **Retrospective, not a criterion:** effective order at the trained rotations (`scripts/effective_order.py`).
- **My expectation, stated in advance:** H_order holds. CZ probes 72–73.5, slightly below no CNOT, because nonzero rotations let CZ couple the X components (effective order a little above 1).

### (3) `cifar_lin`: unbounded encoding φ = z (no saturating map), 4 layouts × lr {3e-3, 1e-2, 3e-2} × seeds 0–2, seeds 3–4 at the selected lr
Layouts: no CNOT (order 1), pairs-last (1.5), default ring (2.5), last ring (5.25). Identity (1) still holds with cos z_j, so the attenuation route remains but there is no sech² trap.
- **Statistic:** the least-squares slope of the mean probe on order across the 4 layouts, compared with the π·tanh slope (−3.10).
- **H_sat-route (saturation carries the order cost):** |lin slope| < 1.55 (half the π·tanh slope).
- **H_order-general (the order cost does not need a saturating encoding):** |lin slope| ≥ 1.55 and the 4 means are ordered by order (at most one adjacent swap).
- Between these, or with a selected lr at a grid edge for any layout: inconclusive until that layout's grid is extended.
- **Caveat recorded in advance:** with φ = z, a linear probe on z faces wrapped (periodic) features that could disadvantage every layout. The slope, not the level, is the criterion, and the MLP probe is reported alongside.
- **My expectation, stated in advance:** H_order-general, with a slope of 60–90% of the π·tanh slope. The attenuation by the other cosines is still there.

### (4) `cifar_readout`: order without entanglement, CNOT-free circuit measured with ⟨Z_kZ_{k+1}⟩ (order 2) or ⟨Z_kZ_{k+1}Z_{k+2}⟩ (order 3), lr {3e-3, 1e-2, 3e-2} × seeds 0–2, seeds 3–4 at the selected lr
Without entangling gates each qubit evolves alone, so these readouts are exactly products of 2 or 3 per-coordinate sinusoids at every rotation. Their order is exact throughout training, not only at zero rotations.
- **H_order-observable:** mean probe ordering no CNOT (73.61) > zz > zzz, with each mean within 3 points of the ring-family fit (70.4 at order 2, 67.3 at order 3).
- **H_entanglement-needed:** zz and zzz both within 1 point of no CNOT.
- **My expectation, stated in advance:** the ordering holds. zz is near or slightly above the fit (about 70–72), and zzz is about 66–69.

### (5) `cifar_satup`: induced saturation in the order-1 (no-CNOT) head, penalty λ · mean over coordinates 0..k−1 of relu(0.9025 − tanh(z)²), k ∈ {2, 4}, λ = 10, lr 1e-2, seeds 0–4
This manipulates saturation in the other direction: it is induced in a head that has no order cost.
- **Manipulation check:** S ≥ 0.20 (k = 2) and ≥ 0.45 (k = 4), seed-averaged.
- **Reference:** the probe cost the ring family shows at the same S, interpolated linearly in S through (0.013, 0), (0.295, −3.84), (0.494, −13.50). This is −3.2 at S = 0.25 and −13.7 at S = 0.50. For each k, ratio = (paired probe drop vs the no-CNOT baseline) / (reference drop at the observed S).
- **H_sat-sufficient:** ratio ≥ 0.5 for both k. Saturation alone accounts for most of the ring family's cost at matched S.
- **H_order-beyond-saturation:** ratio < 0.25 for both k. Saturating coordinates without high order costs little, so order acts through more than saturation.
- **My expectation, stated in advance:** k = 2 near the reference (ratio about 0.5–1), and k = 4 with a ratio of about 0.3–0.6. Two saturated coordinates of eight lose little, and four lose more, but less than the last ring's attenuation plus saturation.

**Selection and seeds:** the lr for (2)–(4) is selected on mean validation accuracy over seeds 0–2, never on the probe. Seeds 3–4 run at that lr only. Fixed-lr cells (1) and (5) use the archived heads' selected 1e-2. No new seeds are added after the outcomes are seen without stating it.

## Escape-only straight-through and training dynamics, predictions written 2026-10-27 23:20 before any run
**Why, disclosed:** this redesign follows a result already seen. On the night of 2026-10-26 every `pi_tanh_st` circuit cell diverged (validation 18–25%, backbone gradient norm up to about 18,000), so those cells are inconclusive under the training guard written above. The cause is the β·sin φ term of a circuit readout: once sech² is removed, a saturated coordinate gets a constant outward gradient while its forward value stays flat. The product-head ST cells did not diverge, and their head test accuracies were seen before writing this section: default products 65.6 (baseline 67.1), last-ring products 54.0 (baseline 48.9). Their probes have not been seen.

**Design (`cifar_esc`, 25 runs):** encoding `pi_tanh_esc` / heads `cosprod_esc`, `cosprod_last_esc`. The forward pass is π·tanh z, unchanged. Backward, d tanh/dz = 1 when the descent step moves z toward 0 (g·z > 0) and sech² otherwise, per sample and coordinate. Saturated coordinates can therefore leave the trap, but nothing pushes them further out than in ordinary training. Cells: default ring, last ring, no CNOT, default products, last-ring products. lr 1e-2, seeds 0–4, CIFAR-10/4,000, the usual recipe. Baselines and paired comparisons are the same as in section (1) above.
- **Manipulation check, training guard, H_trap and Refutation:** the same wording and thresholds as section (1). Manipulation: S falls by ≥ 30%. Guard: clean train ≥ baseline − 10. H_trap: paired gain ≥ +1.5 on the default ring and ≥ +3 on the last ring, each positive in ≥ 4 of 5 seeds, with the no-CNOT gain below half the last-ring gain. Refutation: check and guard pass and both gains are below +1.
- **Secondary:** product heads, with the same thresholds. Also reported is `cosprod_*_esc` against `cosprod_*_st` (paired), which asks whether one-sided escape keeps the plain ST gain.
- **My expectation, stated in advance:** weaker than the plain ST product heads. S falls, though less than 30% on the default ring, which would make that cell inconclusive. Last ring +1 to +5, default 0 to +1.5. The toy model (THEORY_NOTES_20261026.md) leads me to expect the budget, not the trap, to carry most of the cost.

**Dynamics (`cifar_dyn`, 9 runs, plan A5):** no CNOT, default ring and last ring, lr 1e-2, seeds 0–2, checkpoints every 5 epochs. Retrospective-style measurement on new runs: at each checkpoint, S, M (middle-zone fraction), PRn and the linear probe on the frozen backbone.
- **H_precede (saturation mediates):** on the last ring, the epoch at which S first reaches half its final value comes no later than the epoch at which the probe gap to no CNOT (same seed, same epoch) first reaches half its final value, in ≥ 2 of 3 seeds.
- **H_budget (middle-zone emptying comes first):** M's half-way drop (relative to no CNOT) precedes S's half-way rise.
- These are not exclusive. Both are reported, along with the curves.

## Outcomes: mechanism and generality tests (scored 2026-10-28 against the sections above)
Linear probe on test, lr 1e-2 (selected in the middle of the grid for every tuned cell), 5 seeds unless noted. Paired = same seed against the archived baselines (8-qubit sources only: `backbone`, `circuit`, `entangle`, `cosprod`, `seeds` probe files; the `cifar_dyn` reruns reproduce them to 0.01).

| Test | Result | Verdict |
|---|---|---|
| (2) CZ ring | 73.17 vs default 69.77: paired +3.40, 5/5 positive; 0.44 below no CNOT (73.61) | **H_order passes**; H_entanglement fails |
| (4) readouts | zz 70.29 (fit 70.4), zzz 67.38 (fit 67.3); 73.61 > 70.29 > 67.38 | **H_order-observable passes** |
| (3) φ = z | 72.54 / 71.67 / 69.80 / 63.71 at orders 1 / 1.5 / 2.5 / 5.25; slope −2.10 (MLP probe −2.02); ordered | **H_order-general passes** (68% of the π·tanh slope; predicted 60–90%) |
| (1) plain ST, circuits | diverged (probe 23–31) | **inconclusive** (guard fails) |
| (1) plain ST, product heads | default 67.84 (−0.95 vs 68.79); last ring 60.55 (+5.73) | secondary: last ring passes, default fails |
| escape-only ST, circuits | S → 0.000–0.001 on all; default +1.48 (5/5), last ring +6.83 (5/5), no CNOT +0.89 (< half of 6.83) | check and guard pass; **H_trap not met as written** (default 0.02 below +1.5); last-ring criterion and specificity pass; not refuted |
| escape-only ST, product heads | default +2.72 (5/5), last ring +11.34 (5/5) | secondary: **both pass** |
| (5) induced saturation | k = 2: 59.83 ± 22.2, S 0.403, ratio 1.52; k = 4: 61.50 ± 10.1, S 0.529, ratio 0.80 (reference extrapolated past S = 0.494) | **H_sat-sufficient passes as written, but confounded**: see below |

**Induced saturation, the confound.** No training guard was written for (5). The penalty damaged fit: clean train 71.5 (k = 2) and 72.9 (k = 4) against 96.6 for the baseline, so both would fail the "baseline − 10" guard used in (1). One k = 2 seed (s1) never trained: its backbone gradient fell to ~1e-3 by epoch 10 and the head stayed at chance (probe 20.5). Without that seed (not a committed exclusion), the k = 2 drop is −3.75 at S = 0.274, where the reference is −3.56: ratio 1.06, with clean train 86.7. (Corrected 2026-10-28: an earlier version of this line gave −3.94 and 0.43, computed against the reference at the all-seed S.) This test therefore cannot separate saturation from lost fit, and it will be reported that way. A rerun would need a guard and a smaller λ, pre-registered as new.

**Reading.** Order carries a probe cost that does not need CNOTs (readouts, CZ), entangling gates (readouts) or a saturating encoding (φ = z). Removing the saturation trap (escape-only ST) removes saturation entirely and recovers about half of the last-ring cost (+6.8 of 13.5) but little of the default ring's (+1.5 of 3.8). With S ≈ 0, the escape-only heads still order by order: 74.50 / 71.25 / 66.93. Saturation therefore amplifies the cost at high order, and the remainder is order itself, as the signal-budget proposition (THEORY_NOTES_20261026.md) and the toy model predict. Against my stated expectations: the escape-only test came out stronger than expected; CZ (73.17), zz, zzz and the φ = z slope landed inside their expected ranges; induced saturation is uninterpretable rather than right or wrong.

**Dynamics (`cifar_dyn`, 3 seeds; `scripts/dynamics_probe.py` → `results/dyn_curves.csv`).** The reruns reproduce the archived per-seed probes exactly. Seed means:

| epoch | no CNOT S | default S | last ring S | default gap | last-ring gap | last-ring M (no CNOT M) |
|---|---|---|---|---|---|---|
| 5 | 0.44 | 0.67 | 0.77 | −3.2 | −8.4 | 0.21 (0.51) |
| 10 | 0.23 | 0.56 | 0.68 | −7.9 | −14.3 | 0.30 (0.69) |
| 20 | 0.12 | 0.42 | 0.59 | −7.9 | −18.8 | 0.38 (0.80) |
| 40 | — | 0.34 | 0.48 | −4.9 | −17.2 | 0.48 (0.86) |
| 80 | 0.02 | 0.30 | 0.46 | −3.6 | −12.7 | 0.48 (0.85) |

- **H_precede and H_budget are not applicable as written.** Both assumed S *rises* with training. Instead, every head saturates early (S ≈ 0.4–0.8 by epoch 5), and the heads then differ in how far they *desaturate*: no CNOT to 0.02, default to 0.30, last ring to 0.46. All half-way epochs fall at the first checkpoint (epoch 5), so the 5-epoch resolution cannot order them. Inconclusive, not refuted.
- **Descriptive reading (not pre-registered).** The cost comes from failing to leave an early saturated state, not from being driven into saturation. That fits the escape-only result: allowing inward steps removes saturation entirely (S → 0). The probe gap peaks at epochs 15–20 (−7.9 / −18.8) and then partly closes as S slowly falls, so the endpoint gap is a snapshot of an incomplete escape. The middle-zone deficit appears at the same time and persists after S stops falling (last ring M 0.48 vs 0.85), which is where the signal-budget component lives.

## Escape-only replication and dense early dynamics, predictions written 2026-10-28 before any run
**Why, disclosed:** on seeds 0–4 the escape-only default-ring gain was +1.48, 0.02 below the committed +1.5. That outcome stands as scored above and is not re-scored. This is a separate test on fresh seeds, chosen because the margin was this small. The dense-dynamics run follows the finding that every head was already saturated at the first 5-epoch checkpoint.

**(A) `cifar_esc_rep`: escape-only straight-through on fresh seeds 5–9.** Same design as the escape-only section above (CIFAR-10/4,000, lr 1e-2, π·tanh forward). Both arms run on seeds 5–9 for the default ring and no CNOT. The unregularized last-ring arm comes from `cifar_lastring_retest` (same seeds and recipe: 59.49, 64.40, 61.89, 52.78, 61.66), so only its escape-only arm runs here. 25 runs.
- **Manipulation check and guard:** as before (S down ≥ 30%; clean train ≥ baseline − 10).
- **H_trap-rep:** on seeds 5–9 alone, the paired gain is ≥ +1.5 on the default ring and ≥ +3 on the last ring, each positive in ≥ 4 of 5 seeds, and the no-CNOT gain is below half the last-ring gain.
- **Refutation:** check and guard pass, and both circuit gains are below +1.
- **Secondary, reported but not a criterion:** the pooled 10-seed paired gains.
- **My expectation, stated in advance:** the last-ring and specificity parts pass again. For the default ring my estimate is a gain of about +1.5 ± 1. That is roughly an even chance of clearing +1.5, so either outcome is plausible.

**(B) `cifar_dyn2`: checkpoints after each of epochs 1–10 (and every 5 after).** Same runs as `cifar_dyn` (no CNOT, default and last ring, seeds 0–2; training is deterministic, so epochs 5, 10, … reproduce `cifar_dyn`). Measured per checkpoint: linear probe, S, M (`scripts/dynamics_probe.py`).
- **H_trap-timing:** for the last ring, the saturation gap S_last − S_noent reaches ≥ 0.10 no later than the epoch at which the probe gap (noent − last, same seed and epoch) first reaches half its maximum over all checkpoints, in ≥ 2 of 3 seeds.
- **Recorded as not separable:** M ≈ 1 − S − Z0, and Z0 is small, so an M-before-S ordering cannot be told apart from S. No M hypothesis is set.
- **Descriptive:** the epoch of peak S for each head. Whether the no-CNOT head ever reaches the ring heads' S, or separates from the first epoch.
- **My expectation, stated in advance:** the S gap opens at epochs 1–2, before the probe gap reaches half its peak (around epoch 5–10). The no-CNOT head peaks lower (S ≈ 0.5) than the last ring (≈ 0.8).

**Corrections from the zero-context claim audit (2026-10-28).** These change no verdict above. The probe gap peaks at epoch 15 for the default ring and at epoch 35 for the last ring (−19.0), not "epochs 15–20". The epoch-5 default gap is −3.15. The largest plain-ST backbone gradient norm was 2.4×10⁴, not about 18,000. The failed satup seed's probe is 20.5% while its head is at chance. The escape-only last-ring products head's mean S is 0.0044, with a single-run maximum of 0.016.

## Outcomes: escape-only replication and dense dynamics (scored 2026-10-29)
**(A) `cifar_esc_rep`, seeds 5–9, linear probe, paired:** default ring 68.30 → 71.60, gain **+3.30 ± 3.02**, 5/5 positive. Last ring 60.04 (retest arm) → 65.60, gain **+5.56 ± 5.58**, 5/5 positive. No CNOT 73.52 → 74.83, gain **+1.31 ± 0.40**, which is below half the last-ring gain (2.78). Manipulation passes: S 0.242 → 0.0001 on the default ring, and ≤ 0.0006 for every escape head. Guard passes: clean train rises in every cell. **H_trap-rep passes on every criterion.** The MLP probe gives +2.54 (default) and +1.30 (no CNOT). Secondary, pooled over 10 seeds: default +2.39, last ring +6.19, no CNOT +1.10. My expectation (default ≈ +1.5 ± 1) was too low.

**(B) `cifar_dyn2`, checkpoints after epochs 1–10:** at epochs 5 and 10 the runs reproduce `cifar_dyn` (S 0.44 for no CNOT at epoch 5). **H_trap-timing passes in 3/3 seeds.** S_last − S_noent ≥ 0.10 already at epoch 1, while the probe gap reaches half its maximum at epochs 8, 2 and 9. Descriptive: the no-CNOT head peaks at S = 0.55 (epoch 2), and the default and last ring at 0.78–0.82 (epochs 1–2). The saturation gap is present from the first epoch, as expected.

## SVHN: the headline heads on a third dataset, predictions written 2026-10-29 10:00 before any run
**Why:** generality limit 2 (MNIST and CIFAR-10 only). Plan item B4.

**Data:** SVHN cropped digits. The original host was unreachable, so the data come from the Hugging Face mirror `ufldl-stanford/svhn` (`cropped_digits`). It was converted once to `data/svhn_hf/*.npz` and matches the official split sizes (73,257 / 26,032) and class counts. 4,000 stratified training images and 2,000 validation images drawn from the train split; test is the official test split. Augmentation is the CIFAR recipe without horizontal flips.

**Design:** the small CNN and the CIFAR-10/4,000 recipe (80 epochs, clip 2, backbone lr multiplier 0.5, weight decay 1e-4). Six heads: linear; VQC with no CNOT (order 1); pairs after every layer (order 1, entangling); default ring (2.5); ring after the last layer (5.25); classical default products (2.5). Tuning is per head on seeds 0–2, with lrs {3e-3, 1e-2, 3e-2} (linear {1e-2, 3e-2, 1e-1}). Selection is on mean validation accuracy. A head whose best lr is at a grid edge gets one more half-decade step on seeds 0–2 (`svhn_edge`) before selection. Seeds 3–4 then run at the selected lr (`svhn_seeds`). Primary measure: the common linear probe, over 5 seeds at the selected lr; "paired" means same seed. The MLP probe is secondary.

**Criteria:**
- **H_order-svhn:** default − no CNOT ≤ −1.0 and last ring − default ≤ −1.0 (paired, each negative in ≥ 4 of 5 seeds).
- **H_pairs-svhn (entangling but order 1):** pairs − default ≥ +1.0 (≥ 4 of 5 positive), and pairs within 1.5 of no CNOT.
- **H_gap-svhn:** linear − default ≥ +1.0 (≥ 4 of 5 positive).
- **Secondary:** the default products within 2 points of the default VQC.
- **Refutation of order on SVHN:** default − no CNOT > −0.5 and last ring − default > −0.5.
- A head still at a grid edge after the extension step is reported as such, and its comparisons are inconclusive.

**My expectation, stated in advance:** SVHN is easier for this CNN than CIFAR-10 (probe about 85–90%), so the gaps are smaller, closer to MNIST than to CIFAR. Default − no CNOT is about −1.5 to −3, last ring − default about −4 to −10, pairs within 1 of no CNOT, and linear − default about +2 to +4. The ordering holds. The default-ring criterion is the most likely to miss if the gaps compress.

## Outcomes: SVHN (scored 2026-10-29 against the section above)
**Selection (mean validation accuracy, seeds 0–2):** linear 3e-2 (interior), default ring 3e-3 (interior after extension to 1e-3), last ring 1e-2 (interior), products 3e-3 (interior after extension). **No CNOT and pairs both selected 1e-3, the lowest step tried after the one-step extension, so both are still at a grid edge.** By the pre-registered rule, every comparison involving them is inconclusive.

**Linear probe, 5 seeds at the selected lr (mean ± SD; per seed):** linear 83.00 ± 4.76; no CNOT 81.35 ± 0.92; pairs 82.83 ± 1.08; default ring 55.01 ± 22.77 (77.9, 66.2, 47.1, 19.3, 64.4); last ring 31.50 ± 27.95 (81.5, 19.1, 19.5, 19.2, 18.3); products 68.11 ± 10.10. Mean S: no CNOT 0.03, pairs 0.01, default 0.53, last ring 0.72, products 0.55.

- **H_order-svhn: inconclusive.** Last ring − default = −23.51 ± 24.33 with 4/5 negative, which meets its half. Default − no CNOT = −26.34 ± 22.17 with 5/5 negative, but no CNOT is at an edge.
- **H_pairs-svhn: inconclusive** because pairs is at an edge. Taken at face value, both parts would pass: pairs − default = +27.82 (5/5), and pairs − no CNOT = +1.48, within 1.5.
- **H_gap-svhn: passes.** Linear − default = +27.99 ± 20.32, 5/5 positive.
- **Secondary (products within 2 of the default VQC): misses.** Products − default = +13.10 ± 14.83.
- **Refutation: not met.** Both gaps are far below −0.5.
- The MLP probe agrees to within 0.4 points on every contrast.

**My expectation was wrong in kind, not just in size.** I expected compressed gaps (probe 85–90, default − no CNOT −1.5 to −3). Instead the gaps are large and come mostly from **whole-network collapse to the majority class**. A collapsed run has head and probe at 19–20%, so the backbone is dead, not just the head. Collapse at the selected lr is 0/5 for no CNOT and pairs, 1/5 for the default ring, 4/5 for the last ring, and 0/5 for products and linear. It also occurs at higher lrs for every head, linear included (1e-1). On SVHN this recipe sits close to an optimization cliff, and the higher-order heads fall off it more often. This is not the graded representational cost seen on CIFAR-10. Post hoc, not pre-registered: excluding collapsed runs, default − no CNOT is still −5.0, −14.8, −33.5, −17.0 on the four surviving seeds. The single surviving last-ring run (81.5) beats its paired default run (77.9). Linear seeds 3–4 (77.9, 77.8) sit about 8 points below seeds 0–2 without collapsing.

**What this supports:** on SVHN, order-1 heads (no CNOT, pairs) train stably, and the entangling-but-order-1 pairs head matches no CNOT. High-order heads saturate (S 0.53–0.72) and often collapse. **What it does not support:** a graded order cost of the CIFAR-10 kind on SVHN. The no-CNOT and pairs lrs are not bracketed, so the order contrasts cannot be claimed as pre-registered passes.

**Correction from the third claim audit (2026-10-29).** The `cifar_dyn` H_precede was scored above as inconclusive. As committed it says "no later than", and it is met as written in 3/3 seeds: S passes half its final value at epoch 5 in every run, and the probe gap does so at epochs 10, 5 and 10. That pass comes only from saturation already being high at the first checkpoint, so it does not establish precedence. H_budget stays inconclusive. Also: the S ≤ 0.0006 for the escape-only replication heads is a bound on seed means; single runs reach 0.0013.

## Night of 2026-10-29: SVHN extension, re-uploading, second pretrained backbone, predictions written 2026-10-29 20:26 before any run
One queue (`scripts/night29_orchestrator.py`). Stage 1 tunes all three grids (`night29a`). Stage 2 takes one half-decade edge step for every cell whose best lr (mean validation accuracy, seeds 0–2) is at an edge of the lrs run (`night29b`). Stage 3 runs more seeds at each selected lr (`night29c`). Then linear and MLP probes. "Paired" means same seed. The primary measure is the common linear probe; the MLP probe is secondary. **A cell still at an edge after stage 2 is reported as such, and its comparisons are inconclusive.**

### (A) SVHN extension (`svhn`, `svhn_seeds`)
**Why:** the SVHN outcome above left no CNOT and pairs at the lower grid edge, and the gaps were driven by collapse to the majority class. The outcome above stands and is not re-scored. This is a separate test with more lrs and more seeds.
**Design:** the same recipe and heads. New lrs on seeds 0–2: linear 3e-3; no CNOT, pairs, default ring and products 3e-4; last ring 1e-3 and 3e-4. Then one more edge step. Selection uses every lr run for the cell. Seeds 3–9 then run at the selected lr, reusing seeds 3–4 where the selection is unchanged. Scoring uses 10 seeds (0–9) at the selected lr. **Collapse** is defined as head validation accuracy < 25% (majority class 19.6%).
- **H_order-svhn2:** default − no CNOT ≤ −1.0 and last ring − default ≤ −1.0, each negative in ≥ 8 of 10 seeds.
- **H_pairs-svhn2:** pairs − default ≥ +1.0 (≥ 8/10 positive), and |pairs − no CNOT| ≤ 1.5.
- **H_gap-svhn2:** linear − default ≥ +1.0 (≥ 8/10 positive).
- **H_collapse:** collapse counts at the selected lr (of 10) satisfy no CNOT ≤ default ≤ last ring and pairs ≤ default, with last ring − no CNOT ≥ 3.
- **H_graded (a cost beyond collapse):** over the seeds where neither arm collapsed, mean default − no CNOT ≤ −1.0 and mean last ring − default ≤ −1.0. Each part is scored only if ≥ 5 such seeds exist, and is inconclusive otherwise.
- **Secondary:** products within 2 points of the default VQC.
- **Refutation of order:** default − no CNOT > −0.5 and last ring − default > −0.5.
- **My expectation:** no CNOT and pairs select 3e-4 or 1e-3, interior after the extension. The default and last ring keep collapsing in 20–60% of seeds at every lr. H_order, H_gap and H_collapse pass. H_graded passes for default − no CNOT (about −5) and is inconclusive for last ring − default, because too few last-ring seeds survive.

### (B) Re-uploading on CIFAR-10 (`cifar_reup`, `cifar_reup_seeds`)
**Why:** plan item B3. Many QML heads re-upload the input, and the paper's results cover single encoding only.
**Design:** CIFAR-10/4,000, small CNN, the standard recipe. `reupload_mode=strong`: the encoding R_Y(π tanh z) is applied before every one of the 4 variational layers. Four layouts: no CNOT, pairs after the last layer, ring after every layer (default) and ring after the last layer only. Under single encoding their product orders are 1, 1.5, 2.5 and 5.25. lr {3e-3, 1e-2, 3e-2} × seeds 0–2, an edge step, then seeds 3–4 at the selected lr. Under re-uploading the zero-rotation product identity no longer holds, so I have no derived order for these circuits. The prediction is that the single-encoding ordering carries over.
- **H_reup-order:** default − no CNOT ≤ −1.0 and last ring − default ≤ −1.0 (paired, each negative in ≥ 4 of 5 seeds).
- **H_reup-rank:** Spearman correlation between the four layouts' mean probe and their single-encoding orders ≤ −0.8 (at most one adjacent swap).
- **Refutation:** default − no CNOT > −0.5 and last ring − default > −0.5.
- **Descriptive:** each re-uploaded layout against its single-encoding counterpart, unpaired: no CNOT 73.6, default 69.8, last ring 60.1. Mean S per head.
- **My expectation:** the ordering carries over. Re-uploading lowers every layout by 1–4 points, because the repeated encoding raises the frequency of each coordinate's factor. No CNOT about 70–73.

### (C) Second pretrained backbone: MobileNetV3-Small (`cifar_mnv3`, `cifar_mnv3_seeds`)
**Why:** plan item B5. The paper has one pretrained backbone (ResNet-18).
**Design:** ImageNet MobileNetV3-Small (torchvision IMAGENET1K_V1). Its classifier is replaced by a linear map from the pooled 576 features to the 8-dimensional bottleneck. Inputs are mapped exactly as for the ResNet-18 (ImageNet normalization, 128 px). The recipe is the ResNet's: CIFAR-10/4,000, 30 epochs, backbone lr 0.1× the head's, clip 2. Heads: linear, no CNOT, pairs, default ring, last ring. lr {1e-3, 3e-3, 1e-2} × seeds 0–2, an edge step, then seeds 3–4 at the selected lr.
- **H_order-mnv3:** default − no CNOT ≤ −1.0 and last ring − default ≤ −1.0 (paired, each negative in ≥ 4 of 5).
- **H_pairs-mnv3:** pairs − default ≥ +1.0 (≥ 4/5 positive), and |pairs − no CNOT| ≤ 1.5.
- **H_gap-mnv3:** linear − default ≥ +1.0 (≥ 4/5 positive), and no CNOT within 1.0 of linear (mean).
- **Refutation:** default − no CNOT > −0.5 and last ring − default > −0.5.
- **My expectation:** the ResNet-18 pattern repeats with a weaker backbone. Linear probe about 84–88. No CNOT and pairs within 0.5 of linear, the default ring 3–8 below and the last ring 10–20 below. High-order heads may be unstable, as on the ResNet.

### Outcomes: night of 2026-10-29 (scored 2026-10-30 with `scripts/score_night29.py`)
The queue ended at 04:17 with every stage exiting 0. **Every selected lr is interior** (the edge steps added SVHN default 1e-4 and MobileNet no CNOT / last ring 3e-2, none of which won), so no comparison is inconclusive for edge reasons. Numbers are the common linear probe (MLP probe in brackets where it changes a verdict or is notably different). Paired differences: mean, sign counts.

**(A) SVHN, 10 seeds.** Selected lrs: linear 3e-2, no CNOT 1e-3, pairs 1e-3, default 3e-4, last ring 1e-3, products 3e-3.
Probe: linear 83.35±5.13, no CNOT 81.22±1.98 (S .02), pairs 82.09±1.34 (S .02), default 68.47±2.81 (S .25), last ring 50.36±9.07 (S .55), products 66.85±7.90 (S .57).
- **H_order-svhn2: PASS.** default − no CNOT −12.75 (10/10 negative); last ring − default −18.11 (10/10). [MLP −12.99, −17.72, 10/10 each.]
- **H_pairs-svhn2: PASS.** pairs − default +13.63 (10/10 positive); pairs − no CNOT +0.87.
- **H_gap-svhn2: PASS.** linear − default +14.88 (10/10).
- **H_collapse: FAIL.** No run collapsed at any selected lr (counts 0 for every head), so last ring − no CNOT = 0 < 3. At the new lower lrs the collapse that drove the first SVHN outcome disappears.
- **H_graded: PASS on both parts.** With no collapses, all 10 seeds count: default − no CNOT −12.75, last ring − default −18.11. The cost is graded, not a collapse artifact.
- **Secondary (products within 2 of default): PASS.** products − default −1.62 (6/10 positive; per-seed spread −19 to +9).
- **Refutation: not met.**
- **Against my expectation:** I expected the default and last ring to keep collapsing in 20–60% of seeds and H_graded last ring − default to be inconclusive. Both were wrong: 0 collapses, and the order holds seed by seed among non-collapsed runs.

**(B) Re-uploading on CIFAR-10, 5 seeds.** All four layouts select 1e-2 (interior).
Probe: no CNOT 68.02±2.08 (S .18), pairs 66.22±2.02 (S .22), default 58.01±4.03 (S .53), last ring 48.29±4.57 (S .61).
- **H_reup-order: PASS.** default − no CNOT −10.01 (5/5 negative); last ring − default −9.72 (5/5). [MLP −9.86, −9.60.]
- **H_reup-rank: PASS.** Spearman ρ = −1.00: the ranking is exactly the single-encoding order.
- **Refutation: not met.**
- **Descriptive (unpaired, vs single encoding):** no CNOT 68.0 vs 73.6 (−5.6), default 58.0 vs 69.8 (−11.8), last ring 48.3 vs 60.1 (−11.8). Re-uploading costs more than my 1–4 points and costs the entangled layouts about twice as much as no CNOT. Saturation rises with layout order as under single encoding.

**(C) MobileNetV3-Small, 5 seeds.** Selected: linear 3e-3, no CNOT 1e-2, pairs 3e-3, default 3e-3, last ring 1e-2 (all interior).
Probe: linear 86.08±0.63, no CNOT 85.40±0.58 (S .18), pairs 86.04±0.33 (S .15), default 84.51±0.72 (S .13), last ring 78.99±5.93 (S .33).
- **H_order-mnv3: FAIL (near miss on one part).** default − no CNOT −0.89, 5/5 negative, short of the −1.0 threshold [MLP −0.86, 4/5]. last ring − default −5.53 (5/5) passes its part.
- **H_pairs-mnv3: PASS.** pairs − default +1.53 (5/5 positive); pairs − no CNOT +0.64.
- **H_gap-mnv3: PASS.** linear − default +1.56 (5/5); |linear − no CNOT| = 0.68 ≤ 1.0.
- **Refutation: not met** (both differences are below −0.5).
- **Against my expectation:** the pattern's direction repeats, but it is much weaker than on the ResNet-18 and than I predicted (default 1.6 below linear vs my 3–8; last ring 7.1 below vs 10–20). The pretrained MobileNet saturates the default VQC far less (S .13).

## SVHN collapse: sensitivity at initialization, gradient death and order, predictions written 2026-10-30 12:54 before any run or measurement
**Why:** the first SVHN test's gaps were driven by collapse to the majority class (head validation accuracy < 25%). The open question is whether product order causes it.
**What I have already seen (disclosed):**
- **Collapse counts:** while planning this, I printed the collapse counts by head and lr for the existing SVHN runs (`runs/protocol_v2/svhn*`). Collapse there is mostly a learning-rate effect: at 1e-2 the no-CNOT head collapses 3/3, at 3e-3 only default 1/5 and last ring 1/3, at 3e-2 everything. So I make no predictions about collapse counts on those runs.
- **Per-epoch logs:** while locating the data I opened one collapsed run's per-epoch log (default, lr 1e-2, seed 0). Its backbone gradient norm fell from 0.43 at epoch 1 to 0.08 at epoch 2.
- **Not looked at:** the per-epoch logs of all other runs, and any measurement at initialization.

**Theory (lemma):** at zero rotations with independent coordinates, differentiating `eq:identity` gives, for j ∈ S_k,
E[(∂⟨Z_k⟩/∂z_j)²] = π² E[sech⁴ z_j sin² φ_j] ∏_{i∈S_k∖{j}} c_i, with c_i = E[cos² φ_i] ≤ 1.
So each input's readout sensitivity is attenuated by a product of |S_k|−1 factors c_i. Summing over readouts gives the predicted total sensitivity Ĵ = Σ_k Σ_{j∈S_k} π² E[sech⁴ z_j sin² φ_j] ∏_{i∈S_k∖{j}} c_i, computed from the empirical marginals of the initial bottleneck.

### (1) Sensitivity at initialization (`scripts/init_gradients.py`, SVHN, seeds 0–4, validation split, network built exactly as in training)
J = the sample mean of Σ_k Σ_j (∂⟨Z_k⟩/∂z_j)², measured with autograd through the circuit at its initialized rotations (U(−0.1, 0.1)). Layouts: no CNOT, pairs, default ring, last ring.
- **H_init-match:** for every layout, the seed-mean J / Ĵ lies in [0.75, 1.33].
- **H_init-order:** J(no CNOT) > J(default) > J(last ring) in every one of the 5 seeds.
- **My expectation:** c between 0.3 and 0.7; ratios within 10%; the order holds.

### (2) Collapse as gradient death (existing and new SVHN runs, per-epoch `backbone_gnorm`)
For every VQC and products run on SVHN (grids `svhn`, `svhn_seeds` and the new `svhn_collapse`), define ρ = the mean backbone gradient norm over epochs 2–10 divided by its epoch-1 value.
- **H_gdeath:** the AUC of −ρ for separating collapsed from non-collapsed runs is ≥ 0.9, and the median ρ of collapsed runs is ≤ 0.5 × the median ρ of non-collapsed runs.
- **Refutation:** AUC ≤ 0.6.

### (3) Collapse by order at an untried lr (`svhn_collapse`: lr 5e-3, seeds 0–5; no CNOT, pairs, default ring, last ring; SVHN recipe)
- **H_collapse-order:** collapse counts (of 6) satisfy no CNOT ≤ default ≤ last ring and pairs ≤ default, with last ring − no CNOT ≥ 2.
- **Refutation:** count(no CNOT) ≥ count(last ring).
- **Inconclusive** if all 24 runs collapse or none do.
- **My expectation:** no CNOT 0–1, pairs 0–1, default 2–4, last ring 3–5.

### Outcomes: SVHN collapse (scored 2026-10-30 with `scripts/score_svhn_collapse.py`)
**(1) Sensitivity at initialization: both fail.**
- **H_init-match: FAIL.** Seed-mean J/Ĵ: no CNOT 1.95, pairs 1.40, default 1.04, last ring 1.92. Only the default ring lies in [0.75, 1.33].
- **H_init-order: FAIL, reversed.** J: no CNOT 2.37, pairs 1.71, default 3.08, last ring 10.63. The predicted order holds in 0/5 seeds; the last ring has the largest J in all five.
- **My expectation was wrong on c.** The initial bottleneck sits near zero (mean |tanh z| 0.03, z SD 0.04), so c ≈ 0.98, not 0.3–0.7. With no attenuation at initialization, a higher order only adds product terms. The ±0.1 rotations are as large as the encoded angles here, which a zero-rotation model cannot capture.
- **Reading:** order does not suppress gradient signal at initialization. Any attenuation has to develop during training, as the bottleneck moves outward.

**(2) Gradient death: H_gdeath PASS.**
- **Data:** 141 SVHN VQC and products runs, 41 of them collapsed.
- **Separation:** AUC of −ρ is 0.944. Median ρ is 0.124 for collapsed runs and 0.764 for non-collapsed runs (ratio 0.16 ≤ 0.5).
- **Within each layout:** AUC 0.91–1.00 (no CNOT 1.00, pairs 0.91, default 0.94, last ring 0.95, products 0.92).
- **Caveat:** this is an association measured over the same epochs in which collapse happens. It does not show which comes first.

**(3) Collapse by order at lr 5e-3: H_collapse-order PASS (weakly), not refuted.**
- **Counts of 6:** no CNOT 2, pairs 2, default 2, last ring 5.
  - no CNOT ≤ default ≤ last ring holds.
  - pairs ≤ default holds.
  - last ring − no CNOT = 3.
- **Where the order comes from:** it is carried entirely by the last ring. The default ring does not collapse more often than no CNOT. Its non-collapsed runs are poor, though: validation accuracy 0.26–0.40, against 0.61–0.86 for no CNOT.
- **Against my expectation:** no CNOT and pairs collapsed 2/6 each, above my expected 0–1.

**Addendum (2026-10-30, after claim audit run 6; not pre-registered).**
- **What was wrong with the committed measurement:** `scripts/init_gradients.py` evaluated the untrained network with BatchNorm in eval mode. The untouched running statistics (mean 0, variance 1) shrink the bottleneck to near zero, which is not the "network as in training" the prediction described. Training uses batch statistics.
- **Rerun with batch statistics** (`--train_mode`, batches of 32; `results/init_gradients_svhn_trainbn.csv`):
  - The bottleneck is wider: mean |tanh z| 0.23, z SD 0.32, c 0.60 (inside my expected 0.3–0.7).
  - J/Ĵ by seed-mean: no CNOT 1.01, pairs 0.99, default 0.94, last ring 1.02. Per seed: 0.89–1.13.
  - J: no CNOT 23.6, pairs 23.3, default 24.8, last ring 16.7.
  - The last ring is lowest in 5/5 seeds. The default ring is above no CNOT in 4/5 seeds.
- **How this changes the scores:** under the measurement the prediction intended, H_init-match would pass and H_init-order would still fail, because default ≈ no CNOT. The scored outcomes above stand as committed (both FAIL). This rerun is reported as a post-hoc correction of the measurement.

## Decoder compensation: do trained heads keep readout magnitude? Predictions written 2026-10-30 before any measurement
**Why:** an adversarial review (kill-argument, 2026-10-30) noted that the signal budget (`prop:budget`) is conditional on E|r_k| ≥ a with a fixed as order grows. A learned linear decoder could instead let readouts shrink and scale them back up. For example, factors y/2 and ½ give r = 2⁻ᵐy, and a decoder weight of 2ᵐ recovers the logits. Nothing in the paper shows which of the two the trained models do.

**What I have already seen (disclosed):**
- Probe accuracy, saturation S, PRn and CPRn for these runs (`results/bottleneck_geometry.csv`).
- Middle-zone fractions of 0.84, 0.63 and 0.46 at orders 1, 2.5 and 5.25.
- The zero-angle checks (`results/zero_angle_check*.csv`).

**Not looked at:** any readout magnitude, decoder weight or logit scale.

**Design (`scripts/readout_magnitude.py`):**
- **Runs:** the 61 eight-qubit π·tanh runs of `results/bottleneck_geometry.csv`, group q8. That is 13 layouts at orders 1–5.25: 5 seeds each, except L1 and L2 with 3. CIFAR-10/4,000, small CNN, selected lr, best checkpoint, test split.
- **Per run:**
  - A = mean over readouts k of E|⟨Z_k⟩|, where ⟨Z_k⟩ is the trained circuit's actual output on the test embeddings.
  - W = mean over k of the Euclidean norm of decoder column k (`head.proj.weight[:, k]`).
  - L = mean over k of ‖w_k‖ · sd(⟨Z_k⟩), the logit scale each readout contributes.
- **Descriptive only:**
  - Â, the independence zero-rotation prediction ∏_{j∈S_k} E|cos φ_j|, averaged over k.
  - The bound of `prop:budget` (δ = 0.1) with a set to each readout's measured E|⟨Z_k⟩|, and the fraction of readouts of order ≥ 4 for which it is below 1 (non-vacuous).
- Cell values are seed means, and Spearman correlations are over the 13 cell means.

**Hypotheses:**
- **H_floor (premise holds):** A(ring, last) ≥ 0.5 × A(no CNOT), and every cell mean A is ≥ 0.3.
- **H_nocomp (no compensation by the decoder):** Spearman(order, W) ≤ 0.5, and W(ring, last) / W(no CNOT) ≤ 1.5.
- **Refutation (compensation operates):** Spearman(order, A) ≤ −0.8 and Spearman(order, W) ≥ 0.8. If so, the paper must say that the budget's premise does not hold in these models.
- **Otherwise:** mixed, and reported as such.

**My expectation:** saturation is how a high-order readout keeps its magnitude, because factors near ±1 give |cos| ≈ 1. So I expect A to stay roughly flat (0.4–0.7 across cells), W not to grow with order, and both H_floor and H_nocomp to pass. Â should fall less steeply than the product of per-coordinate middle-zone factors would suggest.

### Outcomes: decoder compensation (scored 2026-10-30 with `scripts/score_readout_magnitude.py`)
Data: 61 runs, 13 cells; cell values are seed means.

- **H_floor: PASS.**
  - Readout magnitude A falls only modestly with order: no CNOT 0.66, default 0.51, last ring 0.52, minimum cell 0.39 (L2).
  - A(last)/A(no CNOT) = 0.79 ≥ 0.5, and every cell is ≥ 0.3.
  - The decline is monotone, though: Spearman(order, A) = −0.88.
- **H_nocomp: PASS, and the decoder moves the other way.**
  - Decoder column norms *shrink* with order: Spearman(order, W) = −0.54, and W(last)/W(no CNOT) = 0.71 (5.72 → 4.05).
  - The logit scale per readout falls too: L goes from 4.02 to 2.31, with Spearman(order, L) = −0.91, the same as the probe's −0.91.
  - So nothing scales the shrinking readouts back up. High-order heads have smaller readouts *and* smaller decoder weights.
- **Refutation (compensation operates): not met.** A falls, but W does not rise.
- **Descriptive results:**
  - **Â (zero-rotation independence product):** it tracks A (Spearman with order −0.90) and sits 0.03–0.21 below it. The trained readouts are a little larger than independent products would give.
  - **The bound is vacuous at δ = 0.1 (against my implicit expectation).** At the measured magnitudes, `prop:budget` gives a value below 1 for **none** of the readouts of order ≥ 4, in every cell where they exist. At m = 4–8 and a = 0.3–0.5, the bound is 1.7–5.3 at δ = 0.1 and only becomes non-vacuous for δ ≥ 0.2–0.3 (0.2–0.6 at δ = 0.3). That limits it to the central zone 0.3 < |tanh z| < 0.7.
- **My expectation:**
  - A flat at 0.4–0.7: roughly right, but A is monotone in order, not flat.
  - W not growing: right, and W actually falls.
- **Reading:**
  - The premise of the budget holds empirically in these models: readout magnitude stays bounded away from 0, and the decoder does not compensate.
  - At these orders and magnitudes the proposition constrains only the central part of the encoding. It is an asymptotic statement, not a tight quantitative account of the measured middle-zone fractions.

**Addendum (2026-10-30, after kill-argument run 2; post hoc, not pre-registered).**
- **What was wrong:** the scored A is the trained circuit's actual output magnitude, not the proposition's E|r_k| for the zero-rotation products. Trained rotations can make the two diverge: with R_y(−π/2) before the last ring at tanh z = ½, every output is 1 and every product is 0. The H_nocomp result stands, because the decoder sees the actual outputs. H_floor as scored does not verify the proposition's premise.
- **Recomputed on the same 61 runs** (`results/readout_magnitude_zero.csv`, with the scored columns unchanged): the per-sample zero-rotation products r_k = ∏_{j∈S_k} cos φ_j.
  - Mean over readouts of E|r_k|: no CNOT 0.61, default 0.47, last ring 0.44, minimum cell 0.29 (brick). Spearman with order is −0.88.
  - It is within 0.04 of the product of marginals Â.
- **Readouts of order ≥ 4:**
  - E|r_k| (cell means of the smallest such readout) is 0.16–0.24, and 0.02 for the order-8 readout of star_in_last.
  - The bound with that a is non-vacuous only for δ above 0.20–0.29 (cell means), and at most 0.31 in any run.

## Saturation-gated escape-only control, predictions written 2026-10-30 before any run
**Why:** kill-argument run 3 (2026-10-30) left one point unresolved. Escape-only (`pi_tanh_esc`) drops sech² on every inward step, not only on saturated coordinates: the boost is 1/(1−t²), 1.33× at |tanh z| = 0.5 and 5.3× at 0.9. So its gain does not show that releasing *saturated* coordinates is the route.

**What I have already seen (disclosed):** the full escape-only results on seeds 0–4 (`results/esc_probe.csv`) and the plain baselines.
- Paired linear-probe gains: last ring +6.8 ± 2.2, default +1.5 ± 0.9, no CNOT +0.9 ± 0.5.
- Escape-only probes: 74.5, 71.3 and 66.9.

**Design (`cifar_escgate`, CIFAR-10/4,000, small CNN, recipe as `cifar_esc`, lr 1e-2, seeds 0–4):**
- **Variants:** two gated versions of escape-only (`GatedEscapeTanh`). The inward step drops sech² only where |tanh z| > 0.9 (`pi_tanh_escsat`, saturated-only) or only where |tanh z| ≤ 0.9 (`pi_tanh_escmid`, unsaturated-only). Elsewhere the true gradient is kept.
- **Layouts:** no CNOT, default ring and last ring. 30 runs in total.
- **Primary measure:** common linear probe (test).
- **Gains:** gain_sat and gain_mid are paired by seed against the plain baseline of the same layout and seeds, as in the escape-only test. "Full" means the escape-only gains above.

**Hypotheses:**
- **H_sat-route:** saturated-only recovers most of the last-ring gain: gain_sat(last) ≥ 4.5 (two thirds of 6.8), positive in ≥ 4 of 5 seeds.
- **H_mid-small:** unsaturated-only recovers little: gain_mid(last) ≤ 2.3 (one third of 6.8), and gain_sat(last) > gain_mid(last) in ≥ 4 of 5 seeds.
- **Secondary:**
  - default ring: gain_sat ≥ 0.75 (half of 1.5);
  - no CNOT: both gains ≤ 1.5;
  - saturation S of each variant;
  - clean training accuracy, to check whether fit tracks the gains.
- **Refutation (saturation is not the route):** gain_mid(last) ≥ gain_sat(last), or gain_sat(last) ≤ 2.3.
- **Otherwise:** mixed, and reported as such.

**My expectation:** saturated-only gives 4–7 points on the last ring with S well below the plain head's but above full escape-only's ≈ 0. Coordinates that escape below 0.9 lose the boost and may drift back. Unsaturated-only gives 0–2.

### Outcomes: saturation-gated escape-only (scored 2026-10-31 03:05 with `scripts/score_escgate.py`)
All 30 runs, the probe and the export exited 0. Linear probe; gains paired by seed (0–4) against the plain head of the same layout. S is the percentage of bottleneck coordinates with |tanh z| > 0.9; plain-head S is about 50% for the last ring.

| layout | plain | full gain | sat-only gain | mid-only gain | S% full / sat / mid | train acc plain / full / sat / mid |
|---|---|---|---|---|---|---|
| no CNOT | 73.6 | +0.89 ± 0.51 (5/5) | +0.42 ± 0.56 (4/5) | +0.83 ± 0.27 (5/5) | 0.03 / 0.33 / 0.05 | 96.6 / 98.5 / 97.4 / 98.3 |
| default ring | 69.8 | +1.48 ± 0.90 (5/5) | +1.05 ± 0.68 (4/5) | +2.05 ± 0.94 (5/5) | 0.02 / 0.47 / 0.02 | 89.2 / 93.1 / 92.9 / 94.5 |
| last ring | 60.1 | +6.83 ± 2.17 (5/5) | +5.39 ± 5.28 (4/5) | +4.69 ± 3.13 (4/5) | 0.06 / 2.35 / 0.01 | 65.6 / 78.1 / 73.8 / 75.2 |

Per-seed last-ring gains: sat-only −1.8, +6.4, +3.1, +6.5, +12.7; mid-only +5.8, +5.5, −0.1, +4.0, +8.4.
- **H_sat-route: PASS.** gain_sat(last) +5.39 ≥ 4.5, positive in 4/5 seeds. The spread is large (SD 5.3).
- **H_mid-small: FAIL.** gain_mid(last) +4.69 is far above 2.3. sat > mid holds in 4/5 seeds, but the mean difference is only 0.7.
- **Refutation: not met** (mid < sat, and sat > 2.3).
- **Secondary:** default gain_sat +1.05 ≥ 0.75 PASS. No CNOT, both gains ≤ 1.5 PASS (+0.42, +0.83).
- **Verdict: mixed.** Either gate alone recovers most of the full escape-only gain on the last ring. So the gain does not depend on boosting coordinates that are already saturated: boosting only unsaturated ones works almost as well, and their S stays near 0 (0.01%). This is consistent with the gain coming from keeping coordinates out of saturation, by either route. It does not single out releasing saturated coordinates as the mechanism.
- **Against my expectation:** I expected mid-only to give 0–2 points. It gave 4.7. I expected sat-only S to be "well below plain but above full escape-only's ≈ 0". It is 2.35%, about 20× below plain, as expected.

## Night of 2026-10-31: fresh-seed replications, predictions written 2026-10-31 02:40 before any of these runs
**Why:** the gated escape-only control, the re-uploading test and the MobileNetV3-Small test each have 5 seeds. Seeds 5–9 test whether each verdict replicates on seeds not used to set any threshold. Each is scored on seeds 5–9 alone with the thresholds already committed. The 10-seed pooled numbers are reported descriptively and are what the paper's tables would show. Every committed verdict on seeds 0–4 stands, whatever seeds 5–9 show.

**What I have already seen (disclosed):**
- Re-uploading and MobileNet seeds 0–4: all outcomes in the 2026-10-29 section.
- Escape-only on seeds 5–9 (`cifar_esc_rep`) and the plain seeds 5–9 baselines (`cifar_esc_rep`, `cifar_lastring_retest`).
- `cifar_escgate` seeds 0–4: training is still running. I have counted its finished runs but not opened any accuracy, probe or log line of them.

**Design (`night31`, 75 runs):**
- `cifar_escgate_rep`: the `cifar_escgate` design (escsat, escmid × no CNOT, default ring, last ring; lr 1e-2) on seeds 5–9.
- `cifar_reup_seeds2`: the four re-uploading layouts at their selected lr (all 1e-2), seeds 5–9.
- `cifar_mnv3_seeds2`: the five MobileNet heads at their selected lrs (linear 3e-3, no CNOT 1e-2, pairs 3e-3, default 3e-3, last ring 1e-2), seeds 5–9.
- Measure: common linear probe (test). MLP probe also run for (B) and (C).

### (A) Gated escape-only, seeds 5–9
Gains are paired by seed against the plain head on seeds 5–9: default ring and no CNOT from `cifar_esc_rep`, last ring from `cifar_lastring_retest` (unregularized arm).
- **H_sat-route-rep:** gain_sat(last) ≥ 4.5, positive in ≥ 4 of 5 seeds.
- **H_mid-small-rep:** gain_mid(last) ≤ 2.3, and gain_sat(last) > gain_mid(last) in ≥ 4 of 5 seeds.
- **Refutation:** gain_mid(last) ≥ gain_sat(last), or gain_sat(last) ≤ 2.3.
- **Secondary:** default gain_sat ≥ 0.75; no CNOT both gains ≤ 1.5.
- **My expectation:** the same as for seeds 0–4.

### (B) Re-uploading, seeds 5–9
- **H_reup-order-rep:** default − no CNOT ≤ −1.0 and last ring − default ≤ −1.0, each negative in ≥ 4 of 5 seeds.
- **H_reup-rank-rep:** Spearman correlation of the four layouts' seed-mean probe with their single-encoding orders ≤ −0.8.
- **Refutation:** both differences > −0.5.
- **My expectation:** passes; the seeds 0–4 gaps were about −10 with 5/5 signs.

### (C) MobileNetV3-Small, seeds 5–9
The committed H_order-mnv3 FAIL (default − no CNOT −0.89 against a −1.0 threshold) stands. This is a replication, not a re-score.
- **H_order-mnv3-rep:** default − no CNOT ≤ −1.0 and last ring − default ≤ −1.0, each negative in ≥ 4 of 5.
- **H_pairs-mnv3-rep:** pairs − default ≥ +1.0 (≥ 4/5 positive), and |pairs − no CNOT| ≤ 1.5.
- **H_gap-mnv3-rep:** linear − default ≥ +1.0 (≥ 4/5 positive), and no CNOT within 1.0 of linear.
- **Refutation:** both order differences > −0.5.
- **My expectation:** the direction repeats. default − no CNOT is again near −1 (a coin flip on the threshold). last ring − default is −3 to −8, with at least one unstable last-ring seed.

### Outcomes: night of 2026-10-31 (scored 2026-10-31 06:55 with `scripts/score_night31.py`)
The queue ended at 06:46: all 75 runs, 3 linear probes, 2 MLP probes and 3 exports exited 0. Verdicts use seeds 5–9 alone; 10-seed pooled numbers are descriptive. Linear probe unless stated; S in % of coordinates with |tanh z| > 0.9.

**(A) Gated escape-only, seeds 5–9.** Plain baselines: no CNOT 73.52, default 68.30, last ring 60.04 (S 45.5%).
- Last-ring gains: full +5.56 ± 5.58 (5/5), sat-only +5.85 ± 4.23 (5/5, S 0.43%), mid-only +7.08 ± 4.44 (5/5, S 0.01%). sat > mid in 1/5 seeds.
- Default gains: full +3.30, sat-only +2.57, mid-only +3.22. No-CNOT gains: +1.31, +0.75, +0.45.
- **H_sat-route-rep: PASS** (+5.85 ≥ 4.5, 5/5).
- **H_mid-small-rep: FAIL** (+7.08 > 2.3; sat > mid only 1/5).
- **Refutation: MET** (gain_mid ≥ gain_sat). By the committed criterion, the replication refutes "boosting saturated coordinates is the route".
- **Secondary:** default gain_sat ≥ 0.75 PASS; no CNOT both ≤ 1.5 PASS.
- **Pooled, 10 seeds, last ring:** sat-only +5.62 ± 4.52 and mid-only +5.89 ± 3.84, each positive in 9/10.
- **Against my expectation:** I expected mid-only to give 0–2 points. Over 10 seeds it is at least as large as sat-only. The gain does not need the boost on saturated coordinates.

**(B) Re-uploading, seeds 5–9.** no CNOT 65.72, pairs 61.22, default 55.89 ± 9.37, last ring 48.01 ± 8.93.
- **H_reup-order-rep: FAIL.** default − no CNOT −9.82 (5/5 negative) passes its part. last ring − default −7.89 ± 17.96 is negative in only 3/5, because default seed 5 probes 42.5 and two last-ring seeds beat their default. MLP: −10.29 (5/5), −7.26 (3/5).
- **H_reup-rank-rep: PASS** (Spearman −1.00, both probes).
- **Refutation: not met.**
- **Pooled, 10 seeds:** no CNOT 66.87 ± 3.17, pairs 63.72 ± 4.67, default 56.95 ± 6.89, last ring 48.15 ± 6.69. default − no CNOT −9.91 ± 5.03 (10/10 negative); last ring − default −8.80 ± 12.92 (8/10).
- **Against my expectation:** I expected a pass. The direction holds, but the per-seed sign criterion fails.

**(C) MobileNetV3-Small, seeds 5–9.** linear 85.65, no CNOT 85.16, pairs 86.21, default 84.74, last ring 80.18 ± 4.32.
- **H_order-mnv3-rep: FAIL.** default − no CNOT −0.43 (4/5 negative) misses −1.0. last ring − default −4.55 (5/5) passes its part. MLP: −0.54, −3.75.
- **H_pairs-mnv3-rep: PASS** (pairs − default +1.47, 5/5; pairs − no CNOT +1.04).
- **H_gap-mnv3-rep: FAIL** (linear − default +0.92 < 1.0, though 5/5 positive; |linear − no CNOT| 0.49 passes). Under the MLP probe it would pass (+1.61, 5/5; 1.07 > 1.0 fails the second part).
- **Refutation: not met.**
- **Pooled, 10 seeds:** linear 85.87 ± 0.55, no CNOT 85.28, pairs 86.12, default 84.62, last ring 79.59 ± 4.93. Differences to linear: no CNOT −0.59 ± 0.55, pairs +0.26 ± 0.50, default −1.24 ± 0.69 (10/10 negative), last ring −6.28 ± 4.75 (10/10). default − no CNOT −0.66 ± 0.82 (9/10 negative).
- **Against my expectation:** as predicted, default − no CNOT is near −1 and misses the threshold, and last ring − default is in −3 to −8. The cost on this backbone is small and directionally consistent.
