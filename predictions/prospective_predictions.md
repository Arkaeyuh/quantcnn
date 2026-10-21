# Prospective predictions (verbatim from the project log)

The paper (Sec. 5.3, App. F, App. M) states that the predicted ordering for the MNIST
replication and the predictions and pass criteria for the second circuit family were written
before those runs. They were written in the project's working log (`findings.md`), which is not
under version control. This file copies those entries **verbatim and unedited** so they have a
verifiable record; its commit date is the first verifiable timestamp, not the date of writing.
Outcomes are deliberately left out (they are in the paper).

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
