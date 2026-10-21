#!/usr/bin/env python3
"""Effective product order of a VQC readout at any rotation angles.

With rotations theta, readout k is <psi(z)| O_k |psi(z)>, where O_k = U(theta)^dag Z_k U(theta)
and U is the variational block (rotations and CNOT layers, no encoding). Expanding
O_k = sum_P c_P P over Pauli strings, the encoded product state R_y(a_j)|0> has Bloch vector
(sin a_j, 0, cos a_j), so <P> = prod_j b_j(P_j) with b(I) = 1, b(X) = sin a_j, b(Y) = 0,
b(Z) = cos a_j. Strings containing Y therefore never reach the readout, and a string with
X or Z on qubit j makes the readout depend on input j.

The effective order of readout k is the c_P^2-weighted mean number of inputs a string touches,
over the strings that reach the readout (no Y factor, not the identity):

    order_eff(k) = sum_P c_P^2 |supp(P)| / sum_P c_P^2.

At theta = 0, O_k is the Z-string on S_k, so order_eff(k) = |S_k| and the circuit's mean is the
product order of Eq. (1). ``y_share`` is the fraction of non-identity weight on strings with a
Y factor (invisible to the readout).

From repository root::

    # trained circuits (validation-selected lr, seeds 0-4), CIFAR-10/4,000
    PYTHONPATH=. python -m scripts.effective_order --source trained --out results/effective_order_trained.csv
    # random rotations, theta ~ U(-s, s)
    PYTHONPATH=. python -m scripts.effective_order --source random --scale 3.14159 --samples 20 \
        --out results/effective_order_random_pi.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from quantcnn.models.torch_vqc import TorchStatevectorVQC

N = 8
# Pauli coefficients of a 2x2 block M from vec(M) indexed 2*row + col: rows I, X, Y, Z.
_PAULI_FROM_VEC = torch.tensor([[0.5, 0, 0, 0.5],
                                [0, 0.5, 0.5, 0],
                                [0, 0.5j, -0.5j, 0],
                                [0.5, 0, 0, -0.5]], dtype=torch.complex128)

# name -> (n_var_layers, entangle) for every circuit studied (ring family and second family)
LAYOUTS = {
    "quantum+noent": (4, False), "quantum+pairs": (4, "pairs"), "quantum+pairs_last": (4, "pairs_last"),
    "quantum": (4, True), "quantum+L2": (2, True), "quantum+L1": (1, True), "quantum+entlast": (4, "last"),
    "quantum+star": (4, "star"), "quantum+chain": (4, "chain"), "quantum+star_in_last": (4, "star_in_last"),
    "quantum+ladder2_last": (4, "ladder2_last"), "quantum+brick": (4, "brick"), "quantum+chain_last": (4, "chain_last"),
}


def variational_unitary(vqc: TorchStatevectorVQC, weights: torch.Tensor) -> torch.Tensor:
    """2^n x 2^n unitary of the variational block (rotations + CNOT layers, no encoding)."""
    assert vqc.mode == "none", "single encoding only"
    dim = 2**vqc.n_qubits
    cd = torch.complex128
    state = torch.eye(dim, dtype=cd)  # row i is basis state e_i
    w = weights.detach().double()
    for l in range(vqc.n_var_layers):
        state = vqc._var_layer(state, w[l], cd, l)
    return state.T  # column i = U e_i


def pauli_coefficients(op: torch.Tensor, n: int) -> torch.Tensor:
    """Real coefficients c_P of a Hermitian 2^n x 2^n operator, as a (4,)*n tensor (I, X, Y, Z)."""
    t = op.reshape([2] * (2 * n))  # rows r_0..r_{n-1}, cols c_0..c_{n-1}; qubit 0 is the MSB
    t = t.permute([a for q in range(n) for a in (q, n + q)]).reshape([4] * n)
    for q in range(n):
        t = torch.movedim(torch.tensordot(_PAULI_FROM_VEC, t, dims=([1], [q])), 0, q)
    assert t.imag.abs().max() < 1e-10
    return t.real


def readout_operators(vqc: TorchStatevectorVQC, weights: torch.Tensor) -> list[torch.Tensor]:
    """O_k = U^dag Z_k U for every readout k."""
    u = variational_unitary(vqc, weights)
    return [u.conj().T @ torch.diag(vqc._zsign[:, k].to(torch.complex128)) @ u for k in range(vqc.n_qubits)]


def expectation_from_paulis(coef: torch.Tensor, angles: torch.Tensor) -> float:
    """<psi|O|psi> for the encoded product state with angles a_j, from the Pauli coefficients."""
    t = coef
    for a in angles.tolist():
        b = torch.tensor([1.0, math.sin(a), 0.0, math.cos(a)], dtype=t.dtype)
        t = torch.tensordot(t, b, dims=([0], [0]))
    return float(t)


def readout_stats(coef: torch.Tensor, n: int, tol: float = 1e-12) -> dict:
    """Effective order, Y-share and the set of inputs the readout depends on."""
    idx = torch.cartesian_prod(*[torch.arange(4)] * n).reshape([4] * n + [n])
    has_y = (idx == 2).any(-1)
    weight_sz = ((idx == 1) | (idx == 3)).sum(-1)
    w = coef**2
    w = torch.where(weight_sz == 0, torch.zeros_like(w), w)  # drop identity (constant)
    visible = w * (~has_y)
    total, vis = w.sum().item(), visible.sum().item()
    order = (visible * weight_sz).sum().item() / vis if vis > tol else 0.0
    touched = ((visible > tol)[..., None] & ((idx == 1) | (idx == 3))).reshape(-1, n).any(0)
    return dict(order_eff=order, y_share=(total - vis) / total if total > tol else 0.0,
                n_inputs=int(touched.sum()))


def circuit_stats(vqc: TorchStatevectorVQC, weights: torch.Tensor) -> dict:
    """Mean effective order / Y-share / input count over the circuit's readouts."""
    stats = [readout_stats(pauli_coefficients(o, vqc.n_qubits), vqc.n_qubits) for o in readout_operators(vqc, weights)]
    return {key: sum(s[key] for s in stats) / len(stats) for key in stats[0]}


def _trained_rows() -> list[dict]:
    from scripts.zero_angle_check import CASES, CASES_FAMILY, EXTRA_SEED_GRID, ROOT, SLUG, _load
    from scripts.probe_backbones import _selected_runs

    extra = {"quantum+L1": ("_L1", "cifar_entangle", range(3)), "quantum+L2": ("_L2", "cifar_entangle", range(3))}
    rows = []
    for which, cases in (("cases", {**CASES, **extra}), ("family", CASES_FAMILY)):
        for name, (suffix, grid, seeds) in cases.items():
            if name not in LAYOUTS:
                continue  # e.g. the frozen-at-init control
            lr = _selected_runs(ROOT, [grid])[(4000, name)][0]
            for seed in seeds:
                run = ROOT / (grid if seed < 3 else EXTRA_SEED_GRID[which]) / SLUG.format(suffix, lr, seed)
                vqc = _load(run).head.vqc.qlayer
                rows.append(dict(head=name, source="trained", lr=lr, seed=seed, **circuit_stats(vqc, vqc.weights)))
    return rows


def _random_rows(scale: float, samples: int) -> list[dict]:
    rows = []
    for name, (layers, entangle) in LAYOUTS.items():
        vqc = TorchStatevectorVQC(N, layers, "none", entangle=entangle).double()
        g = torch.Generator().manual_seed(0)
        for s in range(samples):
            w = (torch.rand(vqc.weights.shape, generator=g, dtype=torch.float64) * 2 - 1) * scale
            rows.append(dict(head=name, source=f"random_U({scale:g})", lr="", seed=s, **circuit_stats(vqc, w)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["trained", "random"], required=True)
    ap.add_argument("--scale", type=float, default=math.pi, help="random: theta ~ U(-scale, scale)")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = _trained_rows() if a.source == "trained" else _random_rows(a.scale, a.samples)
    with Path(a.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
