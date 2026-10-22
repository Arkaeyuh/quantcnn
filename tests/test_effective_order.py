"""Checks for scripts/effective_order.py.

* at zero rotations the effective order equals the product order of Eq. (1), readout by readout;
* the Pauli coefficients reproduce the simulator's readouts exactly at random rotations and inputs;
* the inputs a readout depends on (by the Pauli expansion) are exactly those with a nonzero input
  gradient in the simulator;
* without CNOTs every readout depends on its own input only, at any rotations;
* with the phase encoding (H then RZ) the expansion reproduces the simulator's readouts and input
  dependences, with Z strings invisible;
* weight on strings made only of I and Y factors counts as invisible (Y share), and only the
  identity string is dropped as a constant.
"""

import math
import statistics as st

import torch

from quantcnn.models.classical_heads import cnot_ring_supports
from quantcnn.models.torch_vqc import TorchStatevectorVQC, entangle_layout
from scripts.effective_order import (LAYOUTS, N, circuit_stats, expectation_from_paulis, pauli_coefficients,
                                     readout_operators, readout_stats)


def _vqc(layers, entangle, scale=0.0, seed=0, encoding="pi_tanh"):
    torch.manual_seed(seed)
    vqc = TorchStatevectorVQC(N, layers, "none", encoding=encoding, entangle=entangle).double()
    with torch.no_grad():
        vqc.weights.uniform_(-scale, scale) if scale else vqc.weights.zero_()
    return vqc


def _supports(layers, entangle):
    layout = entangle_layout(entangle)
    if layout is None:
        return [[k] for k in range(N)]
    pattern, last = layout
    return cnot_ring_supports(N, layers, last_only=last, pattern=pattern)


def test_zero_rotations_give_product_order():
    for name, (layers, ent) in LAYOUTS.items():
        vqc = _vqc(layers, ent)
        sup = _supports(layers, ent)
        for k, op in enumerate(readout_operators(vqc, vqc.weights)):
            s = readout_stats(pauli_coefficients(op, N), N)
            assert abs(s["order_eff"] - len(sup[k])) < 1e-9 and s["n_inputs"] == len(sup[k]), (name, k, s)
            assert s["y_share"] < 1e-12, (name, k)
        assert abs(circuit_stats(vqc, vqc.weights)["order_eff"] - st.mean(len(x) for x in sup)) < 1e-9, name


def test_pauli_expansion_reproduces_readouts():
    for name in ("quantum", "quantum+brick", "quantum+star_in_last", "quantum+pairs"):
        layers, ent = LAYOUTS[name]
        vqc = _vqc(layers, ent, scale=2.0, seed=1)
        z = torch.randn(3, N, dtype=torch.float64)
        q = vqc(z)
        coefs = [pauli_coefficients(op, N) for op in readout_operators(vqc, vqc.weights)]
        for i in range(len(z)):
            angles = math.pi * torch.tanh(z[i])
            for k in range(N):
                assert abs(expectation_from_paulis(coefs[k], angles) - q[i, k].item()) < 1e-10, (name, i, k)


def test_dependence_set_matches_input_gradients():
    idx = torch.cartesian_prod(*[torch.arange(4)] * N).reshape([4] * N + [N])
    for name in ("quantum+entlast", "quantum+brick", "quantum+chain"):
        layers, ent = LAYOUTS[name]
        vqc = _vqc(layers, ent, scale=1.0, seed=2)
        z = torch.randn(4, N, dtype=torch.float64, requires_grad=True)
        q = vqc(z)
        for k, op in enumerate(readout_operators(vqc, vqc.weights)):
            c = pauli_coefficients(op, N)
            visible = (c.abs() > 1e-12) & ~(idx == 2).any(-1)
            dep = (visible[..., None] & ((idx == 1) | (idx == 3))).reshape(-1, N).any(0)
            (g,) = torch.autograd.grad(q[:, k].sum(), z, retain_graph=True)
            moved = g.abs().amax(0) > 1e-9
            assert torch.equal(dep, moved), (name, k, dep.tolist(), moved.tolist())
            assert readout_stats(c, N)["n_inputs"] == int(moved.sum()), (name, k)


def test_ring_free_stays_order_one():
    vqc = _vqc(4, False, scale=math.pi, seed=3)
    for op in readout_operators(vqc, vqc.weights):
        s = readout_stats(pauli_coefficients(op, N), N)
        assert abs(s["order_eff"] - 1) < 1e-9 and s["n_inputs"] == 1, s


def test_phase_encoding_expansion():
    idx = torch.cartesian_prod(*[torch.arange(4)] * N).reshape([4] * N + [N])
    for name in ("quantum", "quantum+brick", "quantum+entlast", "quantum+noent"):
        layers, ent = LAYOUTS[name]
        vqc = _vqc(layers, ent, scale=1.0, seed=4, encoding="phase")
        z = torch.randn(3, N, dtype=torch.float64, requires_grad=True)
        q = vqc(z)
        for k, op in enumerate(readout_operators(vqc, vqc.weights)):
            c = pauli_coefficients(op, N)
            for i in range(len(z)):
                angles = math.pi * torch.tanh(z[i].detach())
                assert abs(expectation_from_paulis(c, angles, "phase") - q[i, k].item()) < 1e-10, (name, i, k)
            (g,) = torch.autograd.grad(q[:, k].sum(), z, retain_graph=True)
            assert readout_stats(c, N, encoding="phase")["n_inputs"] == int((g.abs().amax(0) > 1e-9).sum()), (name, k)
    # near-zero rotations: last-layer layouts factorize over their supports at any rotations, so
    # their effective order is the R_y product order even when the visible weight is tiny
    for name in ("quantum+entlast", "quantum+chain_last"):
        layers, ent = LAYOUTS[name]
        vqc = _vqc(layers, ent, scale=0.1, seed=5, encoding="phase")
        expect = st.mean(len(x) for x in _supports(layers, ent))
        assert abs(circuit_stats(vqc, vqc.weights, "phase")["order_eff"] - expect) < 1e-9, name
    vqc = _vqc(4, True, encoding="phase")  # zero rotations: every readout is a Z string, invisible
    for op in readout_operators(vqc, vqc.weights):
        s = readout_stats(pauli_coefficients(op, N), N, encoding="phase")
        assert s["y_share"] > 1 - 1e-12 and s["n_inputs"] == 0, s


def test_y_share_counts_pure_y_strings():
    coef = torch.zeros(4, 4, dtype=torch.float64)  # qubits (0, 1); index order I, X, Y, Z
    coef[0, 0] = 0.5  # identity: constant, ignored
    coef[3, 0] = 0.6  # Z on qubit 0: visible, order 1
    coef[2, 0] = 0.8  # Y on qubit 0 only: invisible
    s = readout_stats(coef, 2)
    assert abs(s["y_share"] - 0.64) < 1e-12 and abs(s["order_eff"] - 1) < 1e-12 and s["n_inputs"] == 1, s
    coef[2, 2] = 0.6  # Y on both qubits: invisible
    s = readout_stats(coef, 2)
    assert abs(s["y_share"] - 1.0 / 1.36) < 1e-12 and s["n_inputs"] == 1, s


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
