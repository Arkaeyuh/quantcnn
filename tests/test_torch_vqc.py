"""Check the batched torch simulator against the PennyLane QNode it replaces."""

import math

import pennylane as qml
import torch

from quantcnn.models.quantum_head import QuantumVQCLayer
from quantcnn.models.torch_vqc import TorchStatevectorVQC, cnot_gates, entangle_layout


def _pair(mode, seed=0):
    torch.manual_seed(seed)
    pl = QuantumVQCLayer(8, 4, mode)
    with torch.no_grad():
        pl.qlayer.weights.uniform_(-3.0, 3.0)
    ts = TorchStatevectorVQC(8, 4, mode)
    with torch.no_grad():
        ts.weights.copy_(pl.qlayer.weights)
    return pl, ts


def test_forward_and_grad_match_pennylane():
    for mode in ("none", "light", "strong"):
        pl, ts = _pair(mode)
        x = torch.randn(6, 8, dtype=torch.float64) * 1.5
        x1 = x.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)
        y1 = pl(x1)
        y2 = ts(x2)
        assert torch.allclose(y1.double(), y2, atol=1e-6), (mode, (y1 - y2).abs().max())
        g = torch.randn_like(y2)
        (y1.double() * g).sum().backward()
        (y2 * g).sum().backward()
        assert torch.allclose(x1.grad, x2.grad, atol=1e-6), mode
        assert torch.allclose(pl.qlayer.weights.grad.double(), ts.weights.grad.double(), atol=1e-5), mode


def test_phase_encoding_matches_pennylane():
    """H then RZ(pi tanh z) encoding, written independently in PennyLane, for several layouts."""
    dev = qml.device("default.qubit", wires=8)
    for ent in (True, "last", "brick", "ladder2_last", False):
        torch.manual_seed(1)
        ts = TorchStatevectorVQC(8, 4, "none", encoding="phase", entangle=ent).double()
        with torch.no_grad():
            ts.weights.uniform_(-3.0, 3.0)
        layout = entangle_layout(ent)

        @qml.qnode(dev, interface="torch")
        def circuit(x, w):
            for q in range(8):
                qml.Hadamard(q)
                qml.RZ(math.pi * torch.tanh(x[q]), q)
            for l in range(4):
                for q in range(8):
                    qml.RY(w[l, q, 0], q)
                    qml.RZ(w[l, q, 1], q)
                if layout is not None and (not layout[1] or l == 3):
                    for c, t in cnot_gates(8, layout[0], l):
                        qml.CNOT([c, t])
            return [qml.expval(qml.PauliZ(q)) for q in range(8)]

        x = torch.randn(3, 8, dtype=torch.float64) * 1.5
        ref = torch.stack([torch.stack(circuit(xi, ts.weights.detach())) for xi in x])
        assert torch.allclose(ts(x), ref, atol=1e-10), (ent, (ts(x) - ref).abs().max())



def test_escape_straight_through_and_cz_last():
    """pi_tanh_esc: forward equals pi_tanh; backward is the plain gradient times sech^2 for outward
    steps and without it for inward ones. cz_last equals the no-CNOT circuit (diagonal before Z)."""
    torch.manual_seed(4)
    a = TorchStatevectorVQC(8, 4, "none").double()
    b = TorchStatevectorVQC(8, 4, "none", encoding="pi_tanh_esc").double()
    s = TorchStatevectorVQC(8, 4, "none", encoding="pi_tanh_st").double()
    for m in (b, s):
        with torch.no_grad():
            m.weights.copy_(a.weights)
    x = (torch.randn(6, 8, dtype=torch.float64) * 2).requires_grad_(True)
    xb, xs = (x.detach().clone().requires_grad_(True) for _ in range(2))
    assert torch.allclose(a(x), b(xb), atol=1e-12)
    a(x).sum().backward()
    b(xb).sum().backward()
    s(xs).sum().backward()
    inward = xs.grad * x.detach() > 0
    assert inward.any() and (~inward).any()
    assert torch.allclose(xb.grad, torch.where(inward, xs.grad, x.grad), atol=1e-10)
    c1 = TorchStatevectorVQC(8, 4, "none", entangle="cz").double()
    c2 = TorchStatevectorVQC(8, 4, "none", entangle="cz_last").double()
    with torch.no_grad():
        c2.weights.copy_(c1.weights)
    z = torch.randn(5, 8, dtype=torch.float64)
    c0 = TorchStatevectorVQC(8, 4, "none", entangle=False).double()
    with torch.no_grad():
        c0.weights.copy_(c1.weights)
    assert not torch.allclose(c1(z), c0(z), atol=1e-6)
    assert torch.allclose(c2(z), c0(z), atol=1e-10)


if __name__ == "__main__":
    test_forward_and_grad_match_pennylane()
    test_phase_encoding_matches_pennylane()
    print("ok")


def test_cz_ring_readouts_and_linear_encoding_match_pennylane():
    """CZ ring, multi-qubit Z readouts and the unbounded phi = z encoding (2026-10-26 tests)."""
    dev = qml.device("default.qubit", wires=8)
    for ent, readout, enc in (("cz", "z", "pi_tanh"), (False, "zz", "pi_tanh"), (False, "zzz", "lin"),
                              (True, "z", "lin"), ("cz", "zz", "lin")):
        torch.manual_seed(2)
        ts = TorchStatevectorVQC(8, 4, "none", encoding=enc, entangle=ent, readout=readout).double()
        with torch.no_grad():
            ts.weights.uniform_(-3.0, 3.0)
        w_len = {"z": 1, "zz": 2, "zzz": 3}[readout]

        @qml.qnode(dev, interface="torch")
        def circuit(x, w):
            for q in range(8):
                qml.RY(x[q] if enc == "lin" else math.pi * torch.tanh(x[q]), q)
            for l in range(4):
                for q in range(8):
                    qml.RY(w[l, q, 0], q)
                    qml.RZ(w[l, q, 1], q)
                if ent == "cz":
                    for q in range(8):
                        qml.CZ([q, (q + 1) % 8])
                elif ent is True:
                    for q in range(8):
                        qml.CNOT([q, (q + 1) % 8])
            obs = []
            for k in range(8):
                o = qml.PauliZ(k)
                for i in range(1, w_len):
                    o = o @ qml.PauliZ((k + i) % 8)
                obs.append(qml.expval(o))
            return obs

        x = torch.randn(3, 8, dtype=torch.float64) * 1.5
        ref = torch.stack([torch.stack(circuit(x[b], ts.weights.detach())) for b in range(3)])
        assert torch.allclose(ts(x), ref, atol=1e-8), (ent, readout, enc)


def test_zero_rotation_orders_of_new_heads():
    """At zero rotations the CZ ring reads cos(phi_k) (order 1), and the CNOT-free circuit with
    zz / zzz readout reads products of 2 / 3 consecutive cosines."""
    x = torch.randn(5, 8, dtype=torch.float64)
    c = torch.cos(math.pi * torch.tanh(x))
    for ent, readout, w in (("cz", "z", 1), (False, "zz", 2), (False, "zzz", 3)):
        ts = TorchStatevectorVQC(8, 4, "none", entangle=ent, readout=readout).double()
        with torch.no_grad():
            ts.weights.zero_()
        expect = torch.stack([c[:, [(k + i) % 8 for i in range(w)]].prod(1) for k in range(8)], 1)
        assert torch.allclose(ts(x), expect, atol=1e-10), (ent, readout)


def test_straight_through_encoding():
    """pi_tanh_st: same forward as pi_tanh; backward drops sech^2 (d phi/dz = pi)."""
    torch.manual_seed(3)
    a = TorchStatevectorVQC(8, 4, "none").double()
    b = TorchStatevectorVQC(8, 4, "none", encoding="pi_tanh_st").double()
    with torch.no_grad():
        b.weights.copy_(a.weights)
    x = (torch.randn(4, 8, dtype=torch.float64) * 2).requires_grad_(True)
    xs = x.detach().clone().requires_grad_(True)
    ya, yb = a(x), b(xs)
    assert torch.allclose(ya, yb, atol=1e-12)
    ya.sum().backward()
    yb.sum().backward()
    sech2 = 1 - torch.tanh(x.detach()) ** 2
    assert torch.allclose(x.grad, xs.grad * sech2, atol=1e-10)
