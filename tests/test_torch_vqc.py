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


if __name__ == "__main__":
    test_forward_and_grad_match_pennylane()
    test_phase_encoding_matches_pennylane()
    print("ok")
