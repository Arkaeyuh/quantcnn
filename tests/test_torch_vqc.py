"""Check the batched torch simulator against the PennyLane QNode it replaces."""

import torch

from quantcnn.models.quantum_head import QuantumVQCLayer
from quantcnn.models.torch_vqc import TorchStatevectorVQC


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


if __name__ == "__main__":
    test_forward_and_grad_match_pennylane()
    print("ok")
