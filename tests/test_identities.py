"""Check the analytical claims the paper rests on against the simulator and heads.

* zero-rotation identity: with every rotation zero, <Z_k> = prod_{j in S_k} cos(pi tanh z_j)
  for every CNOT structure used (Eq. 1);
* the classical product heads use exactly those supports;
* ring-free circuit: <Z_j> lies in span{cos a_j, sin a_j} for any rotations and depends on
  z_j only (Eq. 2);
* pairs after every layer cancel only at zero rotations;
* designed supports have the order / overlap / rank stated in the paper;
* gradient-decoupled product heads keep the forward value and pass the stated gradient;
* a VQC head with rotations frozen at zero equals the classical product head in output and
  input gradient (same readout weights);
* the pairs-every-layer circuit, whose CNOTs cancel at zero rotations, can still entangle: one
  final-layer R_y(pi/2) on qubit 0 makes a Bell pair (reduced purity 1/2).
"""

import itertools
import math
import statistics as st

import torch

from quantcnn.models.classical_heads import DESIGNED_SUPPORTS, CosProductHead, cnot_ring_supports
from quantcnn.models.quantum_head import QuantumClassifierHead
from quantcnn.models.torch_vqc import TorchStatevectorVQC

N = 8
# (n_var_layers, entangle) -> supports at zero rotations
CIRCUITS = {
    (4, False): [[k] for k in range(N)],
    (4, True): cnot_ring_supports(N, 4),
    (4, "last"): cnot_ring_supports(N, 4, last_only=True),
    (1, True): cnot_ring_supports(N, 1),
    (2, True): cnot_ring_supports(N, 2),
    (4, "pairs"): cnot_ring_supports(N, 4, pattern="pairs"),
    (4, "pairs_last"): cnot_ring_supports(N, 4, last_only=True, pattern="pairs"),
}


def _vqc(layers, entangle, zero=True, seed=0):
    torch.manual_seed(seed)
    vqc = TorchStatevectorVQC(N, layers, "none", entangle=entangle).double()
    with torch.no_grad():
        if zero:
            vqc.weights.zero_()
        else:
            vqc.weights.uniform_(-2.0, 2.0)
    return vqc


def test_zero_rotation_identity():
    z = torch.randn(16, N, dtype=torch.float64) * 1.5
    c = torch.cos(math.pi * torch.tanh(z))
    for (layers, ent), supports in CIRCUITS.items():
        q = _vqc(layers, ent)(z)
        expected = torch.stack([c[:, s].prod(1) for s in supports], 1)
        assert torch.allclose(q, expected, atol=1e-10), (layers, ent, (q - expected).abs().max())


def test_product_heads_use_circuit_supports():
    for (layers, last, pattern) in [(4, False, "ring"), (4, True, "ring"), (4, True, "pairs")]:
        head = CosProductHead(N, 10, layers, last_only=last, pattern=pattern)
        mask = torch.zeros(N, N)
        for k, s in enumerate(cnot_ring_supports(N, layers, last, pattern)):
            mask[k, s] = 1
        assert torch.equal(head.mask, mask), (layers, last, pattern)


def test_ring_free_is_per_coordinate_sinusoid():
    vqc = _vqc(4, False, zero=False)
    base = torch.randn(1, N, dtype=torch.float64)
    a = torch.linspace(-0.95, 0.95, 9, dtype=torch.float64)  # tanh z values
    for j in range(N):
        z = base.repeat(len(a), 1)
        z[:, j] = torch.atanh(a)
        q = vqc(z)[:, j]
        enc = math.pi * a
        X = torch.stack([torch.cos(enc), torch.sin(enc)], 1)
        coef = torch.linalg.lstsq(X, q[:, None]).solution
        assert torch.allclose(X @ coef, q[:, None], atol=1e-10), j  # exactly alpha cos + beta sin
        other = base.clone(); other[0, (j + 1) % N] += 1.0  # changing another input leaves <Z_j> fixed
        assert torch.allclose(vqc(other)[0, j], vqc(base)[0, j], atol=1e-12), j


def test_pairs_cancel_only_at_zero_rotations():
    z = torch.randn(16, N, dtype=torch.float64)
    assert torch.allclose(_vqc(4, "pairs")(z), _vqc(4, False)(z), atol=1e-10)
    assert (_vqc(4, "pairs", zero=False)(z) - _vqc(4, False, zero=False)(z)).abs().max() > 1e-3


def test_designed_support_statistics():
    def stats(sup):
        m = torch.zeros(N, N)
        for k, s in enumerate(sup):
            m[k, s] = 1
        overlap = st.mean(len(set(x) & set(y)) for x, y in itertools.permutations(sup, 2))
        return st.mean(len(s) for s in sup), round(overlap, 4), int(torch.linalg.matrix_rank(m))
    expected = {"win2": (2, 0.2857, 7), "star2": (2, 0.8214, 8), "win3": (3, 0.8571, 8),
                "star3": (3, 1.7143, 8), "win4": (4, 1.7143, 5), "gap4": (4, 1.7143, 8)}
    for name, exp in expected.items():
        assert stats(DESIGNED_SUPPORTS[name]) == exp, (name, stats(DESIGNED_SUPPORTS[name]))
        assert len({tuple(sorted(s)) for s in DESIGNED_SUPPORTS[name]}) == N, name  # 8 distinct supports


def test_decoupled_head_forward_and_gradient():
    torch.manual_seed(0)
    a = CosProductHead(N, 10, 4, last_only=True)
    b = CosProductHead(N, 10, 4, last_only=True, decouple_grad=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, N, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    oa, ob = a(x), b(y)
    assert torch.allclose(oa, ob)
    ob.sum().backward()
    dc = -math.pi * (1 - torch.tanh(y.detach()) ** 2) * torch.sin(math.pi * torch.tanh(y.detach()))
    expected = (b.fc.weight.sum(0)[None, :, None] * b.mask[None]).sum(1) * dc
    assert torch.allclose(y.grad, expected, atol=1e-5)


def test_zero_rotation_vqc_head_equals_product_head():
    for entangle, last in [(True, False), ("last", True)]:
        torch.manual_seed(0)
        q = QuantumClassifierHead(N, 10, 4, "none", sim="torch", entangle=entangle)
        c = CosProductHead(N, 10, 4, last_only=last)
        with torch.no_grad():
            q.vqc.qlayer.weights.zero_()
            c.fc.load_state_dict(q.proj.state_dict())
        x = torch.randn(32, N) * 1.5
        x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        o1, o2 = q(x1), c(x2)
        y = torch.randint(0, 10, (32,))
        torch.nn.functional.cross_entropy(o1, y).backward()
        torch.nn.functional.cross_entropy(o2, y).backward()
        assert torch.allclose(o1, o2, atol=1e-5), entangle
        assert torch.allclose(x1.grad, x2.grad, atol=1e-6), entangle


def test_pairs_circuit_can_entangle():
    vqc = _vqc(4, "pairs")
    with torch.no_grad():
        vqc.weights[-1, 0, 0] = math.pi / 2  # R_y(pi/2) on qubit 0 in the last layer
    cd = torch.complex128
    state = torch.zeros(1, 2**N, dtype=cd)
    state[:, 0] = 1.0
    state = vqc._encode(state, torch.zeros(1, N, dtype=torch.float64), cd)  # z = 0: product state |0...0>
    w = vqc.weights.detach().double()
    for layer in range(4):
        state = vqc._var_layer(state, w[layer], cd, layer)
    psi = state.reshape(2, 2 ** (N - 1))  # qubit 0 is the most significant bit
    rho = psi @ psi.conj().T
    purity = torch.trace(rho @ rho).real.item()
    assert abs(purity - 0.5) < 1e-12, purity


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
