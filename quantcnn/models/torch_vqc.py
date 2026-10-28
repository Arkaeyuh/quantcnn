"""Batched pure-PyTorch statevector simulator of the VQC in ``quantum_head.py``.

Implements exactly the same circuit as ``build_qnode`` (RY angle encoding with
``pi * tanh``, RY/RZ variational layers, CNOT ring, <Z_i> readout, and the three
re-upload schedules), but evaluates a whole batch at once instead of looping
over samples through PennyLane. Wire 0 is the most significant bit, matching
PennyLane's convention, so trained weights are interchangeable between the two
backends. ``tests/test_torch_vqc.py`` checks outputs and gradients against
PennyLane.

Two ablations exist only in this backend (defaults reproduce the circuit above):
``encoding`` selects the angle map (``"pi_tanh"``: pi*tanh z in (-pi, pi), which
<Z> reads as an even function of z; ``"half"``: (pi/2)*tanh z + pi/2 in (0, pi),
monotone under <Z>; ``"half_sym"``: (pi/2)*tanh z, the same range width as
``"half"`` but even again; ``"phase"``: H then RZ(pi*tanh z), Bloch vector (cos, sin, 0), so every
zero-rotation readout vanishes and the product identity does not hold), and ``entangle`` selects the CNOT layer (``True``: a ring
after every variational layer; ``"last"``: a ring after the last one only; ``"pairs"``
/ ``"pairs_last"``: disjoint CNOT(0,1), CNOT(2,3), ... after every / the last layer;
``False``: none). Further layouts, each after every layer (``"<pattern>"``) or only after the
last one (``"<pattern>_last"``), are listed in ``cnot_gates``. ``"cz"`` is a ring of CZ(q, q+1 mod n)
after every layer: it entangles, but CZ commutes with Z, so its zero-rotation product order is 1.

Mechanism and generality tests (2026-10-26): encoding ``"pi_tanh_st"`` is pi*tanh z forward with a
straight-through backward (d phi / d z = pi, removing the sech^2 factor); ``"lin"`` is phi = z,
unbounded, with no saturating map. ``readout`` selects the measured observables: ``"z"`` (<Z_k>),
``"zz"`` (<Z_k Z_{k+1 mod n}>) or ``"zzz"`` (<Z_k Z_{k+1} Z_{k+2}>), which raise the product order
of a CNOT-free circuit to 2 or 3 without entangling gates.

Escape-only straight-through (2026-10-27): ``"pi_tanh_esc"`` is pi*tanh z forward; backward it
keeps sech^2 when the gradient step would push |z| outward and removes it (d tanh/dz = 1) when the
step pulls |z| inward. Saturated coordinates can leave, but the flat forward cannot drive them
outward without bound, which is what made ``"pi_tanh_st"`` diverge. ``"cz_last"`` is the CZ ring
after the last layer only; being diagonal right before a Z-basis measurement, it leaves every <Z>
readout unchanged, so it equals the no-CNOT circuit (a check, not an experiment).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


ENCODINGS = ("pi_tanh", "half", "half_sym", "phase", "pi_tanh_st", "lin", "pi_tanh_esc")
READOUTS = {"z": 1, "zz": 2, "zzz": 3}  # readout -> weight of the measured Z string
PATTERNS = ("ring", "pairs", "chain", "brick", "star", "star_in", "ladder2")
class EscapeTanh(torch.autograd.Function):
    """tanh forward; backward d tanh/dz = 1 where the descent step -g moves z toward 0
    (g * z > 0), and the usual sech^2 elsewhere."""

    @staticmethod
    def forward(ctx, x):
        t = torch.tanh(x)
        ctx.save_for_backward(x, t)
        return t

    @staticmethod
    def backward(ctx, g):
        x, t = ctx.saved_tensors
        return torch.where(g * x > 0, g, g * (1 - t * t))


ENTANGLE = (True, False, "last", "cz", "cz_last") + tuple(p for name in PATTERNS[1:] for p in (name, f"{name}_last"))


def cnot_gates(n_qubits: int, pattern: str = "ring", layer: int = 0) -> list[tuple[int, int]]:
    """(control, target) pairs of the entangling layer after variational layer ``layer``, in
    application order. ``ring``: CNOT(q, q+1 mod n); ``pairs``: disjoint (0,1), (2,3), ...;
    ``chain``: the ring without its closing gate; ``brick``: pairs (q, q+1 mod n) starting at
    q = 0 after even layers and q = 1 after odd ones; ``star``: CNOT(0, q) for every q;
    ``star_in``: CNOT(q, 0); ``ladder2``: CNOT(q, q+2 mod n)."""
    n = n_qubits
    if pattern == "ring":
        return [(q, (q + 1) % n) for q in range(n)]
    if pattern == "pairs":
        return [(q, q + 1) for q in range(0, n - 1, 2)]
    if pattern == "chain":
        return [(q, q + 1) for q in range(n - 1)]
    if pattern == "brick":
        return [(q, (q + 1) % n) for q in range(layer % 2, n, 2)]
    if pattern == "star":
        return [(0, q) for q in range(1, n)]
    if pattern == "star_in":
        return [(q, 0) for q in range(1, n)]
    if pattern == "ladder2":
        return [(q, (q + 2) % n) for q in range(n)]
    raise ValueError(pattern)


def entangle_layout(entangle: bool | str) -> tuple[str, bool] | None:
    """(pattern, last_only) for an ``entangle`` setting, or None for no CNOTs."""
    if entangle is False or entangle in ("cz", "cz_last"):
        return None
    if entangle is True:
        return "ring", False
    if entangle == "last":
        return "ring", True
    last = entangle.endswith("_last")
    return (entangle[: -len("_last")] if last else entangle), last


def _ring_cnot_permutation(n_qubits: int, pattern: str = "ring", layer: int = 0) -> torch.Tensor:
    """Index map for one entangling layer (default: CNOT(0,1), CNOT(1,2), ..., CNOT(n-1,0))."""
    dim = 2**n_qubits
    perm = torch.arange(dim)
    for c, t in cnot_gates(n_qubits, pattern, layer):
        cbit = 1 << (n_qubits - 1 - c)
        tbit = 1 << (n_qubits - 1 - t)
        # new_state[i] = old_state[i ^ tbit] when control bit of i is set
        idx = torch.arange(dim)
        src = torch.where((idx & cbit) != 0, idx ^ tbit, idx)
        perm = perm[src]
    return perm


class TorchStatevectorVQC(nn.Module):
    """Drop-in replacement for the PennyLane TorchLayer: same ``weights`` parameter."""

    def __init__(self, n_qubits: int, n_var_layers: int, mode: str, light_second_scale: float = 0.5,
                 encoding: str = "pi_tanh", entangle: bool | str = True, readout: str = "z"):
        super().__init__()
        if mode not in ("none", "light", "strong"):
            raise ValueError(mode)
        if encoding not in ENCODINGS:
            raise ValueError(encoding)
        if entangle not in ENTANGLE:
            raise ValueError(entangle)
        if readout not in READOUTS:
            raise ValueError(readout)
        self.encoding = encoding
        self.entangle = entangle
        self.n_qubits = n_qubits
        self.n_var_layers = n_var_layers
        self.mode = mode
        self.light_second_scale = light_second_scale
        self.weights = nn.Parameter(torch.empty(n_var_layers, n_qubits, 2))
        nn.init.uniform_(self.weights, -0.1, 0.1)
        # Plain CPU attributes, not buffers: simulation always runs on CPU in float64,
        # and MPS cannot hold float64 tensors when the model is moved there.
        layout = entangle_layout(entangle)
        self._perms = [None] * n_var_layers
        if layout is not None:
            pattern, last_only = layout
            for l in range(n_var_layers):
                if not last_only or l == n_var_layers - 1:
                    self._perms[l] = _ring_cnot_permutation(n_qubits, pattern, l)
        bits = (torch.arange(2**n_qubits)[:, None] >> torch.arange(n_qubits - 1, -1, -1)) & 1
        # CZ ring as a diagonal sign: (-1)^(b_q b_{q+1}) summed over the ring's pairs.
        self._czsign = None
        self._cz_last = entangle == "cz_last"
        if entangle in ("cz", "cz_last"):
            par = sum(bits[:, q] * bits[:, (q + 1) % n_qubits] for q in range(n_qubits)) % 2
            self._czsign = (1 - 2 * par).to(torch.complex128)
        zs = (1 - 2 * bits).to(torch.float64)
        w = READOUTS[readout]
        self.readout = readout
        self._zsign = torch.stack([zs[:, [(k + i) % n_qubits for i in range(w)]].prod(1) for k in range(n_qubits)], 1)

    def _apply_1q(self, state: torch.Tensor, q: int, mat: torch.Tensor) -> torch.Tensor:
        # state: (B, 2**n) complex; mat: (B, 2, 2) or (2, 2)
        b = state.shape[0]
        s = state.reshape(b, 2**q, 2, 2 ** (self.n_qubits - 1 - q))
        if mat.dim() == 2:
            s = torch.einsum("ij,bajc->baic", mat, s)
        else:
            s = torch.einsum("bij,bajc->baic", mat, s)
        return s.reshape(b, -1)

    @staticmethod
    def _ry(theta: torch.Tensor, cdtype) -> torch.Tensor:
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        m = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        return m.to(cdtype)

    @staticmethod
    def _rz(phi: torch.Tensor, cdtype) -> torch.Tensor:
        e = torch.exp(-0.5j * phi.to(cdtype))
        z = torch.zeros((), dtype=cdtype, device=phi.device)
        return torch.stack([torch.stack([e, z]), torch.stack([z, e.conj()])])

    def _encode(self, state, x, cdtype):
        if self.encoding == "phase":
            h = torch.tensor([[1, 1], [1, -1]], dtype=cdtype) / math.sqrt(2)
            phi = math.pi * torch.tanh(x)
            for q in range(self.n_qubits):
                e = torch.exp(-0.5j * phi[:, q].to(cdtype))
                rz = torch.diag_embed(torch.stack([e, e.conj()], -1))  # batched RZ(phi)
                state = self._apply_1q(self._apply_1q(state, q, h), q, rz)
            return state
        if self.encoding == "pi_tanh":
            phi = math.pi * torch.tanh(x)
        elif self.encoding == "pi_tanh_st":
            phi = math.pi * (torch.tanh(x).detach() + x - x.detach())
        elif self.encoding == "pi_tanh_esc":
            phi = math.pi * EscapeTanh.apply(x)
        elif self.encoding == "lin":
            phi = x
        elif self.encoding == "half":
            phi = 0.5 * math.pi * torch.tanh(x) + 0.5 * math.pi
        else:
            phi = 0.5 * math.pi * torch.tanh(x)
        for q in range(self.n_qubits):
            state = self._apply_1q(state, q, self._ry(phi[:, q], cdtype))
        return state

    def _var_layer(self, state, w, cdtype, l):
        for q in range(self.n_qubits):
            state = self._apply_1q(state, q, self._ry(w[q, 0], cdtype))
            state = self._apply_1q(state, q, self._rz(w[q, 1], cdtype))
        if self._czsign is not None:
            return state * self._czsign if not self._cz_last or l == self.n_var_layers - 1 else state
        perm = self._perms[l]
        return state if perm is None else state[:, perm]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_device, out_dtype = x.device, x.dtype
        # Two-step casts: MPS tensors cannot be converted to float64 in place.
        x = x.cpu().double()
        w = self.weights.cpu().double()
        cdtype = torch.complex128
        state = torch.zeros(x.shape[0], 2**self.n_qubits, dtype=cdtype)
        state[:, 0] = 1.0
        L = self.n_var_layers
        if self.mode == "none":
            state = self._encode(state, x, cdtype)
            for l in range(L):
                state = self._var_layer(state, w[l], cdtype, l)
        elif self.mode == "strong":
            for l in range(L):
                state = self._encode(state, x, cdtype)
                state = self._var_layer(state, w[l], cdtype, l)
        else:  # light
            state = self._encode(state, x, cdtype)
            for l in range(L):
                state = self._var_layer(state, w[l], cdtype, l)
                if l < L - 1 and l % 2 == 1:
                    state = self._encode(state, x * self.light_second_scale, cdtype)
        probs = state.abs() ** 2
        return (probs @ self._zsign).to(out_dtype).to(out_device)
