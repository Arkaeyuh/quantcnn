"""Batched pure-PyTorch statevector simulator of the VQC in ``quantum_head.py``.

Implements exactly the same circuit as ``build_qnode`` (RY angle encoding with
``pi * tanh``, RY/RZ variational layers, CNOT ring, <Z_i> readout, and the three
re-upload schedules), but evaluates a whole batch at once instead of looping
over samples through PennyLane. Wire 0 is the most significant bit, matching
PennyLane's convention, so trained weights are interchangeable between the two
backends. ``tests/test_torch_vqc.py`` checks outputs and gradients against
PennyLane.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _ring_cnot_permutation(n_qubits: int) -> torch.Tensor:
    """Index map for CNOT(0,1), CNOT(1,2), ..., CNOT(n-1,0) applied in that order."""
    dim = 2**n_qubits
    perm = torch.arange(dim)
    for q in range(n_qubits):
        c, t = q, (q + 1) % n_qubits
        cbit = 1 << (n_qubits - 1 - c)
        tbit = 1 << (n_qubits - 1 - t)
        # new_state[i] = old_state[i ^ tbit] when control bit of i is set
        idx = torch.arange(dim)
        src = torch.where((idx & cbit) != 0, idx ^ tbit, idx)
        perm = perm[src]
    return perm


class TorchStatevectorVQC(nn.Module):
    """Drop-in replacement for the PennyLane TorchLayer: same ``weights`` parameter."""

    def __init__(self, n_qubits: int, n_var_layers: int, mode: str, light_second_scale: float = 0.5):
        super().__init__()
        if mode not in ("none", "light", "strong"):
            raise ValueError(mode)
        self.n_qubits = n_qubits
        self.n_var_layers = n_var_layers
        self.mode = mode
        self.light_second_scale = light_second_scale
        self.weights = nn.Parameter(torch.empty(n_var_layers, n_qubits, 2))
        nn.init.uniform_(self.weights, -0.1, 0.1)
        # Plain CPU attributes, not buffers: simulation always runs on CPU in float64,
        # and MPS cannot hold float64 tensors when the model is moved there.
        self._perm = _ring_cnot_permutation(n_qubits)
        bits = (torch.arange(2**n_qubits)[:, None] >> torch.arange(n_qubits - 1, -1, -1)) & 1
        self._zsign = (1 - 2 * bits).to(torch.float64)

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
        phi = math.pi * torch.tanh(x)
        for q in range(self.n_qubits):
            state = self._apply_1q(state, q, self._ry(phi[:, q], cdtype))
        return state

    def _var_layer(self, state, w, cdtype):
        for q in range(self.n_qubits):
            state = self._apply_1q(state, q, self._ry(w[q, 0], cdtype))
            state = self._apply_1q(state, q, self._rz(w[q, 1], cdtype))
        return state[:, self._perm]

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
                state = self._var_layer(state, w[l], cdtype)
        elif self.mode == "strong":
            for l in range(L):
                state = self._encode(state, x, cdtype)
                state = self._var_layer(state, w[l], cdtype)
        else:  # light
            state = self._encode(state, x, cdtype)
            for l in range(L):
                state = self._var_layer(state, w[l], cdtype)
                if l < L - 1 and l % 2 == 1:
                    state = self._encode(state, x * self.light_second_scale, cdtype)
        probs = state.abs() ** 2
        return (probs @ self._zsign).to(out_dtype).to(out_device)
