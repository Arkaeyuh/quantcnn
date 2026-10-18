"""Classical classifier heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TrigFeatureHead(nn.Module):
    """Classical control for the CNOT-free VQC: a fixed per-coordinate feature map, then Linear.

    ``cos``: cos(pi*tanh z_j), what each <Z_j> of the product-state circuit is at zero
    rotation angles. ``cossin``: [cos, sin](pi*tanh z_j); with trained single-qubit
    rotations each <Z_j> = alpha_j*cos(a_j) + beta_j*sin(a_j), so this head contains the
    CNOT-free circuit plus its readout as a special case.
    """

    def __init__(self, in_dim: int, num_classes: int, with_sin: bool) -> None:
        super().__init__()
        self.with_sin = with_sin
        self.fc = nn.Linear(in_dim * (2 if with_sin else 1), num_classes)

    def forward(self, x):
        a = math.pi * torch.tanh(x)
        return self.fc(torch.cat([a.cos(), a.sin()], 1) if self.with_sin else a.cos())


def cnot_ring_supports(n_qubits: int, n_var_layers: int, last_only: bool = False,
                       pattern: str = "ring") -> list[list[int]]:
    """For each readout Z_k, the qubits whose Z it becomes after the VQC's CNOT layers.

    Heisenberg picture with the rotations set to zero: conjugating by CNOT(c, t) maps
    Z_t -> Z_c Z_t and leaves Z_c unchanged, and the circuit's gates are undone in reverse
    order. With angle encoding a_j, <Z_k> is then prod over j in S_k of cos(a_j).
    """
    from quantcnn.models.torch_vqc import cnot_gates

    rings = 1 if last_only else n_var_layers
    gates = cnot_gates(n_qubits, pattern) * rings
    out = []
    for k in range(n_qubits):
        mask = 1 << k
        for c, t in reversed(gates):
            if mask >> t & 1:
                mask ^= 1 << c
        out.append([j for j in range(n_qubits) if mask >> j & 1])
    return out


class CosProductHead(nn.Module):
    """Classical control for the entangled VQC: products of cos(pi*tanh z_j) over each
    readout's support from ``cnot_ring_supports``, then Linear. This is the default
    circuit (``last_only=False``), the last-ring-only circuit, or (``pattern="pairs"``)
    the disjoint-pairs circuits at zero rotation angles."""

    def __init__(self, in_dim: int, num_classes: int, n_var_layers: int = 4, last_only: bool = False,
                 pattern: str = "ring") -> None:
        super().__init__()
        supports = cnot_ring_supports(in_dim, n_var_layers, last_only, pattern)
        mask = torch.zeros(in_dim, in_dim)
        for k, sup in enumerate(supports):
            mask[k, sup] = 1.0
        self.register_buffer("mask", mask)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        c = torch.cos(math.pi * torch.tanh(x))  # (B, n)
        # prod_j c_j^{mask_kj}: take the product over masked entries, others replaced by 1.
        feats = torch.where(self.mask.bool(), c[:, None, :], torch.ones_like(c)[:, None, :]).prod(-1)
        return self.fc(feats)
