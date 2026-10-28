"""Classical classifier heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from quantcnn.models.torch_vqc import EscapeTanh


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

    layers = [n_var_layers - 1] if last_only else range(n_var_layers)
    gates = [g for l in layers for g in cnot_gates(n_qubits, pattern, l)]
    out = []
    for k in range(n_qubits):
        mask = 1 << k
        for c, t in reversed(gates):
            if mask >> t & 1:
                mask ^= 1 << c
        out.append([j for j in range(n_qubits) if mask >> j & 1])
    return out


# Designed supports for the order-vs-overlap control (8 inputs). ``win<w>``: each readout is a
# window of w consecutive inputs, so every input sits in w supports. ``star<w>``: supports
# share inputs 0 (and 1), concentrating the overlap; star2 approximately matches win3's mean
# pairwise overlap (23/28 vs 24/28) at order 2, star3 matches win4's exactly (12/7) at order 3.
# Support-matrix ranks: win2 7, win4 5, all others 8; ``gap4`` ({k,k+1,k+2,k+4}) is a full-rank
# order-4 layout with star3's overlap.
DESIGNED_SUPPORTS = {
    **{f"win{w}": [[(k + i) % 8 for i in range(w)] for k in range(8)] for w in (2, 3, 4)},
    "star2": [[1, 2]] + [[0, k] for k in range(1, 8)],
    "star3": [[1, 2, 3]] + [[0, 1, k] for k in range(2, 8)] + [[0, 2, 3]],
    "gap4": [[(k + i) % 8 for i in (0, 1, 2, 4)] for k in range(8)],
}


class _FlooredProduct(torch.autograd.Function):
    """feat_k = prod_{j in S_k} c_j. Backward replaces d feat_k / d c_j = P_kj (the product of the
    other factors) by sign(P_kj) * max(|P_kj|, floor): floor = 0 is the true gradient; floor = 1
    keeps only its sign, removing all magnitude attenuation; values between change only the paths
    attenuated below the floor."""

    @staticmethod
    def forward(ctx, c, mask, floor):
        ctx.save_for_backward(c, mask)
        ctx.floor = floor
        return torch.where(mask.bool(), c[:, None, :], torch.ones_like(c)[:, None, :]).prod(-1)

    @staticmethod
    def backward(ctx, grad_out):
        c, mask = ctx.saved_tensors
        n = c.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=c.device)
        # others[b, k, j, i] = c_i for i in S_k, i != j; else 1
        keep = mask.bool()[:, None, :] & ~eye[None, :, :]
        others = torch.where(keep[None], c[:, None, None, :], torch.ones_like(c)[:, None, None, :]).prod(-1)
        sign = torch.where(others < 0, -torch.ones_like(others), torch.ones_like(others))
        d = sign * others.abs().clamp_min(ctx.floor) * mask[None]  # (B, k, j)
        return (grad_out[:, :, None] * d).sum(1), None, None


class CosProductHead(nn.Module):
    """Classical control for the entangled VQC: products of cos(pi*tanh z_j) over each
    readout's support from ``cnot_ring_supports``, then Linear. This is the default
    circuit (``last_only=False``), the last-ring-only circuit, or (``pattern="pairs"``)
    the disjoint-pairs circuits at zero rotation angles; or explicit ``supports``.
    ``decouple_grad`` keeps the forward products but removes the multiplicative coupling
    from the backward pass (mechanism test). ``grad_floor`` keeps the forward products and the
    sign of each path's gradient but floors its magnitude (``_FlooredProduct``). ``st_tanh`` keeps
    the forward products but passes d tanh(z)/dz = 1 backward (straight-through; saturation test);
    ``st_tanh="esc"`` does so only for inward steps (``EscapeTanh``)."""

    def __init__(self, in_dim: int, num_classes: int, n_var_layers: int = 4, last_only: bool = False,
                 pattern: str = "ring", supports: list[list[int]] | None = None,
                 decouple_grad: bool = False, grad_floor: float | None = None, st_tanh: bool | str = False) -> None:
        super().__init__()
        self.st_tanh = st_tanh
        self.decouple_grad = decouple_grad
        self.grad_floor = grad_floor
        if supports is None:
            supports = cnot_ring_supports(in_dim, n_var_layers, last_only, pattern)
        mask = torch.zeros(in_dim, in_dim)
        for k, sup in enumerate(supports):
            mask[k, sup] = 1.0
        self.register_buffer("mask", mask)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        if self.st_tanh == "esc":
            t = EscapeTanh.apply(x)
        else:
            t = torch.tanh(x).detach() + x - x.detach() if self.st_tanh else torch.tanh(x)
        c = torch.cos(math.pi * t)  # (B, n)
        # prod_j c_j^{mask_kj}: take the product over masked entries, others replaced by 1.
        if self.grad_floor is not None:
            return self.fc(_FlooredProduct.apply(c, self.mask, self.grad_floor))
        feats = torch.where(self.mask.bool(), c[:, None, :], torch.ones_like(c)[:, None, :]).prod(-1)
        if self.decouple_grad:
            # Same forward value; backward passes d feat_k / d c_j = 1 for j in S_k, i.e. each
            # input's gradient no longer scales with the other factors of the product.
            own = (self.mask * (c - c.detach())[:, None, :]).sum(-1)
            feats = feats.detach() + own
        return self.fc(feats)
