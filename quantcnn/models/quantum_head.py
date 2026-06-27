"""PennyLane VQC classifier head with configurable data re-uploading."""

from __future__ import annotations

import math
from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn

ReuploadMode = Literal["none", "light", "strong"]


def _encode_features(features: torch.Tensor, scale: float = 1.0) -> None:
    phi = math.pi * scale * torch.tanh(features)
    for i in range(phi.shape[0]):
        qml.RY(phi[i], wires=i)


def _var_layer(weights_layer: torch.Tensor) -> None:
    n_qubits = weights_layer.shape[0]
    for q in range(n_qubits):
        qml.RY(weights_layer[q, 0], wires=q)
        qml.RZ(weights_layer[q, 1], wires=q)
    for q in range(n_qubits):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])


def _make_device(wires: int):
    """Prefer Lightning for speed; fall back to ``default.qubit`` if unavailable."""
    try:
        return qml.device("lightning.qubit", wires=wires)
    except Exception:
        return qml.device("default.qubit", wires=wires)


def _preferred_diff_method(device) -> str:
    name = str(getattr(device, "name", "") or device).lower()
    if "lightning" in name:
        return "adjoint"
    return "backprop"


def build_qnode(
    n_qubits: int,
    n_var_layers: int,
    mode: ReuploadMode,
    light_second_scale: float = 0.5,
):
    """Factory for a QNode with fixed re-upload schedule & variational depth."""
    dev = _make_device(n_qubits)
    diff = _preferred_diff_method(dev)

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        weights = weights.view(n_var_layers, n_qubits, 2)
        if mode == "none":
            _encode_features(inputs, 1.0)
            for l in range(n_var_layers):
                _var_layer(weights[l])
        elif mode == "strong":
            for l in range(n_var_layers):
                _encode_features(inputs, 1.0)
                _var_layer(weights[l])
        elif mode == "light":
            _encode_features(inputs, 1.0)
            for l in range(n_var_layers):
                _var_layer(weights[l])
                if l < n_var_layers - 1 and l % 2 == 1:
                    _encode_features(inputs * light_second_scale, 1.0)
        else:
            raise ValueError(mode)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


class QuantumVQCLayer(nn.Module):
    """Batched Torch wrapper around TorchLayer-compatible QNode execution."""

    def __init__(
        self,
        n_qubits: int,
        n_var_layers: int,
        mode: ReuploadMode,
    ) -> None:
        super().__init__()
        qnode = build_qnode(n_qubits, n_var_layers, mode)
        weight_shapes = {"weights": (n_var_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        # TorchLayer uses torch.Tensor() (uninitialized memory) by default, which can
        # place rotation angles near ±π where gradients are flat. Small uniform init
        # starts all qubits near the |0⟩ state where gradients are well-defined.
        torch.nn.init.uniform_(self.qlayer.weights, -0.1, 0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        device = features.device
        features = features.cpu()  # PennyLane statevector simulators require CPU tensors
        preds = [self.qlayer(features[b]) for b in range(features.shape[0])]
        return torch.stack(preds).to(device)


class QuantumClassifierHead(nn.Module):
    """VQC expectations followed by linear map to logits (documented hybrid readout)."""

    def __init__(
        self,
        n_qubits: int,
        num_classes: int,
        n_var_layers: int,
        mode: ReuploadMode,
    ) -> None:
        super().__init__()
        self.vqc = QuantumVQCLayer(n_qubits, n_var_layers, mode)
        self.proj = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qs = self.vqc(x)
        return self.proj(qs)


def count_quantum_head_params(n_qubits: int, n_var_layers: int, num_classes: int) -> int:
    """Total trainable params in QuantumClassifierHead: VQC rotations + output linear."""
    rotation_params = n_qubits * 2 * n_var_layers
    linear_params = n_qubits * num_classes + num_classes  # weights + biases
    return rotation_params + linear_params


def suggested_mlp_hidden_for_matching(
    bottleneck: int,
    num_classes: int,
    total_quantum_params: int,
) -> int:
    """Hidden size so MLP total param count ≈ total_quantum_params (weights + biases both layers)."""
    def mlp_params(h: int) -> int:
        return (bottleneck * h + h) + (h * num_classes + num_classes)

    best_h = max(2, bottleneck)
    best_err = abs(mlp_params(best_h) - total_quantum_params)
    for h in range(2, 512):
        err = abs(mlp_params(h) - total_quantum_params)
        if err < best_err:
            best_err, best_h = err, h
    return best_h
