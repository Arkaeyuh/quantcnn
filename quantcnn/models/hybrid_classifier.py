"""Full hybrid model tying backbone + head."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from quantcnn.models.backbone import CifarCNN, SmallCNN
from quantcnn.models.classical_heads import LinearHead, MLPHead
from quantcnn.models.quantum_head import (
    QuantumClassifierHead,
    ReuploadMode,
    count_quantum_head_params,
    suggested_mlp_hidden_for_matching,
)

DatasetName = Literal["mnist", "cifar10"]
HeadKind = Literal["linear", "mlp", "quantum"]


class HybridClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        bottleneck_dim: int | None,
        backbone: DatasetName | str,
        head: HeadKind,
        in_channels: int,
        mlp_hidden: int | None = None,
        n_var_layers: int = 4,
        reupload_mode: ReuploadMode = "none",
    ) -> None:
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = 8
        b = bottleneck_dim

        bk = backbone.lower()
        if bk == "mnist":
            self.backbone = SmallCNN(b, in_channels=in_channels)
        elif bk == "cifar10":
            self.backbone = CifarCNN(b, in_channels=in_channels)
        else:
            raise ValueError(backbone)

        if head == "linear":
            self.head = LinearHead(b, num_classes)
        elif head == "mlp":
            if mlp_hidden is None:
                total_qpc = count_quantum_head_params(b, n_var_layers, num_classes)
                hid = suggested_mlp_hidden_for_matching(b, num_classes, total_qpc)
            else:
                hid = mlp_hidden
            self.head = MLPHead(b, hid, num_classes)
        elif head == "quantum":
            self.head = QuantumClassifierHead(b, num_classes, n_var_layers, reupload_mode)
        else:
            raise ValueError(head)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)
