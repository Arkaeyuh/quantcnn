"""Full hybrid model tying backbone + head."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from quantcnn.models.backbone import CifarCNN, PretrainedResNet18, SmallCNN
from quantcnn.models.classical_heads import DESIGNED_SUPPORTS, CosProductHead, LinearHead, MLPHead, TrigFeatureHead
from quantcnn.models.quantum_head import (
    QuantumClassifierHead,
    ReuploadMode,
    count_quantum_head_params,
    suggested_mlp_hidden_for_matching,
)

DatasetName = Literal["mnist", "cifar10"]
HeadKind = Literal["linear", "mlp", "quantum", "cos", "cossin", "cosprod", "cosprod_last", "cosprod_pairs_last",
                   "cosprod_win2", "cosprod_win3", "cosprod_win4", "cosprod_star2", "cosprod_star3",
                   "cosprod_gap4", "cosprod_sg", "cosprod_last_sg", "cosprod_fl03", "cosprod_fl1",
                   "cosprod_last_fl03", "cosprod_last_fl1"]
# gradient-floor heads (attenuation test): suffix -> floor on |product of the other factors|
GRAD_FLOORS = {"_fl03": 0.3, "_fl1": 1.0}


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
        quantum_sim: str = "pennylane",
        pre_norm: str = "none",
        encoding: str = "pi_tanh",
        entangle: bool | str = True,
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
        elif bk == "resnet18_pt":
            self.backbone = PretrainedResNet18(b)
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
        elif head in ("cos", "cossin"):
            self.head = TrigFeatureHead(b, num_classes, with_sin=head == "cossin")
        elif head in ("cosprod", "cosprod_last", "cosprod_pairs_last", "cosprod_sg", "cosprod_last_sg"):
            self.head = CosProductHead(b, num_classes, n_var_layers,
                                       last_only=head not in ("cosprod", "cosprod_sg"),
                                       pattern="pairs" if head == "cosprod_pairs_last" else "ring",
                                       decouple_grad=head.endswith("_sg"))
        elif head.startswith(("cosprod_fl", "cosprod_last_fl")):
            self.head = CosProductHead(b, num_classes, n_var_layers, last_only=head.startswith("cosprod_last"),
                                       grad_floor=GRAD_FLOORS[head[head.rindex("_"):]])
        elif head.startswith("cosprod_") and head[len("cosprod_"):] in DESIGNED_SUPPORTS:
            self.head = CosProductHead(b, num_classes, supports=DESIGNED_SUPPORTS[head[len("cosprod_"):]])
        elif head == "quantum":
            self.head = QuantumClassifierHead(b, num_classes, n_var_layers, reupload_mode, sim=quantum_sim,
                                              encoding=encoding, entangle=entangle)
        else:
            raise ValueError(head)

        # Optional normalization between bottleneck and head ("bn": batch statistics
        # plus a learned per-dimension affine). Trained with the head's parameter group.
        if pre_norm == "bn":
            self.pre_norm = nn.BatchNorm1d(b)
        elif pre_norm == "none":
            self.pre_norm = nn.Identity()
        else:
            raise ValueError(pre_norm)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(self.pre_norm(z))
