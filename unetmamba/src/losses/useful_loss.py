import torch.nn as nn

from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss


class UnetMambaLoss(nn.Module):
    """
    Official UNetMamba loss adapted for FloodPlanet.

    Main loss:
        Soft Cross Entropy + Dice

    Auxiliary loss:
        Soft Cross Entropy from Local Supervision Module

    Total:
        main_loss + 0.4 * auxiliary_loss
    """

    def __init__(self, ignore_index=-1):
        super().__init__()

        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(
                smooth_factor=0.05,
                ignore_index=ignore_index,
            ),
            DiceLoss(
                smooth=0.05,
                ignore_index=ignore_index,
            ),
            (1.0, 1.0),
        )

        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05,
            ignore_index=ignore_index,
        )

    def forward(self, logits, labels):

        if self.training and isinstance(logits, (tuple, list)):
            logit_main, logit_aux = logits

            loss = (
                self.main_loss(
                    logit_main,
                    labels,
                )
                + 0.4
                * self.aux_loss(
                    logit_aux,
                    labels,
                )
            )

        else:

            loss = self.main_loss(
                logits,
                labels,
            )

        return loss