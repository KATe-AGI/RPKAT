"""
Implements the knowledge distillation loss, proposed in deit
"""
import torch
from torch.nn import functional as F


class Loss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, criterion: torch.nn.Module):
        super().__init__()
        self.criterion = criterion

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        loss = self.criterion(outputs, labels)
        return loss
