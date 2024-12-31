"""
loss function
"""
import torch
from torch.nn import functional as F


class Loss(torch.nn.Module):

    def __init__(self, criterion: torch.nn.Module):
        super().__init__()
        # Initialize the criterion (e.g., CrossEntropyLoss, MSELoss) that will be used to compute the standard loss
        self.criterion = criterion

    def forward(self, inputs, outputs, labels):
        # Compute the loss using the provided criterion
        loss = self.criterion(outputs, labels)
        return loss

