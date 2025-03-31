"""
model.py

Contains the advanced neural network model (with a deeper residual stack)
and any building blocks needed for it (like ResidualBlock).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ACTION_SIZE

class ResidualBlock(nn.Module):
    """
    A basic residual block that consists of:
      - Two convolutional layers with batch normalization
      - ReLU activation
      - A skip connection that adds the input to the output of the block
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AdvancedChessNet(nn.Module):
    """
    A deeper CNN for chess board evaluation.
    - The input has shape (batch, 15, 8, 8) now because we track:
        12 channels for piece presence,
        1 channel for piece value (like before),
        1 channel for side to move,
        1 channel for castling rights.
    - We use 32 residual blocks with 512 channels each.
    - Output:
        policy: a distribution over all possible moves (ACTION_SIZE=4672)
        value: single scalar estimate of board evaluation
        (We also maintain a trainable materialWeight parameter.)
    """
    def __init__(self):
        super(AdvancedChessNet, self).__init__()
        
        # We add a trainable scalar for weighting material inside the network.
        self.materialWeight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.convBlock = nn.Sequential(
            nn.Conv2d(15, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # 32 residual blocks
        self.residualBlocks = nn.Sequential(*[ResidualBlock(512) for _ in range(32)])

        # Policy head
        self.policyHead = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 1024),   # First dense layer
            nn.ReLU(),
            nn.Linear(1024, ACTION_SIZE)  # Output layer
        )
        
        # Value head
        self.valueHead = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # This outputs raw network value
        )

    def forward(self, x, materialTensor=None):
        """
        The main forward pass.
        'x' is the board input of shape (batch, 15, 8, 8).
        'materialTensor' is an optional shape (batch,) that represents
        the precomputed material score for each sample.
        """
        x = self.convBlock(x)
        x = self.residualBlocks(x)

        policy = self.policyHead(x)
        rawValue = self.valueHead(x)  # shape (batch, 1)

        # If user provided a materialTensor, we incorporate it with our trainable weight
        if materialTensor is not None:
            # shape (batch, 1)
            rawValue = rawValue + self.materialWeight * materialTensor.unsqueeze(1)

        # Typically, we clamp or scale the final value if desired
        return policy, rawValue