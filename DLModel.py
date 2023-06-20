import torch
import torch.nn as nn

import pandas as pd

from dataset import *
from config import *

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.layers = nn.Sequential(
            # In: (N, 1, 28, 28) - Out: (N, 16, 28, 28)
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                padding = 1,
                dtype = torch.float32
            ),

            # In: (N, 16, 28, 28) - Out: (N, 64, 28, 28)
            nn.Conv2d(
                in_channels = 16,
                out_channels = 64,
                kernel_size = 3,
                padding = 1,
                dtype = torch.float32
            ),

            # In: (N, 64, 28, 28) - Out: (N, 64, 14, 14)
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2
            ),

            nn.BatchNorm2d(
                num_features = 64
            ),

            nn.ReLU(),

            # In: (N, 64, 14, 14) - Out: (N, 128, 14, 14)
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                padding = 1,
                dtype = torch.float32
            ),

            # In: (N, 128, 14, 14) - Out: (N, 256, 14, 14)
            nn.Conv2d(
                in_channels = 128,
                out_channels = 256,
                kernel_size = 3,
                padding = 1,
                dtype = torch.float32
            ),

            # In: (N, 256, 14, 14) - Out: (N, 256, 7, 7)
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2
            ),

            nn.BatchNorm2d(
                num_features = 256
            ),

            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                in_features = 12544,
                out_features = 4096
            ),

            nn.BatchNorm1d(
                num_features = 4096
            ),

            nn.ReLU(),

            nn.Linear(
                in_features = 4096,
                out_features = 4096
            ),

            nn.BatchNorm1d(
                num_features = 4096
            ),

            nn.ReLU(),

            nn.Linear(
                in_features = 4096,
                out_features = 10
            )
        )

    def forward(self, x):
        return self.layers(x)