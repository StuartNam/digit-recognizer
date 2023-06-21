import torch
import torch.nn as nn

from config import *

class ConvBlock(nn.Module):
    def __init__(self, num_convlayers, num_in_channels, channels_scale):
        self.layers = nn.Sequential()

        for i in range(num_convlayers):
            self.layers.add_module(
                name = "Conv{}".format(i),
                module = nn.Conv2d(
                    in_channels = num_in_channels * (channels_scale ** i),
                    out_channels = num_in_channels * (channels_scale ** i),
                    kernel_size = 3,
                    padding = 1,
                    dtype = DTYPE
                )
            )
        
        self.layers.add_module(
            name = "BN",
            module = nn.BatchNorm2d(
                num_features = num_in_channels * (channels_scale ** (num_convlayers - 1)),
                dtype = DTYPE
            )
        )

        self.layers.add_module(
            name = "MP",
            module = nn.MaxPool2d(
                kernel_size = 2,
                stride = 2
            )
        )

        self.layers.add_module(
            name = "ReLU",
            module = nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class ConvTranposeBlock(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_convlayers, out = False):
        super().__init__()

        self.layers = nn.Sequential()

        for i in range(num_convlayers):
            self.layers.add_module(
                name = "ConvTranspose{}".format(i),
                module = nn.ConvTranspose2d(
                    in_channels = num_in_channels,
                    out_channels = num_in_channels,
                    kernel_size = 2,
                    stride = 2,
                    dtype = DTYPE
                )
            )
        
        self.layers.add_module(
            name = "Conv",
            module = nn.Conv2d(
                in_channels = num_in_channels,
                out_channels = num_out_channels,
                kernel_size = 1,
                dtype = DTYPE
            )
        )

        self.layers.add_module(
            name = "BN",
            module = nn.BatchNorm2d(
                num_features = num_in_channels,
                dtype = DTYPE
            )
        )

        if not out:
            self.layers.add_module(
                name = "ReLU",
                module = nn.ReLU()
            )
        
    def forward(self, x):
        return self.layers(x)
    
class Unflatten(nn.Module):
    def __init__(self, new_shape):
        super().__init__()

        self.new_shape = new_shape

    def forward(self, x):
        return x.reshape(-1, 1, self.new_shape[0], self.new_shape[1])
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # x: (N, 1, 28, 28), y: (N, 256, 7, 7)
        self.encoder = nn.Sequential(
            # x: (N, 1, 28, 28), y: (N, 16, 14, 14)
            ConvBlock(
                num_convlayers = 2,
                num_in_channels = 1,
                channels_scale = 4
            ),

            # x: (N, 16, 14, 14), y: (N, 256, 7, 7)
            ConvBlock(
                num_convlayers = 2,
                num_in_channels = 1,
                channels_scale = 4
            ),

            nn.Flatten(),

            nn.Linear(
                in_features = 12544,
                out_features = 100,
                dtype = DTYPE
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features = 100,
                out_features = 12544,
                dtype = DTYPE
            ),

            Unflatten(
                new_shape = (28, 28)
            ),

            ConvTranposeBlock(
                num_in_channels = 256,
                num_out_channels = 16,
                num_convlayers = 2
            ),

            ConvTranposeBlock(
                num_in_channels = 16,
                num_out_channels = 1,
                num_convlayers = 2,
                out = True
            ),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)
        

        