#======================================================================================#
#   MODEL DEFINITION
#   - Define model in this file
#======================================================================================#

import torch.nn as nn

from config import *

class ResBlock(nn.Module):
    def __init__(self, num_in_channels, num_inside_channels, is_head = False, dtype = torch.float32):
        super().__init__()

        if not is_head:
            self.layers = nn.Sequential(
                nn.ReLU(),

                nn.Conv2d(
                    in_channels = num_in_channels,
                    out_channels = num_inside_channels,
                    kernel_size = 3,
                    padding = 1,
                    dtype = dtype
                ),

                nn.Conv2d(
                    in_channels = num_inside_channels,
                    out_channels = num_in_channels,
                    kernel_size = 1,
                    dtype = dtype
                ),

                nn.BatchNorm2d(
                    num_features = 1
                )
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels = num_in_channels,
                    out_channels = num_inside_channels,
                    kernel_size = 3,
                    padding = 1,
                    dtype = dtype
                ),

                nn.Conv2d(
                    in_channels = num_inside_channels,
                    out_channels = 1,
                    kernel_size = 1,
                    dtype = dtype
                ),

                nn.BatchNorm2d(
                    num_features = 1
                )
            )

    def forward(self, x):
        # y = self.layers(x)
        # print(y.shape)
        return self.layers(x)
    
class ResNet(nn.Module):
    def __init__(self, num_in_channels, out_features, depth, dtype = torch.float32):
        super().__init__()
        
        res_blocks = nn.Sequential()
        res_blocks.add_module(
            name = "rb0",
            module = ResBlock(num_in_channels, 16, is_head = True)
        )

        for i in range(depth - 1):
            res_blocks.add_module(
                name = "rb{}".format(i + 1),
                module = ResBlock(num_in_channels, 16))
        
        self.res_blocks = res_blocks

        self.out_layers = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(784, out_features, dtype = dtype)
        )

    def forward(self, x):
        y = x

        for _, res_block in self.res_blocks.named_children():
            tmp = y
            y = res_block(y)
            y = tmp + y

        y = self.out_layers(y)

        return y

class ConvBlock(nn.Module):
    def __init__(self, num_convlayers, num_in_channels, channels_scale):
        super().__init__()

        self.layers = nn.Sequential()

        for i in range(num_convlayers):
            self.layers.add_module(
                name = "Conv{}".format(i),
                module = nn.Conv2d(
                    in_channels = num_in_channels * (channels_scale ** i),
                    out_channels = num_in_channels * (channels_scale ** (i + 1)),
                    kernel_size = 3,
                    padding = 1,
                    dtype = DTYPE
                )
            )
        
        self.layers.add_module(
            name = "BN",
            module = nn.BatchNorm2d(
                num_features = num_in_channels * (channels_scale ** num_convlayers),
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
                num_features = num_out_channels,
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
    
class Reshape(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()

        self.new_shape = (C, H, W)

    def forward(self, x):
        return x.reshape(-1, self.new_shape[0], self.new_shape[1], self.new_shape[2])