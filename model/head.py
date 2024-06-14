import torch
import torch.nn as nn

from .csptiny import Conv2dBatchLeaky


class Head(nn.Module):
    def __init__(self, num_classes, no_of_anchors):
        super().__init__()
        self.output_channels = (5 + num_classes) * no_of_anchors

        self.out1 = nn.Sequential(
            Conv2dBatchLeaky(512, 1024, 3, 1, activation="leaky"),
            Conv2dBatchLeaky(1024, self.output_channels, 1, 1, activation="leaky"),
        )

        self.out2 = nn.Sequential(
            Conv2dBatchLeaky(256, 512, 3, 1, activation="leaky"),
            Conv2dBatchLeaky(512, self.output_channels, 1, 1, activation="leaky"),
        )

    def forward(self, x):  # x is list of three features
        out1 = self.out1(x[0])
        out2 = self.out2(x[1])

        return out1, out2
