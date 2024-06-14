import torch
from torch import nn
from .csptiny import Conv2dBatchLeaky


class Neck(nn.Module):
    def __init__(self, input_channels=512):
        super().__init__()
        # for the first input which is the last input from backbone
        self.first_block1 = Conv2dBatchLeaky(
            input_channels, input_channels // 2, 3, 1, activation="leaky"
        )
        self.first_block2 = Conv2dBatchLeaky(
            input_channels // 2, input_channels, 3, 1, activation="leaky"
        )
        self.pointconv1 = Conv2dBatchLeaky(
            input_channels, input_channels, 1, 1, activation="leaky"
        )

        # for second input
        self.sec_block1 = Conv2dBatchLeaky(
            input_channels, input_channels // 2, 3, 1, activation="leaky"
        )
        self.upsample = nn.Upsample(scale_factor=2)
        self.sec_block2 = Conv2dBatchLeaky(
            input_channels, input_channels // 2, 3, 1, activation="leaky"
        )
        self.pointconv2 = Conv2dBatchLeaky(
            input_channels // 2, input_channels // 2, 1, 1, activation="leaky"
        )

    def forward(self, data):  # list  [x1, x2]
        x1, x2 = data[0], data[1]
        x1 = self.first_block1(x1)
        out1 = self.first_block2(x1)
        out1 = self.pointconv1(out1)

        x2 = self.sec_block1(x2)
        out2 = torch.cat([x2, self.upsample(x1)], dim=1)
        out2 = self.sec_block2(out2)
        out2 = self.pointconv2(out2)

        return [out1, out2]  # 256x26x26 and 512x13x13
