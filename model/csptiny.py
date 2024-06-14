# Backbone implementation with pretrained weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


class Conv2dBatchLeaky(nn.Module):
    """
    This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation="leaky",
        leaky_slope=26 / 256,
    ):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(k / 2) for k in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope
        # self.mish = Mish()

        # Layer
        if activation == "leaky":
            self.layers = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.LeakyReLU(self.leaky_slope),
            )

    def __repr__(self):
        s = "{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class CSPBlock(nn.Module):

    def __init__(self, nchannels):
        super().__init__()

        self.conv1 = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation="leaky")

        self.conv2 = Conv2dBatchLeaky(
            nchannels, 2 * nchannels, 3, 1, activation="leaky"
        )

        self.split0 = Conv2dBatchLeaky(
            2 * nchannels, nchannels, 1, 1, activation="leaky"
        )
        self.split1 = Conv2dBatchLeaky(
            2 * nchannels, nchannels, 1, 1, activation="leaky"
        )

        # residual
        self.conv3 = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation="leaky")
        self.conv4 = Conv2dBatchLeaky(nchannels, nchannels, 3, 1, activation="leaky")

        self.conv5 = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation="leaky")

    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)

        split0 = self.split0(conv2)
        split1 = self.split1(conv2)
        conv3 = self.conv3(split1)
        conv4 = self.conv4(conv3)

        shortcut = split1 + conv4
        conv5 = self.conv5(shortcut)

        route = torch.cat([split0, conv5], dim=1)
        return route


class CSPTiny(nn.Module):
    def __init__(self):
        super().__init__()

        input_channels = 32

        # Network , ref:https://wikidocs.net/images/page/176269/Yolo_V4_tiny_Archtecture.png

        self.stage1 = Conv2dBatchLeaky(3, input_channels, 3, 2, activation="leaky")

        self.stage2 = Conv2dBatchLeaky(
            input_channels, 2 * input_channels, 3, 2, activation="leaky"
        )
        self.stage3 = CSPBlock(2 * input_channels)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.stage4 = CSPBlock(4 * input_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.stage5 = CSPBlock(8 * input_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.stage6 = Conv2dBatchLeaky(
            16 * input_channels, 16 * input_channels, 3, 1, activation="leaky"
        )

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        p1 = self.pool1(stage3)
        stage4 = self.stage4(p1)
        p2 = self.pool2(stage4)
        stage5 = self.stage5(p2)
        p3 = self.pool3(stage5)
        stage6 = self.stage6(p3)

        return [stage6, stage5]

    def _init_wts(model, path):
        wts = torch.load(path)
        model.load_state_dict(wts)
        print("Weight Copied")
        return model
