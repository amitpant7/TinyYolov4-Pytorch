# config for wider face dataset, TinyYOLOv4# config for pascal VOC

import torch

NO_OF_ANCHOR_BOX = N = 3

S = [13, 26]  # Three output prediction Scales of Yolov3

NO_OF_CLASS = C = 20
HEIGHT = H = 416
WIDTH = W = 416
SCALE = [32, 16]


DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = batch_size = 16


anchors = torch.tensor(
    [(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)]
)


ANCHOR_BOXES = A = torch.tensor(
    [
        [[0.2788, 0.2163], [0.3750, 0.4760], [0.8966, 0.7837]],
        [[0.0721, 0.1466], [0.1490, 0.1082], [0.1418, 0.2861]],
    ]
) * torch.tensor(S).view(
    -1, 1, 1
)  # sale up in range 0-S[i], accordingly


CLASS_ENCODING = class_encoding = {
    "face": 1,
}

class_decoding = {v: k for k, v in class_encoding.items()}

COLORS = ["#0000FF"]
