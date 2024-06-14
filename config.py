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


ANCHOR_BOXES = A = torch.tensor(
    [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    ]
) * torch.tensor(S).view(
    -1, 1, 1
)  # sale up in range 0-S[i], accordingly


CLASS_ENCODING = class_encoding = {
    "face": 1,
}

class_decoding = {v: k for k, v in class_encoding.items()}

COLORS = ["#0000FF"]
