# config for wider face dataset, TinyYOLOv4# config for pascal VOC

import torch

TOTAL_ANCHORS = 6
NO_OF_ANCHOR_BOX = N = 3  # anchors per scale

S = [13, 26]  # Three output prediction Scales of Yolov3

NO_OF_CLASS = C = 1
HEIGHT = H = 416
WIDTH = W = 416
SCALE = [32, 16]


DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = batch_size = 64


ANCHOR_BOXES = A = torch.tensor(
    [
        [
            [0.04242651279156025, 0.06406801709761986],
            [0.07061653412305392, 0.10405815564669095],
            [0.15996712904710036, 0.23178568253150353],
        ],
        [
            [0.013506840054805461, 0.020800003638634317],
            [0.01904695939559203, 0.029293939471244812],
            [0.027408367166152366, 0.04197697226817791],
        ],
    ]
) * torch.tensor(S).view(
    -1, 1, 1
)  # sale up in range 0-S[i], accordingly


CLASS_ENCODING = class_encoding = {
    "face": 1,
}

class_decoding = {v: k for k, v in class_encoding.items()}

COLORS = ["#0000FF", "#0000FF"]
