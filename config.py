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
        [[0.04669265, 0.07010818], [0.08200667, 0.11690065], [0.19566397, 0.2667503]],
        [[0.01401189, 0.02136289], [0.02018682, 0.03110242], [0.02997164, 0.04524642]],
    ]
) * torch.tensor(S).view(
    -1, 1, 1
)  # sale up in range 0-S[i], accordingly


CLASS_ENCODING = class_encoding = {
    "face": 1,
}

class_decoding = {v: k for k, v in class_encoding.items()}

COLORS = ["#0000FF", "#0000FF"]


ANCHORS_9 = torch.tensor(
    [
        [[0.04902728, 0.07361358], [0.086107, 0.12274568], [0.20544716, 0.28008781]],
        [[0.01471248, 0.02243103], [0.02119616, 0.03265754], [0.03147022, 0.04750875]],
        [[0.021, 0.02483848], [0.00668075, 0.00949571], [0.01013696, 0.01530269]],
    ]
)
