from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN, BACKBONE
from .yolox import YOLOX

def get_model(depth, width, num_classes=80, in_channels=[256, 512, 1024], act="silu", backbone_type="origin", depthwise=False):
    backbone = BACKBONE[backbone_type](depth, width, in_channels=in_channels, act=act, depthwise=depthwise)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
    return YOLOX(backbone, head)
