import copy
from .yolov5_backbone import YoloV5BackBone
from .yolov8_backbone import YoloV8BackBone
from .yolov7_backbone import YoloV7BackBone

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.Model.Backbone.name
    if name == "YoloV5":
        return YoloV5BackBone(backbone_cfg)
    elif name == "YoloV7":
        return YoloV7BackBone(backbone_cfg)
    elif name == "YoloV8":
        return YoloV8BackBone(backbone_cfg)
    else:
        raise NotImplementedError