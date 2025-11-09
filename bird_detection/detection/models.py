"""Model factory for object detection architectures."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssd300_vgg16,
)
from torchvision.models import VGG16_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead


def _resolve_weights(enum_cls: Type, name: Optional[str]):
    if not name:
        return None
    if isinstance(name, enum_cls):
        return name
    attr_name = name
    if not hasattr(enum_cls, attr_name):
        attr_name = name.upper()
    return getattr(enum_cls, attr_name)


def _freeze_backbone_parameters(model: nn.Module) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False


def build_fasterrcnn(
    model_cfg: Dict[str, Any],
    num_classes: int,
) -> nn.Module:
    weights = None
    if model_cfg.get("pretrained", True):
        weight_name = model_cfg.get("weights", "DEFAULT")
        weights = _resolve_weights(
            FasterRCNN_MobileNet_V3_Large_FPN_Weights, weight_name
        )
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        min_size=model_cfg.get("min_size", 640),
        max_size=model_cfg.get("max_size", 1280),
        trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
    )
    if model_cfg.get("freeze_backbone", False):
        _freeze_backbone_parameters(model)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_ssd(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    weights = None
    if model_cfg.get("pretrained", True):
        weight_name = model_cfg.get("weights", "DEFAULT")
        weights = _resolve_weights(SSD300_VGG16_Weights, weight_name)
    backbone_flag = model_cfg.get("pretrained_backbone")
    if backbone_flag is None:
        backbone_flag = model_cfg.get("pretrained", True)
    backbone_weights = None
    if backbone_flag:
        backbone_name = model_cfg.get("weights_backbone", "IMAGENET1K_V1")
        backbone_weights = _resolve_weights(VGG16_Weights, backbone_name)
    model = ssd300_vgg16(weights=weights, weights_backbone=backbone_weights)
    if model_cfg.get("freeze_backbone", False):
        _freeze_backbone_parameters(model)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model


MODEL_BUILDERS = {
    "fasterrcnn_mobilenet_v3_large_fpn": build_fasterrcnn,
    "ssd300_vgg16": build_ssd,
}


def build_model(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    """Instantiate a detection model declared in the config."""
    name = model_cfg.get("name")
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unsupported model '{name}'. Available: {list(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[name](model_cfg, num_classes)
