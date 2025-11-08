# PyTorch Single Shot Detector (SSD)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


class SSD(nn.Module):
    def __init__(self, num_classes, # TODO):
        super(SSD, self).__init__()

        self.num_classes = num_classes

        # TODO: implement the model (use SSD with pretrained weights from torchvision)
        # torchvision.models.detection.SSD300_VGG16_Weights(value)
        # The model builder above accepts the following values as the weights parameter. SSD300_VGG16_Weights.DEFAULT is equivalent to SSD300_VGG16_Weights.COCO_V1. You can also use strings, e.g. weights='DEFAULT' or weights='COCO_V1'.

class FasterRCNN(nn.Module):

    #TODO: torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights(value)
    pass



# Pretty close to what we need
def get_model(model_conf: Dict[str, Any]) -> torch.nn.Module:
    """Select model to run experiments"""
    label = model_conf["label"]
    if label == "linear":
        return SimpleClassifier(
            model_conf["num_channels"],
            model_conf["image_height"],
            model_conf["image_width"],
            model_conf["num_classes"],
        )
    if label == "conv":
        return ConvClassifier(model_conf["num_classes"])
    else:
        raise ValueError(f"There is no such model with label {label}")