import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from bird_detection.detection.topk import enable_topk_predictions


def test_enable_topk_predictions_adds_alternatives_to_fasterrcnn():
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    enable_topk_predictions(model, topk=3)
    model.eval()

    images = [torch.rand(3, 64, 64)]
    with torch.no_grad():
        outputs = model(images)

    assert isinstance(outputs, list)
    assert outputs, "Model should return at least one detection dict per image"
    assert "topk_labels" in outputs[0]
    assert "topk_scores" in outputs[0]
    assert outputs[0]["topk_scores"].shape[1] <= 3
