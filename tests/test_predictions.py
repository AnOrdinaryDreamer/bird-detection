import torch

from bird_detection.detection.dataloaders import DatasetMetadata
from bird_detection.detection.predictions import (
    ApiPrediction,
    format_prediction_for_api,
)


def _metadata():
    return DatasetMetadata(
        id2label={
            0: "__background__",
            1: "sparrow",
            2: "squirrel",
        },
        label2id={
            "__background__": 0,
            "sparrow": 1,
            "squirrel": 2,
        },
    )


def test_format_prediction_filters_background_and_threshold():
    raw = {
        "boxes": torch.tensor(
            [[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 2, 2]], dtype=torch.float32
        ),
        "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
        "scores": torch.tensor([0.99, 0.2, 0.95], dtype=torch.float32),
    }

    predictions = format_prediction_for_api(
        raw, metadata=_metadata(), score_threshold=0.5
    )

    assert len(predictions) == 1
    assert isinstance(predictions[0], ApiPrediction)
    assert predictions[0].label == "squirrel"
    assert predictions[0].bbox == {
        "x_min": 1.0,
        "y_min": 1.0,
        "x_max": 2.0,
        "y_max": 2.0,
    }
    assert 0.94 < predictions[0].score < 0.96


def test_format_prediction_returns_not_found_when_no_hits():
    raw = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "scores": torch.zeros((0,), dtype=torch.float32),
    }
    predictions = format_prediction_for_api(
        raw, metadata=_metadata(), score_threshold=0.5
    )

    assert len(predictions) == 1
    assert predictions[0].label == "not_found"
    assert predictions[0].bbox is None


def test_format_prediction_limits_results_and_includes_background_when_requested():
    raw = {
        "boxes": torch.tensor([[0, 0, 3, 3], [2, 2, 4, 4]], dtype=torch.float32),
        "labels": torch.tensor([0, 1], dtype=torch.int64),
        "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
    }

    predictions = format_prediction_for_api(
        raw,
        metadata=_metadata(),
        score_threshold=0.1,
        max_detections=1,
        include_background=True,
    )

    assert len(predictions) == 1
    assert predictions[0].label == "__background__"
    assert predictions[0].bbox == {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 3.0,
        "y_max": 3.0,
    }
