import albumentations as A
import numpy as np
import pytest
import torch

from bird_detection.detection.augmentations import DetectionTransformPipeline


@pytest.fixture
def sample_detection_inputs():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    boxes = np.array([[1.0, 0.5, 3.0, 2.5]], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)
    return image, boxes, labels


def test_pipeline_outputs_tensor_in_unit_range():
    image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    boxes = np.array([[1, 2, 10, 12]], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)

    pipeline = DetectionTransformPipeline(aug=None, mean=None, std=None)

    tensor_image, tensor_boxes, tensor_labels = pipeline(image, boxes, labels)

    assert tensor_image.dtype == torch.float32
    assert tensor_image.min().item() >= 0.0
    assert tensor_image.max().item() <= 1.0
    assert torch.equal(tensor_boxes, torch.tensor(boxes))
    assert torch.equal(tensor_labels, torch.tensor(labels))


def test_pipeline_applies_normalization():
    image = np.full((8, 8, 3), fill_value=255, dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    labels = np.zeros((0,), dtype=np.int64)

    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    pipeline = DetectionTransformPipeline(aug=None, mean=mean, std=std)

    tensor_image, tensor_boxes, tensor_labels = pipeline(image, boxes, labels)

    expected_value = torch.tensor((1.0 - 0.5) / 0.25, dtype=torch.float32)
    assert torch.allclose(tensor_image[0, 0, 0], expected_value)
    assert tensor_boxes.shape == (0, 4)
    assert tensor_labels.shape == (0,)


def test_pipeline_resize_scales_image_and_boxes():
    image = np.zeros((20, 10, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 10, 20]], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)
    resize_cfg = {"enabled": True, "target_longest_side": 10, "tolerance": 0.0}

    pipeline = DetectionTransformPipeline(
        aug=None, mean=None, std=None, resize_cfg=resize_cfg
    )

    tensor_image, tensor_boxes, tensor_labels = pipeline(image, boxes, labels)

    assert tensor_image.shape == (3, 10, 5)
    assert tensor_boxes.shape == (1, 4)
    assert torch.allclose(tensor_boxes[0], torch.tensor([0.0, 0.0, 5.0, 10.0]))
    assert tensor_labels.tolist() == [1]


def test_pipeline_applies_albumentations_flip(sample_detection_inputs):
    image, boxes, labels = sample_detection_inputs
    aug = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    pipeline = DetectionTransformPipeline(aug=aug, mean=None, std=None)

    _, tensor_boxes, tensor_labels = pipeline(image, boxes, labels)

    assert tensor_labels.tolist() == [2]
    expected = torch.tensor([[3.0, 0.5, 5.0, 2.5]], dtype=torch.float32)
    assert torch.allclose(tensor_boxes, expected)


class DummyRemoveBoxes:
    def __call__(self, image, bboxes, labels):
        return {
            "image": image,
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "labels": [],
        }


def test_pipeline_handles_augmented_empty_boxes(sample_detection_inputs):
    image, boxes, labels = sample_detection_inputs
    pipeline = DetectionTransformPipeline(
        aug=DummyRemoveBoxes(),
        mean=None,
        std=None,
    )

    _, tensor_boxes, tensor_labels = pipeline(image, boxes, labels)

    assert tensor_boxes.shape == (0, 4)
    assert tensor_labels.shape == (0,)
