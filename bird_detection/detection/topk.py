"""Helpers to expose top-k class predictions from torchvision detectors."""

from __future__ import annotations

import types
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops


def enable_topk_predictions(model: Module, topk: int) -> Module:
    """
    Enable emission of top-k class alternatives for supported detection models.

    Args:
        model: Torchvision detection model (GeneralizedRCNN or SSD).
        topk: Number of alternative classes to keep per detection (>=2 enables it).

    Returns:
        The same model instance for convenience.
    """

    if topk is None or topk <= 1:
        return model

    if isinstance(model, GeneralizedRCNN):
        _enable_topk_generalized_rcnn(model, topk)
    elif isinstance(model, SSD):
        _enable_topk_ssd(model, topk)
    else:
        raise TypeError(
            f"Top-k predictions are not supported for model type '{type(model).__name__}'"
        )
    return model


def _enable_topk_generalized_rcnn(model: GeneralizedRCNN, topk: int) -> None:
    if getattr(model, "_topk_enabled", False):
        model._topk_predictions = topk
        return

    model._topk_enabled = True
    model._topk_predictions = topk
    model._topk_cache: Dict[str, Any] = {}
    hooks = []

    def _store_roi_inputs(_module, inputs):
        if len(inputs) < 3:
            return
        proposals = inputs[1]
        image_shapes = inputs[2]
        model._topk_cache["proposals"] = [p.detach() for p in proposals]
        model._topk_cache["image_shapes"] = list(image_shapes)

    hooks.append(model.roi_heads.register_forward_pre_hook(_store_roi_inputs))

    def _store_box_outputs(_module, _inputs, output):
        class_logits, box_regression = output
        model._topk_cache["class_logits"] = class_logits.detach()
        model._topk_cache["box_regression"] = box_regression.detach()

    hooks.append(
        model.roi_heads.box_predictor.register_forward_hook(_store_box_outputs)
    )
    model._topk_handles = hooks

    original_forward = model.forward

    def forward_with_topk(self, images, targets=None):  # type: ignore[override]
        outputs = original_forward(images, targets)
        cache = getattr(self, "_topk_cache", None)
        if (
            self.training
            or getattr(self, "_topk_predictions", 0) <= 1
            or not isinstance(outputs, list)
            or cache is None
        ):
            if cache is not None:
                cache.clear()
            return outputs

        topk_labels, topk_scores = _roi_postprocess_with_topk(
            self.roi_heads, cache, self._topk_predictions
        )
        cache.clear()
        if topk_labels is None or topk_scores is None:
            return outputs
        if len(outputs) != len(topk_labels):
            return outputs

        for det, labels, scores in zip(outputs, topk_labels, topk_scores):
            if not isinstance(det, dict):
                continue
            scores_tensor = det.get("scores")
            if scores_tensor is None:
                continue
            if labels.shape[0] != scores_tensor.shape[0]:
                continue
            det["topk_labels"] = labels
            det["topk_scores"] = scores
        return outputs

    model.forward = types.MethodType(forward_with_topk, model)


def _roi_postprocess_with_topk(
    roi_heads: RoIHeads, cache: Dict[str, Any], topk: int
) -> Tuple[List[Tensor], List[Tensor]] | Tuple[None, None]:
    required = ("class_logits", "box_regression", "proposals", "image_shapes")
    if any(key not in cache for key in required):
        return None, None
    if topk <= 1:
        return None, None

    class_logits: Tensor = cache["class_logits"]
    box_regression: Tensor = cache["box_regression"]
    proposals: Sequence[Tensor] = cache["proposals"]
    image_shapes: Sequence[Tuple[int, int]] = cache["image_shapes"]

    boxes_per_image = [p.shape[0] for p in proposals]
    if not boxes_per_image:
        return None, None

    with torch.no_grad():
        pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)
    num_classes = class_logits.shape[-1]

    topk_labels: List[Tensor] = []
    topk_scores: List[Tensor] = []

    for boxes, scores, image_shape in zip(
        pred_boxes_list, pred_scores_list, image_shapes
    ):
        device = boxes.device
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        boxes = boxes[:, 1:]
        scores_fg = scores[:, 1:]
        labels = labels[:, 1:]

        num_props, num_fg_classes = scores_fg.shape
        if num_props == 0 or num_fg_classes == 0:
            zero = min(topk, max(num_fg_classes, 1))
            topk_labels.append(torch.zeros((0, zero), dtype=torch.int64, device=device))
            topk_scores.append(
                torch.zeros((0, zero), dtype=scores.dtype, device=device)
            )
            continue

        flat_boxes = boxes.reshape(-1, 4)
        flat_scores = scores_fg.reshape(-1)
        flat_labels = labels.reshape(-1)
        proposal_indices = (
            torch.arange(flat_scores.numel(), device=device) // num_fg_classes
        )

        keep = torch.where(flat_scores > roi_heads.score_thresh)[0]
        flat_boxes = flat_boxes[keep]
        flat_scores = flat_scores[keep]
        flat_labels = flat_labels[keep]
        proposal_indices = proposal_indices[keep]

        keep = box_ops.remove_small_boxes(flat_boxes, min_size=1e-2)
        flat_boxes = flat_boxes[keep]
        flat_scores = flat_scores[keep]
        flat_labels = flat_labels[keep]
        proposal_indices = proposal_indices[keep]

        keep = box_ops.batched_nms(
            flat_boxes, flat_scores, flat_labels, roi_heads.nms_thresh
        )
        keep = keep[: roi_heads.detections_per_img]
        proposal_indices = proposal_indices[keep]

        det_count = proposal_indices.numel()
        if det_count == 0:
            zero = min(topk, num_fg_classes)
            topk_labels.append(torch.zeros((0, zero), dtype=torch.int64, device=device))
            topk_scores.append(
                torch.zeros((0, zero), dtype=scores.dtype, device=device)
            )
            continue

        k = min(topk, num_fg_classes)
        if k == 0:
            topk_labels.append(
                torch.zeros((det_count, 0), dtype=torch.int64, device=device)
            )
            topk_scores.append(
                torch.zeros((det_count, 0), dtype=scores.dtype, device=device)
            )
            continue

        per_det_scores = scores_fg[proposal_indices]
        det_topk_scores, det_topk_idx = torch.topk(per_det_scores, k=k, dim=1)
        det_topk_labels = det_topk_idx + 1
        topk_labels.append(det_topk_labels)
        topk_scores.append(det_topk_scores)

    return topk_labels, topk_scores


def _enable_topk_ssd(model: SSD, topk: int) -> None:
    if getattr(model, "_topk_enabled", False):
        model._topk_predictions = topk
        return

    model._topk_enabled = True
    model._topk_predictions = topk

    def postprocess_with_topk(head_outputs, image_anchors, image_shapes):
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.size(-1)
        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, image_anchors, image_shapes
        ):
            boxes = model.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            anchor_indices = []
            for label in range(1, num_classes):
                score = scores[:, label]
                keep_idxs = score > model.score_thresh
                if not torch.any(keep_idxs):
                    continue
                filtered_scores = score[keep_idxs]
                filtered_boxes = boxes[keep_idxs]
                candidate_indices = torch.where(keep_idxs)[0]

                num_topk = det_utils._topk_min(
                    filtered_scores, model.topk_candidates, 0
                )
                filtered_scores, idxs = filtered_scores.topk(num_topk)
                filtered_boxes = filtered_boxes[idxs]
                candidate_indices = candidate_indices[idxs]

                image_boxes.append(filtered_boxes)
                image_scores.append(filtered_scores)
                image_labels.append(
                    torch.full_like(
                        filtered_scores,
                        fill_value=label,
                        dtype=torch.int64,
                        device=filtered_scores.device,
                    )
                )
                anchor_indices.append(candidate_indices)

            if image_boxes:
                image_boxes_t = torch.cat(image_boxes, dim=0)
                image_scores_t = torch.cat(image_scores, dim=0)
                image_labels_t = torch.cat(image_labels, dim=0)
                anchor_indices_t = torch.cat(anchor_indices, dim=0)
            else:
                device = boxes.device
                image_boxes_t = boxes.new_zeros((0, 4))
                image_scores_t = boxes.new_zeros((0,))
                image_labels_t = torch.zeros((0,), dtype=torch.int64, device=device)
                anchor_indices_t = torch.zeros((0,), dtype=torch.int64, device=device)

            keep = box_ops.batched_nms(
                image_boxes_t, image_scores_t, image_labels_t, model.nms_thresh
            )
            keep = keep[: model.detections_per_img]

            image_boxes_t = image_boxes_t[keep]
            image_scores_t = image_scores_t[keep]
            image_labels_t = image_labels_t[keep]
            anchor_indices_t = anchor_indices_t[keep]

            det: Dict[str, Tensor] = {
                "boxes": image_boxes_t,
                "scores": image_scores_t,
                "labels": image_labels_t,
            }

            if (
                getattr(model, "_topk_predictions", 0) > 1
                and anchor_indices_t.numel() > 0
            ):
                k = min(model._topk_predictions, num_classes - 1)
                per_det_scores = scores[anchor_indices_t]
                topk_scores, topk_idx = torch.topk(per_det_scores[:, 1:], k=k, dim=1)
                det["topk_scores"] = topk_scores
                det["topk_labels"] = topk_idx + 1
            else:
                det["topk_scores"] = image_scores_t.new_zeros(
                    (anchor_indices_t.numel(), 0)
                )
                det["topk_labels"] = anchor_indices_t.new_zeros(
                    (anchor_indices_t.numel(), 0), dtype=torch.int64
                )

            detections.append(det)

        return detections

    model.postprocess_detections = postprocess_with_topk  # type: ignore[assignment]
