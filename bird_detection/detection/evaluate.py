"""Evaluation script for trained detection models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from bird_detection.detection.augmentations import build_transform_pipeline
from bird_detection.detection.dataloaders import build_dataloaders
from bird_detection.detection.inference import load_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def evaluate_detection(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    num_classes: int = 52,
) -> tuple[Dict[str, Any], List[Dict], List[Dict], Dict[str, Dict]]:
    """Evaluate object detection model.
    
    Returns:
        Tuple of (metrics, predictions, targets, pr_curves_data) for further analysis.
    """
    model.eval()
    
    # Storage for predictions and ground truth
    all_predictions: List[Dict[str, Any]] = []
    all_ground_truths: List[Dict[str, Any]] = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                # Store predictions
                scores = output["scores"].cpu().numpy()
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                # Filter by score threshold
                keep = scores >= score_threshold
                all_predictions.append({
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                })
                
                # Store ground truth
                all_ground_truths.append({
                    "boxes": target["boxes"].cpu().numpy(),
                    "labels": target["labels"].cpu().numpy(),
                })
    
    # Compute mAP for each class
    aps = []
    class_metrics = {}
    pr_curves_data = {}  # Store PR curves for plotting
    
    for class_id in range(1, num_classes):  # Skip background (0)
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_ground_truths = []
        
        for pred, gt in zip(all_predictions, all_ground_truths):
            # Predictions for this class
            class_mask = pred["labels"] == class_id
            if class_mask.any():
                class_predictions.append({
                    "boxes": pred["boxes"][class_mask],
                    "scores": pred["scores"][class_mask],
                })
            else:
                class_predictions.append({"boxes": np.array([]), "scores": np.array([])})
            
            # Ground truths for this class
            gt_mask = gt["labels"] == class_id
            class_ground_truths.append(gt["boxes"][gt_mask])
        
        # Count total ground truth instances
        num_gt = sum(len(gt) for gt in class_ground_truths)
        
        if num_gt == 0:
            continue
        
        # Flatten predictions across all images and sort by score
        all_scores = []
        all_boxes = []
        all_image_ids = []
        
        for img_id, pred in enumerate(class_predictions):
            if len(pred["scores"]) > 0:
                all_scores.extend(pred["scores"])
                all_boxes.extend(pred["boxes"])
                all_image_ids.extend([img_id] * len(pred["scores"]))
        
        if len(all_scores) == 0:
            aps.append(0.0)
            continue
        
        # Sort by confidence
        sorted_indices = np.argsort(all_scores)[::-1]
        all_scores = np.array(all_scores)[sorted_indices]
        all_boxes = np.array(all_boxes)[sorted_indices]
        all_image_ids = np.array(all_image_ids)[sorted_indices]
        
        # Compute precision-recall
        tp = np.zeros(len(all_scores))
        fp = np.zeros(len(all_scores))
        matched_gt = [set() for _ in range(len(class_ground_truths))]
        
        for i, (box, img_id) in enumerate(zip(all_boxes, all_image_ids)):
            gt_boxes = class_ground_truths[img_id]
            
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            # Compute IoU with all ground truth boxes
            box_tensor = torch.tensor(box).unsqueeze(0)
            gt_tensor = torch.tensor(gt_boxes)
            ious = box_iou(box_tensor, gt_tensor)[0]
            
            max_iou_idx = ious.argmax().item()
            max_iou = ious[max_iou_idx].item()
            
            if max_iou >= iou_threshold and max_iou_idx not in matched_gt[img_id]:
                tp[i] = 1
                matched_gt[img_id].add(max_iou_idx)
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
        class_metrics[f"class_{class_id}"] = {
            "ap": float(ap),
            "num_gt": int(num_gt),
            "num_predictions": len(all_scores),
        }
        
        # Store PR curve data for plotting
        pr_curves_data[f"class_{class_id}"] = {
            "precision": precisions.tolist() if len(precisions) > 0 else [],
            "recall": recalls.tolist() if len(recalls) > 0 else [],
            "ap": float(ap),
        }
    
    # Compute mean AP
    mean_ap = np.mean(aps) if aps else 0.0
    
    metrics = {
        "mAP@0.5": float(mean_ap),
        "num_classes_evaluated": len(aps),
        "class_metrics": class_metrics,
    }
    
    return metrics, all_predictions, all_ground_truths, pr_curves_data


def plot_confusion_matrix(predictions: List, targets: List, num_classes: int, output_path: Path):
    """Generate confusion matrix plot."""
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred, target in zip(predictions, targets):
        # Simple version: take highest scoring prediction per image
        if len(pred["scores"]) > 0:
            pred_class = pred["labels"][0]  # Highest scoring
            if len(target["labels"]) > 0:
                true_class = target["labels"][0]  # First GT
                confusion[true_class, pred_class] += 1
    
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained detection model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for evaluation results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for mAP computation")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold for predictions")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_dir}")
    model, metadata, config = load_detector(args.model_dir, device=device)
    
    # Build evaluation transform
    eval_transform = build_transform_pipeline(
        items=None,  # No augmentations for evaluation
        mean=config["data"]["image_mean"],
        std=config["data"]["image_std"],
    )
    
    # Load test dataset using build_dataloaders
    data_cfg = {
        "birds_dir": str(data_dir / "birds"),
        "birds_mapping_csv": str(data_dir / "birds" / "class_mapping.csv"),
        "include_squirrels": True,
        "squirrels_dir": str(data_dir / "squirrels") if (data_dir / "squirrels").exists() else None,
        "squirrel_label_id": 84,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
    }
    
    logger.info("Loading test dataset")
    _, _, test_loader, test_metadata = build_dataloaders(
        data_cfg=data_cfg,
        train_transform=eval_transform,  # Not used for test
        eval_transform=eval_transform,
        seed=42,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
    )
    
    logger.info(f"Test set size: {len(test_loader.dataset)}")
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics, predictions, targets, pr_curves_data = evaluate_detection(
        model=model,
        dataloader=test_loader,
        device=device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        num_classes=test_metadata.num_classes,
    )
    
    # Add additional metrics
    metrics["iou_threshold"] = args.iou_threshold
    metrics["score_threshold"] = args.score_threshold
    metrics["test_set_size"] = len(test_loader.dataset)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    
    # Save PR curves data (actual precision/recall arrays for plotting)
    pr_curves_path = output_dir / "pr_curves.json"
    with open(pr_curves_path, "w") as f:
        json.dump(pr_curves_data, f, indent=2)
    
    logger.info(f"PR curves data saved to {pr_curves_path}")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix...")
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(predictions, targets, test_metadata.num_classes, cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
