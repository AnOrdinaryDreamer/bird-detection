"""
Script to export trained model to TorchServe format.

This script loads the trained model and exports it as a state_dict
that can be packaged with torch-model-archiver.
"""

import json
import logging
import shutil
from pathlib import Path

import torch

from bird_detection.detection.inference import load_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_model_for_torchserve(
    model_dir: Path,
    output_dir: Path,
    device: str = "cpu",
) -> None:
    """
    Export model for TorchServe deployment.
    
    Args:
        model_dir: Directory with trained model (pytorch_model.bin, config.json)
        output_dir: Directory to save TorchServe artifacts
        device: Device to load model on
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {model_dir}")
    model, metadata, config = load_detector(model_dir, device=device)
    
    # Save model state_dict
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model state_dict to {model_path}")
    
    # Copy config and label map
    config_src = model_dir / "config.json"
    config_dst = output_dir / "config.json"
    shutil.copy(config_src, config_dst)
    logger.info(f"Copied config to {config_dst}")
    
    # Save label map separately for easy access
    label_map = {
        "id2label": metadata.id2label,
        "label2id": metadata.label2id,
    }
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved label map to {label_map_path}")
    
    # Create index_to_name.json for TorchServe
    index_to_name = {str(k): v for k, v in metadata.id2label.items()}
    index_path = output_dir / "index_to_name.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_to_name, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved index_to_name.json to {index_path}")
    
    logger.info("Export completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export model for TorchServe")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/trained_model",
        help="Directory with trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="torchserve/model_artifacts",
        help="Output directory for TorchServe artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on",
    )
    
    args = parser.parse_args()
    
    export_model_for_torchserve(
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
    )
