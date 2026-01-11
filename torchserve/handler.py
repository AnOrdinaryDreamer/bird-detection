"""
Custom TorchServe handler for bird detection model.

This handler implements preprocessing and postprocessing for the object detection model.
"""

import base64
import io
import json
import logging
from typing import Any, Dict, List

import torch
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class BirdDetectionHandler(BaseHandler):
    """
    Custom handler for bird detection model.
    
    Handles:
    - Image preprocessing (base64, file upload, URL)
    - Model inference
    - Postprocessing with configurable thresholds
    - Top-K alternative predictions
    """
    
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.device = None
        self.id2label = {}
        self.label2id = {}
        self.config = {}
        self.transform = None
        
    def initialize(self, context):
        """
        Initialize model and load metadata.
        
        Args:
            context: TorchServe context with model artifacts
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Set device
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        # Load config
        config_path = f"{model_dir}/config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # Load label mappings
        id2label_raw = self.config.get("id2label", {})
        self.id2label = {int(k): v for k, v in id2label_raw.items()}
        
        label2id_raw = self.config.get("label2id", {})
        self.label2id = {str(k): int(v) for k, v in label2id_raw.items()}
        
        logger.info(f"Loaded {len(self.id2label)} classes")
        
        # Build model
        from bird_detection.detection.models import build_model
        from bird_detection.detection.topk import enable_topk_predictions
        
        model_cfg = self.config.get("model", {})
        num_classes = len(self.id2label)
        
        self.model = build_model(model_cfg, num_classes)
        
        # Enable top-k predictions
        topk = model_cfg.get("prediction_topk", 3)
        enable_topk_predictions(self.model, topk)
        
        # Load weights
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = f"{model_dir}/{serialized_file}"
        
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Setup transform
        self.transform = transforms.ToTensor()
        
        self.initialized = True
        
    def preprocess(self, data: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Preprocess input data.
        
        Supports:
        - Binary image data (with service_envelope=body)
        - Base64 encoded images
        - Custom threshold via X-Threshold HTTP header (default 0.2)
        
        Args:
            data: List of request data (binary or dict)
            
        Returns:
            List of preprocessed image tensors
        """
        images = []
        
        # Note: threshold is set in handle() from HTTP header
        # Default is 0.2 if not specified
        
        for row in data:
            # Handle TorchServe envelope format
            # row can be: bytes, dict with "data"/"body", or dict with "image"
            image_data = None
            
            if isinstance(row, (bytearray, bytes)):
                # Direct binary data
                image_data = row
            elif isinstance(row, dict):
                # Try different keys in order of preference
                image_data = row.get("body") or row.get("data") or row.get("image")
                
                # If still dict, might have nested structure
                if isinstance(image_data, dict):
                    image_data = image_data.get("image") or image_data.get("data")
            else:
                # Might be string (base64)
                image_data = row
            
            # Now process image_data
            if image_data is None:
                raise ValueError(f"Could not extract image data from request. Keys: {row.keys() if isinstance(row, dict) else 'not a dict'}")
            
            if isinstance(image_data, (bytearray, bytes)):
                # Binary image data
                try:
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                except Exception as e:
                    logger.error(f"Failed to open image from bytes: {e}")
                    raise ValueError(f"Invalid image data: {e}")
            elif isinstance(image_data, str):
                # Base64 encoded string
                try:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    raise ValueError(f"Invalid base64 image: {e}")
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
            # Convert to tensor [C, H, W] in range [0, 1]
            tensor = self.transform(image)
            images.append(tensor)
        
        return images
    
    def inference(self, data: List[torch.Tensor], *args, **kwargs) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on preprocessed data.
        
        Args:
            data: List of image tensors
            
        Returns:
            List of model outputs (boxes, labels, scores, topk_labels, topk_scores)
        """
        # Move tensors to device
        data = [img.to(self.device) for img in data]
        
        with torch.no_grad():
            predictions = self.model(data)
        
        return predictions
    
    def postprocess(self, inference_output: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to API-friendly format.
        
        Args:
            inference_output: Raw model predictions
            
        Returns:
            List of formatted predictions
        """
        results = []
        
        # Get threshold from context (if provided in request) or use default
        score_threshold = getattr(self, '_threshold', 0.2)
        
        for output in inference_output:
            detections = []
            
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            
            # Get top-k predictions if available
            topk_labels = output.get("topk_labels")
            topk_scores = output.get("topk_scores")
            
            if topk_labels is not None:
                topk_labels = topk_labels.cpu().numpy()
                topk_scores = topk_scores.cpu().numpy()
            
            for i in range(len(boxes)):
                score = float(scores[i])
                
                # Filter by threshold
                if score < score_threshold:
                    continue
                
                label_id = int(labels[i])
                label_name = self.id2label.get(label_id, f"class_{label_id}")
                
                # Skip background
                if label_name == "__background__":
                    continue
                
                box = boxes[i]
                detection = {
                    "label": label_name,
                    "label_id": label_id,
                    "score": round(score, 4),
                    "bbox": {
                        "x_min": round(float(box[0]), 2),
                        "y_min": round(float(box[1]), 2),
                        "x_max": round(float(box[2]), 2),
                        "y_max": round(float(box[3]), 2),
                    },
                }
                
                # Add top-k alternatives
                if topk_labels is not None and i < len(topk_labels):
                    alternatives = []
                    for k in range(len(topk_labels[i])):
                        alt_label_id = int(topk_labels[i][k])
                        alt_label_name = self.id2label.get(alt_label_id, f"class_{alt_label_id}")
                        alt_score = float(topk_scores[i][k])
                        
                        # Skip if same as main prediction or background
                        if alt_label_id == label_id or alt_label_name == "__background__":
                            continue
                        
                        alternatives.append({
                            "label": alt_label_name,
                            "label_id": alt_label_id,
                            "score": round(alt_score, 4),
                        })
                    
                    if alternatives:
                        detection["alternatives"] = alternatives
                
                detections.append(detection)
            
            results.append({
                "detections": detections,
                "num_detections": len(detections),
                "threshold_used": score_threshold,
            })
        
        return results


# TorchServe entry point
_service = BirdDetectionHandler()


def handle(data, context):
    """
    Entry point for TorchServe.
    
    Supports custom threshold via HTTP header:
    - X-Threshold: float (default 0.2) - confidence threshold for detections
    
    Examples:
        # Default threshold (0.2)
        curl -X POST http://localhost:8080/predictions/bird_detection -T image.jpg
        
        # Custom threshold via header
        curl -X POST http://localhost:8080/predictions/bird_detection \
            -H "X-Threshold: 0.5" -T image.jpg
    
    Args:
        data: Input data from request
        context: TorchServe context
        
    Returns:
        Prediction results
    """
    if not _service.initialized:
        _service.initialize(context)
    
    if data is None:
        return None
    
    # Extract threshold from HTTP header
    _service._threshold = 0.2  # default
    try:
        # Get request headers from context
        if hasattr(context, 'get_request_header'):
            threshold_header = context.get_request_header(0, 'X-Threshold')
            if threshold_header:
                _service._threshold = float(threshold_header)
                logger.info(f"Using custom threshold from header: {_service._threshold}")
            else:
                logger.debug(f"No X-Threshold header, using default: {_service._threshold}")
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Error reading threshold header: {e}, using default: {_service._threshold}")
    
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    
    return data
