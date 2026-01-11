"""
Test script for TorchServe deployment.

Tests the deployed model with sample images.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_ping(base_url: str) -> bool:
    """Test if service is alive."""
    try:
        response = requests.get(f"{base_url}/ping", timeout=5)
        if response.status_code == 200:
            print("✓ Service is alive")
            return True
        else:
            print(f"✗ Ping failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ping failed: {e}")
        return False


def test_models_list(management_url: str) -> bool:
    """List registered models."""
    try:
        response = requests.get(f"{management_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✓ Registered models: {json.dumps(models, indent=2)}")
            return True
        else:
            print(f"✗ Failed to list models: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
        return False


def test_model_info(management_url: str, model_name: str) -> bool:
    """Get model information."""
    try:
        response = requests.get(f"{management_url}/models/{model_name}", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✓ Model info: {json.dumps(info, indent=2)}")
            return True
        else:
            print(f"✗ Failed to get model info: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to get model info: {e}")
        return False


def test_prediction_binary(inference_url: str, model_name: str, image_path: Path) -> bool:
    """Test prediction with binary image data."""
    try:
        print(f"\nTesting prediction with binary image: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        response = requests.post(
            f"{inference_url}/predictions/{model_name}",
            data=image_data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction successful:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_prediction_json(inference_url: str, model_name: str, image_path: Path) -> bool:
    """Test prediction with JSON (base64 encoded image)."""
    try:
        print(f"\nTesting prediction with JSON: {image_path}")
        
        image_base64 = encode_image_to_base64(image_path)
        
        payload = {
            "image": image_base64
        }
        
        response = requests.post(
            f"{inference_url}/predictions/{model_name}",
            json=payload,
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction successful:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test TorchServe deployment")
    parser.add_argument(
        "--inference_url",
        type=str,
        default="http://localhost:8080",
        help="Inference API URL",
    )
    parser.add_argument(
        "--management_url",
        type=str,
        default="http://localhost:8081",
        help="Management API URL",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bird_detection",
        help="Model name",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image",
    )
    
    args = parser.parse_args()
    
    print("=== TorchServe Service Test ===\n")
    
    # Test 1: Ping
    print("Test 1: Ping service")
    if not test_ping(args.inference_url):
        print("\n✗ Service is not responding. Make sure TorchServe is running.")
        return 1
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: List models
    print("Test 2: List registered models")
    test_models_list(args.management_url)
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Model info
    print("Test 3: Get model information")
    test_model_info(args.management_url, args.model_name)
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Prediction
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"✗ Image not found: {image_path}")
            return 1
        
        print("Test 4: Run prediction (binary)")
        test_prediction_binary(args.inference_url, args.model_name, image_path)
        
        print("\n" + "="*50 + "\n")
        
        print("Test 5: Run prediction (JSON)")
        test_prediction_json(args.inference_url, args.model_name, image_path)
    else:
        print("Test 4 & 5: Skipped (no image provided)")
        print("Use --image to test predictions")
    
    print("\n" + "="*50 + "\n")
    print("✓ All tests completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
