#!/bin/bash
# Script to create TorchServe model archive (.mar file)

set -e

echo "Creating TorchServe model archive..."

# Configuration
MODEL_NAME="bird_detection"
MODEL_VERSION="1.0"
HANDLER="handler.py"
MODEL_FILE="model.pt"
ARTIFACTS_DIR="model_artifacts"
MODEL_STORE_DIR="model-store"

# Check if artifacts exist
if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo "Error: $ARTIFACTS_DIR directory not found"
    echo "Please run: python torchserve/export_model.py first"
    exit 1
fi

if [ ! -f "$ARTIFACTS_DIR/$MODEL_FILE" ]; then
    echo "Error: $ARTIFACTS_DIR/$MODEL_FILE not found"
    echo "Please run: python torchserve/export_model.py first"
    exit 1
fi

# Create model-store directory
mkdir -p "$MODEL_STORE_DIR"

# Create the model archive
echo "Packaging model with torch-model-archiver..."

torch-model-archiver \
    --model-name "$MODEL_NAME" \
    --version "$MODEL_VERSION" \
    --serialized-file "$ARTIFACTS_DIR/$MODEL_FILE" \
    --handler "$HANDLER" \
    --extra-files "$ARTIFACTS_DIR/config.json,$ARTIFACTS_DIR/label_map.json,$ARTIFACTS_DIR/index_to_name.json" \
    --export-path "$MODEL_STORE_DIR" \
    --force

echo "Model archive created: $MODEL_STORE_DIR/${MODEL_NAME}.mar"
