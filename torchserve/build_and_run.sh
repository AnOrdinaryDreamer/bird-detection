#!/bin/bash
# Script to build and run TorchServe Docker container

set -e

echo "=== Bird Detection TorchServe Deployment ==="
echo ""

# Configuration
IMAGE_NAME="bird-detection-serve"
IMAGE_TAG="v1"
CONTAINER_NAME="bird-detection-torchserve"

# Step 1: Export model
echo "Step 1: Exporting model for TorchServe..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHONPATH="$PROJECT_DIR" poetry run python torchserve/export_model.py \
    --model_dir outputs/trained_model \
    --output_dir torchserve/model_artifacts

echo "✓ Model exported"
echo ""

# Step 2: Create model archive
echo "Step 2: Creating model archive (.mar file)..."
cd torchserve
bash create_mar.sh

echo "✓ Model archive created"
echo ""

# Step 3: Build Docker image
echo "Step 3: Building Docker image..."
cd ..
docker build -f torchserve/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "✓ Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Step 4: Stop existing container if running
echo "Step 4: Checking for existing container..."
if docker ps -a | grep -q ${CONTAINER_NAME}; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
fi

echo "✓ Ready to start"
echo ""

# Step 5: Run container
echo "Step 5: Starting TorchServe container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    ${IMAGE_NAME}:${IMAGE_TAG}

echo "✓ Container started: ${CONTAINER_NAME}"
echo ""

# Wait for service to be ready
echo "Waiting for TorchServe to be ready..."
sleep 10

# Check health
echo "Checking service health..."
for i in {1..30}; do
    if curl -s http://localhost:8080/ping > /dev/null 2>&1; then
        echo "✓ TorchServe is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ TorchServe failed to start. Check logs with: docker logs ${CONTAINER_NAME}"
        exit 1
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service endpoints:"
echo "  Inference API: http://localhost:8080"
echo "  Management API: http://localhost:8081"
echo "  Metrics API: http://localhost:8082"
echo ""
echo "Test the service:"
echo "  curl http://localhost:8080/ping"
echo "  curl http://localhost:8081/models"
echo ""
echo "To view logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "To stop the service:"
echo "  docker stop ${CONTAINER_NAME}"
