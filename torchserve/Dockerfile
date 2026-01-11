# TorchServe Dockerfile for Bird Detection Model
# Using specific version with better HTTP handling

FROM pytorch/torchserve:0.11.1-cpu

# Set working directory
WORKDIR /home/model-server

# Install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy model artifacts and handler
COPY torchserve/model-store /home/model-server/model-store
COPY torchserve/config.properties /home/model-server/config.properties

# Copy bird_detection package (needed by handler)
COPY bird_detection /home/model-server/bird_detection

# Install Python dependencies
COPY torchserve/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set PYTHONPATH so handler can import bird_detection
ENV PYTHONPATH="/home/model-server"

# Set proper permissions
RUN chown -R model-server:model-server /home/model-server

# Switch back to model-server user
USER model-server

# Expose ports
# 8080: Inference API
# 8081: Management API
# 8082: Metrics API
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Set TorchServe environment variables
ENV TS_NUMBER_OF_GPU=0
ENV TS_NUMBER_OF_NETTY_THREADS=4
ENV TS_NETTY_CLIENT_THREADS=4
ENV TS_DECODE_INPUT_REQUEST=true

# Critical: Set max request size at TorchServe level (in bytes)
# This is the most reliable way to override the Netty limit
ENV TS_MAX_REQUEST_SIZE=104857600
ENV TS_MAX_RESPONSE_SIZE=104857600

# Set JVM options for additional safety
ENV JAVA_TOOL_OPTIONS="-Xmx2048m -Xms1024m"

# Start TorchServe with explicit foreground mode
CMD ["torchserve", \
     "--start", \
     "--model-store", "/home/model-server/model-store", \
     "--ts-config", "/home/model-server/config.properties", \
     "--models", "bird_detection=bird_detection.mar", \
     "--foreground"]
