#!/bin/bash

# InsightFace-REST Jetson AGX Entrypoint Script
# Optimized for JetPack 6.2 with CUDA 12.6 and TensorRT 10.3

set -e

echo "===========================================" 
echo "InsightFace-REST on Jetson AGX JetPack 6.2"
echo "==========================================="
echo "CUDA Version: $(nvcc --version | grep release)"
echo "GPU Info:"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"
echo "Available CUDA devices: ${NVIDIA_VISIBLE_DEVICES:-all}"
echo "==========================================="

# Set environment variables with Jetson optimizations
export PYTHONPATH=/app:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Jetson memory and performance optimizations
export CUDA_CACHE_DISABLE=0
export TENSORRT_LOG_LEVEL=3
export TRT_ENGINE_CACHE_ENABLE=1

# Enable memory pool and optimizations for limited memory
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=131072

# Set default values for environment variables
NUM_WORKERS=${NUM_WORKERS:-1}
PORT=${PORT:-18080}
LOG_LEVEL=${LOG_LEVEL:-INFO}
INFERENCE_BACKEND=${INFERENCE_BACKEND:-trt}

# Model configuration optimized for Jetson
DET_NAME=${DET_NAME:-scrfd_10g_gnkps}
REC_NAME=${REC_NAME:-glintr100}
MASK_DETECTOR=${MASK_DETECTOR:-None}
GA_NAME=${GA_NAME:-None}

# Performance settings for Jetson
FORCE_FP16=${FORCE_FP16:-True}
MAX_SIZE=${MAX_SIZE:-640,640}
DET_BATCH_SIZE=${DET_BATCH_SIZE:-1}
REC_BATCH_SIZE=${REC_BATCH_SIZE:-2}

# Jetson-specific settings
USE_NVJPEG=${USE_NVJPEG:-True}
DET_THRESH=${DET_THRESH:-0.6}

echo "Configuration:"
echo "  Workers: $NUM_WORKERS"
echo "  Port: $PORT"
echo "  Backend: $INFERENCE_BACKEND"
echo "  Detection Model: $DET_NAME"
echo "  Recognition Model: $REC_NAME"
echo "  FP16 Mode: $FORCE_FP16"
echo "  Max Size: $MAX_SIZE"
echo "  Detection Batch Size: $DET_BATCH_SIZE"
echo "  Recognition Batch Size: $REC_BATCH_SIZE"
echo "==========================================="

# Create models directory if it doesn't exist
mkdir -p /models

# Check if GPU is available
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not available for GPU check"

# Change to app directory
cd /app

# Prepare models (download and setup)
echo "Preparing models..."
python3 -m api_trt.prepare_models

# Start the application with Jetson-optimized settings
echo "Starting InsightFace-REST server using $NUM_WORKERS workers..."
exec gunicorn --log-level=${LOG_LEVEL:-INFO} \
    -w $NUM_WORKERS \
    -k uvicorn.workers.UvicornWorker \
    --keep-alive 60 \
    --timeout 60 \
    api_trt.app:app -b 0.0.0.0:$PORT 