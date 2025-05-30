#!/bin/bash

# GPU-optimized deployment script for InsightFace-REST on Jetson AGX with JetPack 6.2
# With TensorRT backend for maximum performance

set -e

IMAGE='insightface-rest-jetson-gpu'
TAG='v1.0.0-trt'

echo "=============================================="
echo "InsightFace-REST Jetson AGX GPU Deployment"
echo "=============================================="
echo "Image: $IMAGE:$TAG"
echo "Target: JetPack 6.2 (L4T r36.4.3)"  
echo "Architecture: ARM64 (aarch64)"
echo "Backend: TensorRT (GPU optimized)"
echo "=============================================="

# GPU-optimized configuration
log_level=INFO
n_workers=1
det_model=scrfd_10g_gnkps
rec_model=glintr100
max_size=640,640
force_fp16=True
det_batch_size=1
rec_batch_size=4  # Increased for GPU
mask_detector=None
ga_model=None
use_nvjpeg=True
cuda_cache_disable=0

echo "GPU Configuration:"
echo "  Workers: $n_workers"
echo "  Detection Model: $det_model"
echo "  Recognition Model: $rec_model"
echo "  Max Image Size: $max_size"
echo "  FP16 Mode: $force_fp16"
echo "  Backend: TensorRT"
echo "  GPU Batch Sizes: det=$det_batch_size, rec=$rec_batch_size"
echo "=============================================="

# Check GPU availability
echo "Checking GPU status..."
nvidia-smi -L || echo "WARNING: nvidia-smi not available"

# Build the GPU-optimized Docker image
echo "Building GPU-optimized Docker image..."
docker build -t $IMAGE:$TAG -f src/Dockerfile_jetson_gpu src/

# Stop any existing containers
echo "Stopping any existing containers..."
docker rm -f insightface-gpu-jetson 2>/dev/null || true

# Start GPU-optimized container with TensorRT backend
echo "Starting GPU-optimized container..."
docker run \
    --gpus all \
    -d \
    --restart=unless-stopped \
    -e LOG_LEVEL=$log_level \
    -e USE_NVJPEG=$use_nvjpeg \
    -e PYTHONUNBUFFERED=1 \
    -e PORT=18080 \
    -e NUM_WORKERS=$n_workers \
    -e INFERENCE_BACKEND=trt \
    -e FORCE_FP16=$force_fp16 \
    -e DET_NAME=$det_model \
    -e REC_NAME=$rec_model \
    -e MASK_DETECTOR=$mask_detector \
    -e REC_BATCH_SIZE=$rec_batch_size \
    -e DET_BATCH_SIZE=$det_batch_size \
    -e GA_NAME=$ga_model \
    -e KEEP_ALL=True \
    -e MAX_SIZE=$max_size \
    -e CUDA_CACHE_DISABLE=$cuda_cache_disable \
    -e TENSORRT_LOG_LEVEL=3 \
    -e TRT_ENGINE_CACHE_ENABLE=1 \
    -v $PWD/models:/models \
    -v $PWD/src:/app \
    --name=insightface-gpu-jetson \
    $IMAGE:$TAG

echo "=============================================="
echo "GPU deployment completed!"
echo ""
echo "Container name: insightface-gpu-jetson"
echo "Backend: TensorRT (GPU optimized)"
echo "Check status with: docker ps"
echo "View logs with: docker logs insightface-gpu-jetson"
echo "Access container with: docker exec -it insightface-gpu-jetson bash"
echo ""
echo "To test API (from inside container):"
echo "  curl http://localhost:18080/info"
echo ""
echo "Expected improvements:"
echo "  - 3-10x faster inference with TensorRT"
echo "  - FP16 optimization enabled"
echo "  - GPU memory management optimized"
echo "==============================================" 