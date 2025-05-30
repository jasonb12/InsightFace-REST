#!/bin/bash

# Test deployment script for InsightFace-REST on Jetson AGX with JetPack 6.2
# WITHOUT port mapping to avoid iptables issues

set -e

IMAGE='insightface-rest-jetson'
TAG='v1.0.0-jetpack6.2'

echo "=============================================="
echo "InsightFace-REST Jetson AGX TEST Deployment"
echo "=============================================="
echo "Image: $IMAGE:$TAG"
echo "Target: JetPack 6.2 (L4T r36.4.3)"  
echo "Architecture: ARM64 (aarch64)"
echo "=============================================="

# Configuration
log_level=INFO
n_workers=1
det_model=scrfd_10g_gnkps
rec_model=glintr100
max_size=640,640
force_fp16=True
det_batch_size=1
rec_batch_size=2
mask_detector=None
ga_model=None
use_nvjpeg=True
cuda_cache_disable=0

echo "Configuration:"
echo "  Workers: $n_workers"
echo "  Detection Model: $det_model"
echo "  Recognition Model: $rec_model"
echo "  Max Image Size: $max_size"
echo "  FP16 Mode: $force_fp16"
echo "=============================================="

# Stop any existing test container
echo "Stopping any existing test container..."
docker rm -f insightface-test-jetson 2>/dev/null || true

# Start container WITHOUT port mapping to avoid iptables issues
echo "Starting test container..."
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
    -v $PWD/models:/models \
    -v $PWD/src:/app \
    --name=insightface-test-jetson \
    $IMAGE:$TAG

echo "=============================================="
echo "Test deployment completed!"
echo ""
echo "Container name: insightface-test-jetson"
echo "Check status with: docker ps"
echo "View logs with: docker logs insightface-test-jetson"
echo "Access container with: docker exec -it insightface-test-jetson bash"
echo ""
echo "To test API (from inside container):"
echo "  curl http://localhost:18080/info"
echo "==============================================" 