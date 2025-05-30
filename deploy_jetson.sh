#!/bin/bash

# Deployment script for InsightFace-REST on Jetson AGX with JetPack 6.2
# Optimized for ARM64 architecture with CUDA 12.6 and TensorRT 10.3

set -e

IMAGE='insightface-rest-jetson'
TAG='v1.0.0-jetpack6.2'

echo "=============================================="
echo "InsightFace-REST Jetson AGX Deployment Script"
echo "=============================================="
echo "Image: $IMAGE:$TAG"
echo "Target: JetPack 6.2 (L4T r36.4.3)"
echo "Architecture: ARM64 (aarch64)"
echo "=============================================="

# Change InsightFace-REST logging level (DEBUG,INFO,WARNING,ERROR)
log_level=INFO

# Container port configuration
START_PORT=18081

# Jetson AGX typically has 1 GPU, but support multiple containers
n_gpu=1
n_workers=1

# Jetson-optimized settings
# Maximum image size (W,H) - optimized for Jetson memory constraints
max_size=640,640

# Force FP16 mode for TensorRT engines (recommended for Jetson)
force_fp16=True

# Model configuration optimized for Jetson performance
# Detection models (choose based on performance requirements):
# - scrfd_10g_gnkps: Best accuracy, higher memory usage
# - scrfd_2.5g_gnkps: Balanced accuracy/performance  
# - scrfd_500m_gnkps: Fastest, lowest memory usage
det_model=scrfd_10g_gnkps

# Recognition models (choose based on requirements):
# - glintr100: Best accuracy
# - w600k_r50: Good balance
# - w600k_mbf: Fastest
rec_model=glintr100

# Batch sizes optimized for Jetson AGX memory
det_batch_size=1
rec_batch_size=2

# Additional models (disabled by default to save memory)
mask_detector=None
ga_model=None

# Default API settings
return_face_data=False
extract_embeddings=True
detect_ga=False
det_thresh=0.6

# Jetson-specific optimizations
use_nvjpeg=True  # Enable NVJPEG for faster JPEG decoding
cuda_cache_disable=0  # Enable CUDA caching

echo "Configuration:"
echo "  GPU Count: $n_gpu"
echo "  Workers per GPU: $n_workers"
echo "  Detection Model: $det_model"
echo "  Recognition Model: $rec_model"
echo "  Max Image Size: $max_size"
echo "  FP16 Mode: $force_fp16"
echo "  Detection Batch Size: $det_batch_size"
echo "  Recognition Batch Size: $rec_batch_size"
echo "=============================================="

# Create directory to store downloaded models
echo "Creating models directory..."
mkdir -p models

# Check if nvidia-container-toolkit is installed
if ! command -v nvidia-container-runtime &> /dev/null; then
    echo "WARNING: nvidia-container-toolkit may not be installed."
    echo "Please ensure it's installed with: sudo apt install nvidia-container-toolkit"
fi

# Check available GPU memory
echo "Checking GPU status..."
nvidia-smi -L || echo "WARNING: nvidia-smi not available"

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE:$TAG -f src/Dockerfile_jetson src/

echo "Starting $((n_gpu * n_workers)) workers on $n_gpu GPUs ($n_workers workers per GPU)"
echo "Container port range: $START_PORT - $(($START_PORT + ($n_gpu) - 1))"

p=0

for i in $(seq 0 $(($n_gpu - 1))); do
    device='"device='$i'"'
    port=$((START_PORT + $p))
    name=$IMAGE-gpu$i-jetson
    
    echo "Stopping any existing container: $name"
    docker rm -f $name 2>/dev/null || true
    
    echo "Starting container $name with $device at port $port"
    ((p++))
    
    docker run \
        -p $port:18080 \
        --gpus $device \
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
        -e DET_THRESH=$det_thresh \
        -e REC_NAME=$rec_model \
        -e MASK_DETECTOR=$mask_detector \
        -e REC_BATCH_SIZE=$rec_batch_size \
        -e DET_BATCH_SIZE=$det_batch_size \
        -e GA_NAME=$ga_model \
        -e KEEP_ALL=True \
        -e MAX_SIZE=$max_size \
        -e DEF_RETURN_FACE_DATA=$return_face_data \
        -e DEF_EXTRACT_EMBEDDING=$extract_embeddings \
        -e DEF_EXTRACT_GA=$detect_ga \
        -e CUDA_CACHE_DISABLE=$cuda_cache_disable \
        -v $PWD/models:/models \
        -v $PWD/src:/app \
        --health-cmd='curl -f http://localhost:18080/info || exit 1' \
        --health-interval=1m \
        --health-timeout=10s \
        --health-retries=3 \
        --name=$name \
        $IMAGE:$TAG

    echo "Container $name started successfully!"
done

echo "=============================================="
echo "Deployment completed!"
echo ""
echo "Access the API at:"
for i in $(seq 0 $(($n_gpu - 1))); do
    port=$((START_PORT + $i))
    echo "  http://localhost:$port"
    echo "  http://localhost:$port/docs (API documentation)"
done
echo ""
echo "To check container status:"
echo "  docker ps"
echo ""
echo "To view container logs:"
echo "  docker logs $IMAGE-gpu0-jetson"
echo ""
echo "To stop containers:"
echo "  docker stop $IMAGE-gpu0-jetson"
echo "==============================================" 