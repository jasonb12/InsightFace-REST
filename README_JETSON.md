# 🚀 InsightFace-REST for NVIDIA Jetson AGX Orin (GPU Accelerated)

**Branch:** `jetson_runner` - GPU-accelerated deployment  
**Platform:** NVIDIA Jetson AGX Orin with JetPack 6.2  
**Achievement:** **7.23x GPU speedup confirmed** ⚡  

## 🎯 QUICK START

```bash
# Clone the GPU-accelerated Jetson branch
git clone -b jetson_runner https://github.com/jasonb12/InsightFace-REST.git
cd InsightFace-REST

# Deploy with GPU acceleration (one command)
./deploy_jetson_gpu.sh
```

## 📊 PERFORMANCE RESULTS

| Execution Provider | Average Time | Speedup |
|-------------------|--------------|---------|
| **CUDA GPU** | **16.1ms** | **7.23x** |
| CPU Only | 118.9ms | 1.0x |

## 🔧 TECHNICAL STACK

- **Base**: NVIDIA L4T JetPack r36.4.0
- **CUDA**: 12.6.10 
- **TensorRT**: 10.3.0
- **cuDNN**: 9.3.0
- **ONNX Runtime**: dustynv's GPU-optimized build
- **Models**: scrfd_10g_gnkps (detection), glintr100 (recognition)

## 🌐 API ACCESS

After deployment:
- **Documentation**: http://localhost:18081/docs
- **Health Check**: http://localhost:18081/info
- **Extract API**: http://localhost:18081/extract

## 📚 COMPREHENSIVE DOCUMENTATION

- **[JETSON_GPU_SUCCESS.md](JETSON_GPU_SUCCESS.md)** - Complete technical details
- **[JETSON_DEPLOYMENT_PLAN.md](JETSON_DEPLOYMENT_PLAN.md)** - Original deployment strategy
- **[JETSON_PERFORMANCE_RESULTS.md](JETSON_PERFORMANCE_RESULTS.md)** - Detailed benchmarks

## 🧪 TESTING & VERIFICATION

- `gpu_inference_test.py` - Performance verification
- `gpu_load_test.py` - Stress testing  
- `debug_gpu_providers.py` - Provider diagnostics

## 🏆 ACHIEVEMENT

✅ **Production-ready GPU acceleration on Jetson AGX Orin**  
✅ **7.23x performance improvement over CPU**  
✅ **Real-time face detection at 16ms per image**  
✅ **Full JetPack 6.2 compatibility**  

---

**Note:** This branch specifically targets GPU-accelerated deployment on NVIDIA Jetson platforms. For CPU-only deployment, see the master branch. 