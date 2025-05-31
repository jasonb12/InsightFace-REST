#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import time

print("=== DETAILED GPU EXECUTION PROVIDER DEBUG ===")
print()

# Check available providers
print("1. AVAILABLE PROVIDERS:")
available = ort.get_available_providers()
print(f"   {available}")
print()

# Check GPU specific information
print("2. CUDA AVAILABILITY:")
try:
    import cuda
    print(f"   CUDA module available: Yes")
except ImportError:
    print(f"   CUDA module available: No")

try:
    import cupy
    print(f"   CuPy available: Yes")
    print(f"   CuPy CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}")
    print(f"   Available GPUs: {cupy.cuda.runtime.getDeviceCount()}")
except ImportError as e:
    print(f"   CuPy available: No - {e}")
print()

# Test each execution provider explicitly
providers_to_test = [
    (['TensorrtExecutionProvider'], "TensorRT"),
    (['CUDAExecutionProvider'], "CUDA"),
    (['CPUExecutionProvider'], "CPU")
]

detection_model = '/models/onnx/scrfd_10g_gnkps/scrfd_10g_gnkps.onnx'
recognition_model = '/models/onnx/glintr100/glintr100.onnx'

for model_path, model_name in [(detection_model, "Detection"), (recognition_model, "Recognition")]:
    print(f"3. {model_name.upper()} MODEL PROVIDER TEST:")
    
    for providers, provider_name in providers_to_test:
        try:
            print(f"   Testing {provider_name}...", end=" ")
            session = ort.InferenceSession(model_path, providers=providers)
            actual_providers = session.get_providers()
            print(f"✅ Success - Using: {actual_providers[0]}")
            
            # Quick inference test
            if provider_name == "TensorRT":
                print(f"      TensorRT session created successfully")
            elif provider_name == "CUDA":
                print(f"      CUDA session created successfully")
                
        except Exception as e:
            print(f"❌ Failed - {e}")
    print()

# Test with forced GPU providers (current app configuration)
print("4. CURRENT APP CONFIGURATION TEST:")
try:
    print("   Testing detection model with current app config...")
    session = ort.InferenceSession(detection_model, 
                                 providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"   ✅ Detection: {session.get_providers()}")
    
    print("   Testing recognition model with current app config...")  
    session2 = ort.InferenceSession(recognition_model,
                                  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"   ✅ Recognition: {session2.get_providers()}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
print()

# Check CUDA context and device information
print("5. CUDA DEVICE INFORMATION:")
try:
    import pycuda.driver as cuda
    cuda.init()
    print(f"   CUDA devices: {cuda.Device.count()}")
    for i in range(cuda.Device.count()):
        device = cuda.Device(i)
        print(f"   Device {i}: {device.name()}")
except ImportError:
    print("   PyCUDA not available")
except Exception as e:
    print(f"   CUDA error: {e}")
print()

print("6. ONNX RUNTIME BUILD INFO:")
print(f"   Version: {ort.__version__}")
print("   Build info available providers:", ort.get_available_providers())
print()

print("=== DEBUG COMPLETE ===") 