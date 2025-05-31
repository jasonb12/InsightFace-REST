#!/usr/bin/env python3

import onnxruntime as ort
import os

print("=== ONNX NODE ASSIGNMENT DEBUG ===")

# Enable verbose logging to see node assignments
os.environ['ORT_LOG_LEVEL'] = '1'  # Verbose
ort.set_default_logger_severity(1)

print("\n1. DETECTION MODEL NODE ASSIGNMENT:")
print("=" * 50)

# Test detection model with detailed logging
try:
    # Create session with explicit provider configuration
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider'
    ]
    
    print("Creating detection model session with TensorRT -> CUDA -> CPU fallback...")
    session = ort.InferenceSession('/models/onnx/scrfd_10g_gnkps/scrfd_10g_gnkps.onnx', 
                                 providers=providers)
    
    print(f"Actual providers used: {session.get_providers()}")
    
    # Get model inputs/outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print(f"Model inputs: {[(inp.name, inp.shape, inp.type) for inp in inputs]}")
    print(f"Model outputs: {[(out.name, out.shape, out.type) for out in outputs]}")
    
except Exception as e:
    print(f"Detection model error: {e}")

print("\n2. RECOGNITION MODEL NODE ASSIGNMENT:")
print("=" * 50)

try:
    print("Creating recognition model session with TensorRT -> CUDA -> CPU fallback...")
    session2 = ort.InferenceSession('/models/onnx/glintr100/glintr100.onnx',
                                  providers=providers)
    
    print(f"Actual providers used: {session2.get_providers()}")
    
    # Get model inputs/outputs
    inputs2 = session2.get_inputs()
    outputs2 = session2.get_outputs()
    
    print(f"Model inputs: {[(inp.name, inp.shape, inp.type) for inp in inputs2]}")
    print(f"Model outputs: {[(out.name, out.shape, out.type) for out in outputs2]}")
    
except Exception as e:
    print(f"Recognition model error: {e}")

print("\n3. TESTING CPU-ONLY FOR COMPARISON:")
print("=" * 50)

try:
    print("Creating CPU-only session...")
    cpu_session = ort.InferenceSession('/models/onnx/scrfd_10g_gnkps/scrfd_10g_gnkps.onnx',
                                     providers=['CPUExecutionProvider'])
    print(f"CPU session providers: {cpu_session.get_providers()}")
    
except Exception as e:
    print(f"CPU session error: {e}")

print("\n=== NODE ASSIGNMENT DEBUG COMPLETE ===") 