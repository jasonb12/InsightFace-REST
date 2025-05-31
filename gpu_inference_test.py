#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import time
import os

# Enable detailed session logging
os.environ['ORT_LOG_LEVEL'] = '2'  # Warnings
ort.set_default_logger_severity(2)

print("=== REAL GPU INFERENCE VERIFICATION ===")

def test_gpu_inference():
    """Test if actual inference operations use GPU"""
    
    # Test with detection model first
    detection_model = '/models/onnx/scrfd_10g_gnkps/scrfd_10g_gnkps.onnx'
    
    print("\n1. TESTING ACTUAL GPU INFERENCE:")
    print("=" * 50)
    
    # Create sessions with different providers
    providers_configs = [
        (['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], "TensorRT->CUDA->CPU"),
        (['CUDAExecutionProvider', 'CPUExecutionProvider'], "CUDA->CPU"),
        (['CPUExecutionProvider'], "CPU Only")
    ]
    
    # Create dummy input (640x640)
    dummy_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
    
    results = []
    
    for providers, name in providers_configs:
        try:
            print(f"\n   Testing {name}...")
            
            # Create session
            session = ort.InferenceSession(detection_model, providers=providers)
            actual_providers = session.get_providers()
            print(f"     Loaded providers: {actual_providers[0]}")
            
            # Warm up
            for _ in range(3):
                _ = session.run(None, {'input.1': dummy_input})
            
            # Timed inference
            times = []
            for _ in range(5):
                start = time.time()
                outputs = session.run(None, {'input.1': dummy_input})
                end = time.time()
                times.append((end - start) * 1000)
            
            avg_time = sum(times) / len(times)
            results.append((name, actual_providers[0], avg_time, times))
            print(f"     Average inference time: {avg_time:.1f}ms")
            print(f"     Individual times: {[f'{t:.1f}ms' for t in times]}")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results.append((name, "ERROR", 0, []))
    
    print(f"\n2. PERFORMANCE COMPARISON:")
    print("=" * 50)
    for name, provider, avg_time, times in results:
        if avg_time > 0:
            print(f"   {name:20} | Provider: {provider:25} | Time: {avg_time:6.1f}ms")
    
    # Check for GPU acceleration evidence
    if len(results) >= 2:
        cpu_time = next((r[2] for r in results if r[0] == "CPU Only"), None)
        gpu_time = next((r[2] for r in results if "TensorRT" in r[0] or "CUDA" in r[0]), None)
        
        if cpu_time and gpu_time and cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n   🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
            if speedup > 1.5:
                print("   ✅ GPU acceleration is working!")
            else:
                print("   ⚠️ Limited GPU acceleration - may be CPU fallback")
        
    print(f"\n3. TESTING RECOGNITION MODEL:")
    print("=" * 50)
    
    # Test recognition model
    recognition_model = '/models/onnx/glintr100/glintr100.onnx'
    face_input = np.random.random((1, 3, 112, 112)).astype(np.float32)
    
    try:
        session = ort.InferenceSession(recognition_model, 
                                     providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"   Recognition model providers: {session.get_providers()[0]}")
        
        # Timed recognition inference
        times = []
        for _ in range(5):
            start = time.time()
            _ = session.run(None, {'input.1': face_input})
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"   Recognition inference time: {avg_time:.1f}ms")
        
    except Exception as e:
        print(f"   ❌ Recognition test failed: {e}")

if __name__ == '__main__':
    test_gpu_inference() 