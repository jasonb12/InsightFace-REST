import onnxruntime as ort

print("=== ONNX Runtime GPU Test ===")
print("Available providers:", ort.get_available_providers())

# Test detection model
try:
    session = ort.InferenceSession('/models/onnx/scrfd_10g_gnkps/scrfd_10g_gnkps.onnx', 
                                  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Detection model providers:", session.get_providers())
except Exception as e:
    print("Detection model error:", e)

# Test recognition model  
try:
    session2 = ort.InferenceSession('/models/onnx/glintr100/glintr100.onnx',
                                   providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Recognition model providers:", session2.get_providers())
except Exception as e:
    print("Recognition model error:", e) 