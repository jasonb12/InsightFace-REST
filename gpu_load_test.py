#!/usr/bin/env python3

import requests
import base64
import time
import threading
import concurrent.futures
from statistics import mean, stdev

def load_test_image(image_path):
    """Load and encode test image"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def single_inference_test(image_data, test_id):
    """Single inference request"""
    data = {
        'images': {'data': [image_data]},
        'extract_embedding': True,
        'return_face_data': False,
        'return_landmarks': False
    }
    
    start_time = time.time()
    try:
        response = requests.post('http://localhost:18080/extract', json=data, timeout=60)
        result = response.json()
        end_time = time.time()
        
        if 'detail' in result:
            return {'error': result['detail'], 'test_id': test_id}
            
        processing_time = result["took"]["total_ms"]
        total_time = (end_time - start_time) * 1000
        faces_detected = len(result["data"][0]["faces"])
        
        return {
            'test_id': test_id,
            'processing_time': processing_time,
            'total_time': total_time,
            'faces_detected': faces_detected,
            'status': 'success'
        }
    except Exception as e:
        return {'error': str(e), 'test_id': test_id}

def run_load_test():
    print("=== InsightFace GPU Load Test ===")
    print("Loading test images...")
    
    # Load test images
    stallone_data = load_test_image('/app/Stallone.jpg')
    
    # Test configurations
    tests = [
        {"name": "Single Face (Stallone)", "data": stallone_data, "runs": 20},
    ]
    
    all_results = []
    
    for test_config in tests:
        print(f"\n🔥 Running {test_config['name']} - {test_config['runs']} iterations")
        print("=" * 60)
        
        test_results = []
        start_time = time.time()
        
        # Sequential test for baseline
        print(f"Sequential execution...")
        for i in range(test_config['runs']):
            print(f"  Test {i+1}/{test_config['runs']}", end=' ')
            result = single_inference_test(test_config['data'], i+1)
            if 'error' not in result:
                test_results.append(result)
                print(f"✅ {result['processing_time']:.1f}ms ({result['faces_detected']} faces)")
            else:
                print(f"❌ {result['error']}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if test_results:
            processing_times = [r['processing_time'] for r in test_results]
            total_times = [r['total_time'] for r in test_results]
            
            print(f"\n📊 Results for {test_config['name']}:")
            print(f"  Successful runs: {len(test_results)}/{test_config['runs']}")
            print(f"  Total duration: {total_duration:.1f}s")
            print(f"  Average processing time: {mean(processing_times):.1f}ms (±{stdev(processing_times):.1f}ms)")
            print(f"  Average total time: {mean(total_times):.1f}ms")
            print(f"  Throughput: {len(test_results)/total_duration:.2f} inferences/second")
            print(f"  Min/Max processing: {min(processing_times):.1f}ms / {max(processing_times):.1f}ms")
            
            all_results.extend(test_results)
    
    # Parallel load test
    print(f"\n🚀 PARALLEL LOAD TEST - High GPU Utilization")
    print("=" * 60)
    parallel_results = []
    
    # Run 10 parallel requests
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(16):  # 16 parallel requests for heavy load
            future = executor.submit(single_inference_test, stallone_data, f"parallel_{i+1}")
            futures.append(future)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if 'error' not in result:
                parallel_results.append(result)
                print(f"  Parallel {i+1}/16 ✅ {result['processing_time']:.1f}ms")
            else:
                print(f"  Parallel {i+1}/16 ❌ {result['error']}")
    
    end_time = time.time()
    parallel_duration = end_time - start_time
    
    if parallel_results:
        parallel_times = [r['processing_time'] for r in parallel_results]
        print(f"\n📊 Parallel Load Test Results:")
        print(f"  Successful parallel runs: {len(parallel_results)}/16")
        print(f"  Total parallel duration: {parallel_duration:.1f}s")
        print(f"  Average processing time: {mean(parallel_times):.1f}ms")
        print(f"  Parallel throughput: {len(parallel_results)/parallel_duration:.2f} inferences/second")
    
    print(f"\n🎯 GPU Load Test Complete!")
    print(f"Total inferences executed: {len(all_results) + len(parallel_results)}")
    
if __name__ == '__main__':
    run_load_test() 