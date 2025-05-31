import requests
import base64
import time

# Load test image
with open('/app/Stallone.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('ascii')

data = {
    'images': {'data': [img_data]},
    'extract_embedding': True,
    'return_face_data': False
}

print('Starting GPU test...')
start_time = time.time()
response = requests.post('http://localhost:18080/extract', json=data, timeout=30)
result = response.json()
end_time = time.time()

print(f'Status: {result["data"][0]["status"]}')
print(f'Processing time: {result["took"]["total_ms"]:.1f} ms')
print(f'Total time: {(end_time - start_time)*1000:.1f} ms')
print(f'Faces detected: {len(result["data"][0]["faces"])}')
print('GPU test completed!') 