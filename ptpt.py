import torch
import urllib.request
url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
urllib.request.urlretrieve(url, 'yolov8n.pt')

model = torch.hub.load('ultralytics/yolov8', 'custom', path='yolov8n.pt', source='local')