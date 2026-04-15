# Guide: Build a Tiny Front/Back Classifier for 416 Model

## Goal
Pair the fast `yolox_idcard_416.onnx` detector with a tiny CNN classifier that runs in ~10-30ms to accurately distinguish:
- `id-front`
- `id-back`
- `no-card` (rejects false positives)

## Why This Works
- YOLOX 416 detects "card present" very fast (~157ms)
- A tiny classifier on the cropped bbox is much faster than running full YOLOX 640
- Total latency: ~157ms + ~20ms = **~180ms**

---

## Step 1: Extract Cropped Training Images

Run this script to crop bbox regions from your annotated dataset:

```python
# extract_crops.py
import json
import cv2
import numpy as np
from pathlib import Path

def extract_crops(annotation_json, image_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(annotation_json) as f:
        data = json.load(f)
    
    for ann in data['annotations']:
        img_info = next(i for i in data['images'] if i['id'] == ann['image_id'])
        img_path = Path(image_dir) / img_info['file_name']
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        x, y, w, h = map(int, ann['bbox'])
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        crop = img[y:y+h, x:x+w]
        
        # Resize to fixed size for classifier
        crop = cv2.resize(crop, (224, 224))
        
        class_name = 'id-front' if ann['category_id'] == 0 else 'id-back'
        out_path = output_dir / class_name / img_info['file_name']
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), crop)

# Extract from train and val
extract_crops('training/algerian_id_cards/train/annotations.json',
              'training/algerian_id_cards/train/images',
              'training/classifier_crops/train')

extract_crops('training/algerian_id_cards/val/annotations.json',
              'training/algerian_id_cards/val/images',
              'training/classifier_crops/val')

# Also extract random no-card crops from no-card images
import random
no_card_dir = Path('/mnt/d/dataset_280326/no-card')
no_card_out = Path('training/classifier_crops/train/no-card')
no_card_out.mkdir(parents=True, exist_ok=True)
for img_path in list(no_card_dir.glob('*.jpg'))[:200]:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    # Random crop same size as typical card
    h, w = img.shape[:2]
    ch, cw = min(h, w) // 2, min(h, w) // 3
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    crop = img[y:y+ch, x:x+cw]
    crop = cv2.resize(crop, (224, 224))
    cv2.imwrite(str(no_card_out / img_path.name), crop)

print("Crops extracted!")
```

---

## Step 2: Train a Tiny Classifier

Use this simple PyTorch script. You can run it locally (CPU is fine, ~10 minutes) or in Colab.

```python
# train_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder('training/classifier_crops/train', transform=transform)
val_ds = datasets.ImageFolder('training/classifier_crops/val', transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Tiny MobileNetV2
model = models.mobilenet_v2(weights='IMAGENET1K_V2')
model.classifier[1] = nn.Linear(model.last_channel, 3)  # front, back, no-card
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    print(f'Epoch {epoch+1}: val_acc={acc:.3f}')
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'models/id_card_classifier.pth')
        
print(f'Best accuracy: {best_acc:.3f}')
```

---

## Step 3: Export to ONNX

```python
# export_classifier.py
import torch
from torchvision import models
import torch.nn as nn

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 3)
model.load_state_dict(torch.load('models/id_card_classifier.pth', map_location='cpu'))
model.eval()

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/id_card_classifier.onnx',
    input_names=['image'], output_names=['class_probs'],
    opset_version=11, dynamic_axes={'image': {0: 'batch'}})
print("Classifier exported!")
```

---

## Step 4: Integrate into Detector

Modify `src/document_detection/detector.py` to run the classifier on each YOLOX detection:

```python
# In DocumentDetector.__init__, load the classifier
self.classifier = ort.InferenceSession('models/id_card_classifier.onnx', providers=['CPUExecutionProvider'])
self.classifier_input = self.classifier.get_inputs()[0].name

# In postprocess, after getting the bbox, classify the crop
for pred in predictions:
    # ... existing confidence and bbox logic ...
    
    # Crop from original image
    x1, y1, x2, y2 = ...
    crop = orig_image[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    crop = cv2.resize(crop, (224, 224))
    crop = crop[:, :, ::-1]  # BGR->RGB
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    crop = np.transpose(crop, (2, 0, 1))
    crop = np.expand_dims(crop, axis=0)
    
    probs = self.classifier.run(None, {self.classifier_input: crop})[0][0]
    class_id = int(np.argmax(probs))
    class_score = float(probs[class_id])
    
    # Reject if classifier says no-card
    if class_id == 2:  # no-card
        continue
    
    confidence = obj_conf * class_score
    class_name = self.class_names.get(class_id, "unknown")
```

---

## Expected Performance

| Component | Time |
|---|---|
| YOLOX 416 detection | ~157ms |
| Crop + resize | ~5ms |
| MobileNetV2 classifier | ~15ms |
| **Total** | **~180ms** |
| **Accuracy** | **>95% front/back** |

---

## Next Steps

1. Run `extract_crops.py`
2. Run `train_classifier.py`
3. Run `export_classifier.py`
4. I can then modify `detector.py` to integrate it

Want me to create these scripts as actual files in your repo?
