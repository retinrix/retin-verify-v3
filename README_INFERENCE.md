# YOLOX Document Detection - Inference Apps

Once you download your trained `yolox_idcard.onnx` model, place it in:
```
retin-verify/V3/models/yolox_idcard.onnx
```

Then use any of these inference apps to test it.

---

## 1. Command Line App (`inference_app.py`)

Simplest option — run from terminal.

```bash
cd retin-verify/V3
source venv/bin/activate

# Basic usage
python inference_app.py --image test_images/id_card.jpg

# With custom model path
python inference_app.py --image test_images/id_card.jpg --model models/yolox_idcard.onnx

# Lower confidence threshold
python inference_app.py --image test_images/id_card.jpg --conf 0.3

# Custom output path
python inference_app.py --image test_images/id_card.jpg --output result.jpg
```

Output will be saved as `id_card_detected.jpg` in the same folder.

---

## 2. Web App (`web_inference_app.py`)

Upload images via browser and see detection results instantly.

```bash
cd retin-verify/V3
source venv/bin/activate

# Start server
python web_inference_app.py --model models/yolox_idcard.onnx --port 8080
```

Then open **http://localhost:8080** in your browser.

Features:
- Drag & drop or click to upload images
- See bounding boxes drawn on results
- View confidence scores and bbox coordinates
- See inference time per image

---

## 3. Python API (for integration)

Use the `DocumentDetector` directly in your code:

```python
import cv2
from src.document_detection import DocumentDetector

detector = DocumentDetector(model_path="models/yolox_idcard.onnx")

image = cv2.imread("test_images/id_card.jpg")
results = detector.detect(image)

for det in results:
    print(f"Detected: {det.class_name} @ {det.confidence:.2f}")
    print(f"BBox: {det.bbox}")

# Visualize
vis = detector.visualize(image, results)
cv2.imwrite("output.jpg", vis)
```

---

## Model Performance

From training (epoch 86):
- **mAP@0.5**: 100.0%
- **mAP@0.5:0.95**: 84.6%
- **Inference time**: ~10 ms on GPU, ~300 ms on CPU
