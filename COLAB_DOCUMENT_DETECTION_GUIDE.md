# YOLOX Document Detection - Complete Colab Training Guide

Step-by-step guide to train YOLOX for Algerian ID card detection on Google Colab.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Prepare Your Dataset](#step-1-prepare-your-dataset)
3. [Step 2: Upload to Google Drive](#step-2-upload-to-google-drive)
4. [Step 3: Setup Colab Environment](#step-3-setup-colab-environment)
5. [Step 4: Train the Model](#step-4-train-the-model)
6. [Step 5: Test & Export](#step-5-test--export)
7. [Step 6: Download & Local Inference](#step-6-download--local-inference)

---

## Prerequisites

### What You Need

- Google account (for Google Drive and Colab)
- Dataset of Algerian ID card images (minimum 500 images recommended)
- 2-4 hours of training time (depending on dataset size)

### Dataset Requirements

**Option A: Use Your Own Images**
- Photos of Algerian ID cards (front side)
- Various angles, lighting conditions, backgrounds
- Minimum: 500 images (400 train / 100 val)
- Recommended: 2000+ images

**Option B: Generate Synthetic Data**
- Use the synthetic generator (see below)
- Mix with real images for best results

---

## Step 1: Prepare Your Dataset

### 1.1 Collect Images

Take photos of Algerian ID cards with variations:
- Different angles (0°, 15°, 30° rotation)
- Different lighting (bright, dim, shadow)
- Different backgrounds (table, hand, desk)
- Different distances (close, medium, far)

### 1.2 Annotate Images

Use a labeling tool to create bounding boxes around ID cards:

**Recommended Tools:**
- [LabelImg](https://github.com/tzutalin/labelImg) (Free, desktop)
- [CVAT](https://cvat.org/) (Free, online)
- [Roboflow](https://roboflow.com/) (Free tier available)

**Annotation Format: COCO JSON**

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "id_card_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 50, 400, 300],
      "area": 120000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "id_card"
    }
  ]
}
```

### 1.3 Organize Dataset Structure

Create this folder structure on your computer:

```
algerian_id_cards/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    │   ├── img_101.jpg
    │   ├── img_102.jpg
    │   └── ...
    └── annotations.json
```

### 1.4 (Alternative) Generate Synthetic Dataset

If you don't have real images, generate synthetic ones:

```bash
# On your local machine
cd retin-verify/V3/training
python scripts/generate_synthetic_documents.py \
    --output-dir synthetic_data \
    --num-samples 2000
```

This creates realistic synthetic ID cards with various augmentations.

---

## Step 2: Upload to Google Drive

### 2.1 Create Folder Structure

In Google Drive, create:

```
MyDrive/
├── datasets/
│   └── algerian_id_cards/     ← Upload dataset here
│       ├── train/
│       │   ├── images/
│       │   └── annotations.json
│       └── val/
│           ├── images/
│           └── annotations.json
└── models/                     ← Trained models will be saved here
```

### 2.2 Upload Dataset

**Method 1: Web Browser**
1. Go to [drive.google.com](https://drive.google.com)
2. Navigate to `MyDrive/datasets/`
3. Drag and drop your `algerian_id_cards` folder
4. Wait for upload to complete

**Method 2: Google Drive Desktop App**
1. Install Google Drive for Desktop
2. Copy `algerian_id_cards` folder to your Drive folder
3. Let it sync automatically

**Method 3: Command Line (gdrive)**
```bash
# Install gdrive
gdrive upload --recursive algerian_id_cards/
```

### 2.3 Verify Upload

Check that your Drive has:
- ✅ `MyDrive/datasets/algerian_id_cards/train/images/` with .jpg files
- ✅ `MyDrive/datasets/algerian_id_cards/train/annotations.json`
- ✅ `MyDrive/datasets/algerian_id_cards/val/images/` with .jpg files
- ✅ `MyDrive/datasets/algerian_id_cards/val/annotations.json`

---

## Step 3: Setup Colab Environment

### 3.1 Open the Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Select `retin-verify/V3/colab/yolox_document_detection.ipynb`
4. Or open from GitHub: paste the notebook URL

### 3.2 Enable GPU

**CRITICAL: Must use GPU for training**

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** from Hardware accelerator dropdown
3. Click **Save**
4. Verify GPU is available:

```python
# Run this in a cell
!nvidia-smi
```

Expected output: Shows GPU info (Tesla T4, K80, or P100)

### 3.3 Mount Google Drive

Run the first cells to mount your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

When prompted:
1. Click the link
2. Select your Google account
3. Copy the authorization code
4. Paste in Colab
5. Press Enter

### 3.4 Verify Dataset Access

```python
# Check dataset is accessible
!ls -la /content/drive/MyDrive/datasets/algerian_id_cards/train/images/ | head -10
!cat /content/drive/MyDrive/datasets/algerian_id_cards/train/annotations.json | head -50
```

---

## Step 4: Train the Model

### 4.1 Clone YOLOX

```python
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
%cd YOLOX
```

### 4.2 Install Dependencies

```python
!pip install -q torch torchvision torchaudio
!pip install -q -e .
!pip install -q onnx onnxruntime
```

### 4.3 Copy Dataset to Colab

```python
# Faster training when data is local
!mkdir -p /content/dataset
!cp -r /content/drive/MyDrive/datasets/algerian_id_cards/* /content/dataset/
```

### 4.4 Create Custom Config

The notebook has a config cell. Key parameters you can adjust:

```python
# Training parameters
self.max_epoch = 100        # Increase for better accuracy (150-200)
self.warmup_epochs = 5
self.basic_lr_per_img = 0.01 / 64.0

# Data augmentation
self.mosaic_prob = 1.0      # Mosaic augmentation probability
self.mixup_prob = 0.5       # Mixup augmentation probability
self.hsv_prob = 1.0         # HSV color augmentation

# Input size
self.input_size = (640, 640)  # Can use (416, 416) for faster training
```

### 4.5 Download Pretrained Weights (Optional but Recommended)

```python
!wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
!mkdir -p /content/YOLOX/pretrained
!mv yolox_s.pth /content/YOLOX/pretrained/
```

### 4.6 Start Training

```python
!python tools/train.py \
    -f exps/custom/yolox_idcard.py \
    -d 1 \              # 1 GPU
    -b 16 \             # Batch size (reduce to 8 if OOM)
    --fp16 \            # Mixed precision training
    -o \                # Occupy GPU memory
    -c pretrained/yolox_s.pth  # Pretrained weights
```

**Training Time:**
- 100 epochs × 2000 images ≈ 2-3 hours on Tesla T4
- Monitor loss values - should decrease steadily

**If Out of Memory:**
```python
# Reduce batch size
-b 8  # or even -b 4

# Reduce input size
self.input_size = (416, 416)
```

---

## Step 5: Test & Export

### 5.1 Export to ONNX

```python
!python tools/export_onnx.py \
    --output-name yolox_idcard.onnx \
    -f exps/custom/yolox_idcard.py \
    -c YOLOX_outputs/yolox_idcard/best_ckpt.pth
```

### 5.2 Test Inference

```python
# Run demo on test image
!python tools/demo.py \
    image \
    -f exps/custom/yolox_idcard.py \
    -c YOLOX_outputs/yolox_idcard/best_ckpt.pth \
    --path /content/dataset/val/images/test_001.jpg \
    --conf 0.5 \
    --nms 0.45 \
    --tsize 640 \
    --save_result
```

### 5.3 Check Results

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load result
result = Image.open('YOLOX_outputs/yolox_idcard/vis_res/test_001.jpg')
plt.figure(figsize=(10, 10))
plt.imshow(result)
plt.title('Detection Result')
plt.axis('off')
plt.show()
```

### 5.4 Save to Google Drive

```python
# Create models folder
!mkdir -p /content/drive/MyDrive/models

# Copy ONNX model
!cp yolox_idcard.onnx /content/drive/MyDrive/models/

# Copy checkpoint (optional, for retraining)
!cp YOLOX_outputs/yolox_idcard/best_ckpt.pth /content/drive/MyDrive/models/

print("✅ Model saved to Google Drive!")
```

---

## Step 6: Download & Local Inference

### 6.1 Download Model from Drive

**Method 1: Web Browser**
1. Go to [drive.google.com](https://drive.google.com)
2. Navigate to `MyDrive/models/`
3. Download `yolox_idcard.onnx`

**Method 2: Command Line (gdrive)**
```bash
gdrive download --recursive MyDrive/models/yolox_idcard.onnx
```

### 6.2 Place Model in Project

```bash
# On your local machine
cd retin-verify/V3
mkdir -p models
cp ~/Downloads/yolox_idcard.onnx models/
```

### 6.3 Test Local Inference

Create a test script:

```python
# test_document_detection.py
import cv2
import numpy as np
from src.document_detection.detector import DocumentDetector

def test_detection(image_path, model_path="models/yolox_idcard.onnx"):
    # Initialize detector
    detector = DocumentDetector(model_path=model_path)
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect
    result = detector.detect(image)
    
    print(f"Detection Result:")
    print(f"  Bounding Box: {result.bbox}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Inference Time: {result.inference_time:.3f}s")
    
    # Visualize
    vis_image = detector.visualize(image, result)
    
    # Save result
    output_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"  Output saved to: {output_path}")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_images/id_card_sample.jpg"
    
    test_detection(image_path)
```

Run the test:

```bash
cd retin-verify/V3
python test_document_detection.py test_images/your_image.jpg
```

### 6.4 Run Full Pipeline Test

```bash
# Run the existing tests
pytest tests/test_document_detector.py -v

# Or test with your own image
python -c "
from src.document_detection.detector import DocumentDetector
import cv2

detector = DocumentDetector('models/yolox_idcard.onnx')
img = cv2.imread('your_image.jpg')
result = detector.detect(img)
print(f'Detected: {result.bbox}, Confidence: {result.confidence}')
"
```

---

## Expected Results

### Training Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| mAP@0.5 | > 0.95 | > 0.90 |
| mAP@0.5:0.95 | > 0.70 | > 0.60 |
| Loss | < 2.0 | < 3.0 |

### Inference Performance

| Platform | Speed | FPS |
|----------|-------|-----|
| GPU (RTX 3060) | ~10ms | ~100 |
| CPU (i7) | ~50ms | ~20 |
| Edge (Raspberry Pi) | ~500ms | ~2 |

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```python
# Reduce batch size
-b 8  # or -b 4

# Reduce image size
self.input_size = (416, 416)
self.test_size = (416, 416)
```

### Issue: "Dataset not found"

**Solution:**
```python
# Verify path
!ls -la /content/drive/MyDrive/datasets/algerian_id_cards/

# Check mount
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Issue: "Low accuracy"

**Solutions:**
1. Increase dataset size (minimum 500 images)
2. Train longer (150-200 epochs)
3. Check annotations are correct
4. Use pretrained weights
5. Increase data augmentation

### Issue: "Model not saving to Drive"

**Solution:**
```python
# Ensure Drive is mounted
from google.colab import drive
drive.mount('/content/drive')

# Create directory if needed
!mkdir -p /content/drive/MyDrive/models

# Copy with verbose
!cp -v yolox_idcard.onnx /content/drive/MyDrive/models/
```

---

## Next Steps

After successful training:

1. **Train MRZ OCR**: Use `colab/paddleocr_mrz_training.ipynb`
2. **Download Face Detection**: Use `colab/download_yunet_model.ipynb`
3. **Test Full Pipeline**: Run `pytest tests/ -v`
4. **Deploy API**: `uvicorn api.main:app --reload`

---

## Quick Reference

### Colab Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run cell |
| `Shift+Enter` | Run cell and select next |
| `Ctrl+Shift+Enter` | Run cell and insert below |
| `Ctrl+M B` | Insert code cell below |
| `Ctrl+M D` | Delete cell |

### Useful Colab Commands

```python
# Check GPU
!nvidia-smi

# Check disk space
!df -h

# Check memory
!free -h

# List files
!ls -la

# Download file
from google.colab import files
files.download('yolox_idcard.onnx')
```
