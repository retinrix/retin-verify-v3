# Retin-Verify V3 - Model Training Guide

Complete guide for training all V3 models from scratch.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Step-by-Step Training](#step-by-step-training)
5. [Google Colab Training](#google-colab-training)
6. [Troubleshooting](#troubleshooting)

---

## Overview

V3 uses 4 models for ID card verification:

| Model | Purpose | Framework | Training Required |
|-------|---------|-----------|-------------------|
| **Document Detection** | Locate ID card in image | YOLOX | ✅ Yes |
| **MRZ OCR** | Read MRZ text | PaddleOCR | ✅ Yes |
| **Face Detection** | Extract face photo | YuNet | ❌ No (pre-trained) |
| **Security Features** | Detect holograms/laser | Custom CV | ⚠️ Optional |

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060 12GB+ |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB free |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

### Software Requirements

```bash
# Check CUDA version
nvidia-smi

# Required versions
CUDA >= 11.7
cuDNN >= 8.5
Python >= 3.8
```

### Python Environment

```bash
cd retin-verify/V3
python -m venv venv
source venv/bin/activate

# Install base dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python opencv-contrib-python
pip install pillow numpy matplotlib
```

---

## Quick Start

### Option 1: Automated Training (Local)

```bash
cd training

# Train everything from scratch
python scripts/train_all.py

# Quick test with smaller dataset
python scripts/train_all.py \
    --num-document-samples 200 \
    --num-mrz-samples 500 \
    --epochs-yolox 10
```

### Option 2: Google Colab (Recommended)

1. Open the notebooks in `colab/` directory
2. Upload to Google Colab
3. Run all cells
4. Download trained models

---

## Step-by-Step Training

### Step 1: Document Detection (YOLOX)

#### 1.1 Generate Training Data

```bash
python scripts/generate_synthetic_documents.py \
    --output-dir data/document_detection \
    --num-samples 2000 \
    --img-size 640
```

This creates:
- `data/document_detection/images/` - Synthetic ID card images
- `data/document_detection/labels/` - YOLO format bounding boxes

#### 1.2 Train Model

```bash
python scripts/train_yolox.py \
    --data-dir data/document_detection \
    --batch-size 16 \
    --epochs 100 \
    --output-dir models
```

**Training time:** ~2-4 hours on RTX 3060

**Expected output:**
- `models/yolox_idcard.onnx` - ONNX model for inference
- mAP@0.5: > 0.95

#### 1.3 Test Model

```python
from src.document_detection.detector import DocumentDetector

detector = DocumentDetector(model_path="models/yolox_idcard.onnx")
result = detector.detect("test_image.jpg")
print(f"Detected: {result.bbox}, Confidence: {result.confidence}")
```

---

### Step 2: MRZ OCR (PaddleOCR)

#### 2.1 Generate Training Data

```bash
python scripts/generate_synthetic_mrz.py \
    --output-dir data/mrz_ocr \
    --num-samples 5000 \
    --split-ratio 0.9
```

This creates:
- `data/mrz_ocr/train/images/` - Training MRZ images
- `data/mrz_ocr/train/label.txt` - Labels
- `data/mrz_ocr/val/` - Validation set

#### 2.2 Train Model

```bash
python scripts/train_paddleocr.py \
    --data-dir data/mrz_ocr \
    --output-dir models
```

**Training time:** ~3-5 hours on RTX 3060

**Expected output:**
- `models/paddleocr_mrz/` - Paddle inference model
- `models/paddleocr_mrz.onnx` - ONNX model
- Character accuracy: > 99%

#### 2.3 Test Model

```python
from src.mrz_ocr.mrz_recognizer import MRZRecognizer

recognizer = MRZRecognizer(model_dir="models/paddleocr_mrz")
text = recognizer.recognize("mrz_image.jpg")
print(f"MRZ Text: {text}")
```

---

### Step 3: Face Detection (YuNet)

No training required! Just download the pre-trained model:

```bash
python scripts/download_yunet.py --output-dir models
```

Or manually:
```bash
wget -O models/face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

---

### Step 4: Copy Models to Production

```bash
# Copy to main project models directory
cp models/*.onnx ../models/
cp -r models/paddleocr_mrz ../models/
```

Or use the automated script:
```bash
python scripts/train_all.py --skip-train-yolox --skip-train-mrz --skip-face
```

---

## Google Colab Training

### Document Detection

1. Open `colab/yolox_document_detection.ipynb`
2. Upload to Google Colab
3. Connect to GPU runtime (Runtime → Change runtime type → GPU)
4. Mount Google Drive
5. Run all cells
6. Download `yolox_idcard.onnx` from Drive

### MRZ OCR

1. Open `colab/paddleocr_mrz_training.ipynb`
2. Upload to Google Colab
3. Connect to GPU runtime
4. Mount Google Drive
5. Run all cells
6. Download `paddleocr_mrz/` folder from Drive

### Face Detection

1. Open `colab/download_yunet_model.ipynb`
2. Run to download model
3. Save to Google Drive

---

## Training Parameters

### YOLOX Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 0.33 | Model depth (YOLOX-S) |
| `width` | 0.50 | Model width |
| `input_size` | 640x640 | Input resolution |
| `max_epoch` | 100 | Training epochs |
| `batch_size` | 16 | Batch size |
| `lr` | 0.01/64 | Learning rate |

### PaddleOCR Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch_num` | 100 | Training epochs |
| `batch_size` | 128 | Batch size |
| `lr` | 0.001 | Learning rate |
| `max_text_length` | 30 | Max MRZ length |

---

## Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce batch size
```bash
python scripts/train_yolox.py --batch-size 8  # Instead of 16
python scripts/train_paddleocr.py --batch-size 64  # Instead of 128
```

### Poor Accuracy

**Document Detection:**
- Increase dataset size: `--num-samples 5000`
- Train longer: `--epochs 150`
- Check data quality

**MRZ OCR:**
- Generate more synthetic data
- Add real MRZ samples
- Fine-tune with custom dictionary

### Slow Training

- Use SSD instead of HDD
- Enable mixed precision (FP16) - already enabled by default
- Use multiple GPUs if available

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall torch torchvision
pip install --force-reinstall opencv-python
```

---

## Expected Results

After successful training:

| Model | Metric | Target | Typical |
|-------|--------|--------|---------|
| YOLOX | mAP@0.5 | > 0.95 | 0.97-0.99 |
| PaddleOCR | Char Acc | > 99% | 99.2-99.8% |
| YuNet | Detection | > 99% | 99.5%+ |

---

## Next Steps

After training:

1. **Validate Models:**
   ```bash
   pytest tests/test_document_detector.py -v
   pytest tests/test_mrz_ocr.py -v
   ```

2. **Test Full Pipeline:**
   ```bash
   python -m pytest tests/ -v
   ```

3. **Start API:**
   ```bash
   cd ..
   uvicorn api.main:app --reload
   ```

4. **Deploy with Docker:**
   ```bash
   docker-compose up -d
   ```
