# Retin-Verify V3 Model Training Guide

This directory contains everything needed to train the V3 models locally or on Google Colab.

## Training Pipeline Overview

```
Phase 1: Document Detection (YOLOX)     → models/yolox_idcard.onnx
Phase 2: MRZ OCR (PaddleOCR)            → models/paddleocr_mrz/
Phase 3: Face Detection (YuNet)         → models/face_detection_yunet.onnx (pre-trained)
Phase 4: Security Features (Custom)     → models/security_features/
```

## Quick Start

### Option 1: Local Training (GPU Required)
```bash
cd training
python scripts/train_all.py
```

### Option 2: Google Colab (Recommended)
Open the notebooks in `colab/` directory:
1. `yolox_document_detection.ipynb` - Document detection training
2. `paddleocr_mrz_training.ipynb` - MRZ OCR training
3. `download_yunet_model.ipynb` - Face detection model download

## Directory Structure

```
training/
├── data/                   # Training datasets
│   ├── document_detection/ # ID card images for YOLOX
│   ├── mrz_ocr/           # MRZ images for PaddleOCR
│   └── security_features/ # Security feature samples
├── models/                # Trained model outputs
├── scripts/               # Training scripts
├── configs/               # Model configurations
└── README.md             # This file
```

## Training Steps

### Step 1: Document Detection (YOLOX)

**Requirements:**
- GPU with 8GB+ VRAM
- 1000+ annotated ID card images

**Data Format:** COCO format
```
data/document_detection/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

**Run Training:**
```bash
python scripts/train_yolox.py \
    --data-dir data/document_detection \
    --epochs 100 \
    --batch-size 16
```

**Expected Output:**
- `models/yolox_idcard.onnx` (ONNX format for inference)
- mAP@0.5 > 0.95

### Step 2: MRZ OCR (PaddleOCR)

**Requirements:**
- GPU with 4GB+ VRAM
- 5000+ MRZ images (synthetic + real)

**Data Format:** PaddleOCR format
```
data/mrz_ocr/
├── train/
│   ├── images/
│   └── label.txt
└── val/
    ├── images/
│   └── label.txt
```

Label format: `image_path\tlabel\n`

**Generate Synthetic Data:**
```bash
python scripts/generate_synthetic_mrz.py \
    --output-dir data/mrz_ocr/train \
    --num-samples 5000
```

**Run Training:**
```bash
python scripts/train_paddleocr.py \
    --data-dir data/mrz_ocr \
    --epochs 100 \
    --batch-size 128
```

**Expected Output:**
- `models/paddleocr_mrz/` (Paddle inference model)
- `models/paddleocr_mrz.onnx` (ONNX format)
- Character accuracy > 99%

### Step 3: Face Detection (YuNet)

**No training required!** Download pre-trained model:
```bash
python scripts/download_yunet.py
```

Or use the Colab notebook to download and save to Google Drive.

### Step 4: Security Features

**Requirements:**
- 500+ authentic ID card images
- 200+ fake/forged samples

**Run Training:**
```bash
python scripts/train_security_features.py \
    --authentic-dir data/security_features/authentic \
    --fake-dir data/security_features/fake \
    --epochs 50
```

## Model Validation

After training, validate all models:
```bash
python scripts/validate_models.py
```

This will:
1. Test document detection on sample images
2. Test MRZ OCR accuracy
3. Test face detection
4. Test security feature detection
5. Generate validation report

## Export to Production

Once validated, copy models to the main project:
```bash
python scripts/export_to_production.py
```

This copies:
- `models/yolox_idcard.onnx` → `../models/`
- `models/paddleocr_mrz/` → `../models/`
- `models/face_detection_yunet.onnx` → `../models/`

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use smaller model variant (YOLOX-S instead of YOLOX-M)
- Enable gradient checkpointing

### Poor Accuracy
- Increase dataset size
- Check data quality and annotations
- Adjust learning rate
- Increase training epochs

### CUDA Errors
- Check CUDA version compatibility
- Update GPU drivers
- Use Docker container with correct CUDA version
