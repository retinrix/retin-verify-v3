# Retin-Verify V3 Specification
## Computer Vision-Based Algerian ID Card Verification

---

## 1. Executive Summary

**Project Status:** V2 suspended pending response from Algerian Ministry of Interior  
**New Approach:** Pure computer vision-based ID card verification without chip reading  
**Rationale:** The V2 hardware approach revealed that Algerian ID cards have inconsistent chip data. A CV-based approach provides immediate value for document verification.

**Key Requirements:**
- ✅ Open-source components with **non-AGPL licenses** (free commercial use)
- ✅ **Arabic + Latin** OCR support
- ✅ Step-by-step validation through testing
- ✅ Training on **Google Colab** when required

---

## 2. Objectives

### Primary Objectives
1. **Automatic MRZ Extraction** from ID card images with high accuracy
2. **MRZ Validation** using ICAO 9303 check digit algorithms
3. **Document Authenticity Detection** via security feature analysis
4. **Face Extraction** for biometric comparison
5. **Tampering Detection** to identify forged documents

### Secondary Objectives
1. Support for both scanned and photographed ID cards
2. Real-time processing capability
3. Web API for integration
4. Mobile-friendly processing pipeline

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Camera  │  │  Scanner │  │  Upload  │  │  Batch   │   │
│  │  Capture │  │   Input  │  │   File   │  │  Process │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                PREPROCESSING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Document    │  │   Image      │  │   Quality    │      │
│  │  Detection   │  │  Enhancement │  │   Check      │      │
│  │  (YOLOv8)    │  │ (OpenCV)     │  │ (Custom)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              COMPUTER VISION CORE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   MRZ        │  │  Security    │  │    Face      │      │
│  │  Detection   │  │   Feature    │  │  Detection   │      │
│  │  & OCR       │  │   Analysis   │  │  & Extraction│      │
│  │ (PaddleOCR)  │  │ (Custom CNN) │  │  (YOLOv8-Face)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              VALIDATION & OUTPUT LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  MRZ Check   │  │  Tampering   │  │   Report     │      │
│  │  Digit Verify│  │   Detection  │  │  Generation  │      │
│  │ (ICAO 9303)  │  │ (ELA/Noise)  │  │  (PDF/JSON)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Component Selection (Open Source, Non-AGPL)

### 4.1 Document Detection Module

| Component | Choice | License | Commercial Use |
|-----------|--------|---------|----------------|
| **Framework** | **YOLOv8 (Ultralytics)** | AGPL-3.0 | ❌ |
| **Alternative** | **YOLOv5 (Ultralytics)** | GPL-3.0 | ❌ |
| **Selected** | **YOLOX (Megvii)** | Apache-2.0 | ✅ YES |
| **Alternative** | **YOLO-NAS (Deci)** | Apache-2.0 | ✅ YES |
| **Fallback** | **OpenCV DNN + MobileNet** | Apache-2.0 | ✅ YES |

**Decision:** Use **YOLOX** (Apache-2.0) or train custom MobileNet-SSD with OpenCV DNN

**Validation Test:**
```python
# Test document detection on sample images
# Expected: >95% accuracy on Algerian ID cards
# Metric: IoU > 0.8 for document boundary
```

---

### 4.2 MRZ Detection & OCR Module

| Component | Choice | License | Arabic | Latin | Commercial |
|-----------|--------|---------|--------|-------|------------|
| Tesseract | Apache-2.0 | ✅ | ⚠️ Limited | ✅ | ✅ |
| EasyOCR | Apache-2.0 | ✅ | ✅ | ✅ | ✅ |
| **PaddleOCR** | **Apache-2.0** | ✅ | ✅ | ✅ | ✅ |
| Kraken | GPL-3.0 | ❌ | ✅ | ✅ | ❌ |
| docTR | Apache-2.0 | ✅ | ⚠️ Limited | ✅ | ✅ |

**Decision:** Use **PaddleOCR** (Apache-2.0) - best multilingual support

**Validation Test:**
```python
# Test MRZ OCR on synthetic and real MRZ images
# Expected: >99% character accuracy
# Languages: Arabic (ID front) + Latin (MRZ)
```

**Training on Colab:**
- Notebook: `colab/mrz_ocr_training.ipynb`
- Dataset: Synthetic MRZ generation + real samples
- Export: PaddleOCR model → ONNX format

---

### 4.3 Face Detection Module

| Component | Choice | License | Commercial Use |
|-----------|--------|---------|----------------|
| MTCNN | MIT | ✅ | ✅ YES |
| RetinaFace | MIT | ✅ | ✅ YES |
| **YOLOv8-Face** | **AGPL-3.0** | ❌ | ❌ |
| **Selected** | **YuNet (OpenCV)** | **Apache-2.0** | ✅ YES |
| MediaPipe | Apache-2.0 | ✅ | ✅ YES |

**Decision:** Use **YuNet** (OpenCV DNN) - lightweight, Apache-2.0

**Validation Test:**
```python
# Test face detection on ID card photos
# Expected: >99% detection rate
# Metric: Face quality score > 80
```

---

### 4.4 Security Feature Analysis

| Component | Choice | License | Commercial Use |
|-----------|--------|---------|----------------|
| Custom CNN | PyTorch | BSD-3 | ✅ YES |
| OpenCV Filters | Apache-2.0 | ✅ | ✅ YES |
| scikit-image | BSD-3 | ✅ | ✅ YES |

**Decision:** Custom implementation with OpenCV + scikit-image

**Validation Test:**
```python
# Test hologram detection, laser photo detection
# Expected: >90% feature detection accuracy
```

---

### 4.5 Tampering Detection

| Technique | Implementation | License |
|-----------|----------------|---------|
| Error Level Analysis (ELA) | Custom Python | MIT |
| Noise Analysis | OpenCV + NumPy | BSD/Apache |
| Copy-Move Detection | Custom CNN | MIT |

**Decision:** Custom implementation

---

## 5. Final Technology Stack

### Core Framework
```python
# requirements.txt

# Computer Vision (Apache-2.0)
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0

# Deep Learning (BSD-3 / Apache-2.0)
torch>=2.0.0          # BSD-3
torchvision>=0.15.0   # BSD-3
onnx>=1.14.0          # Apache-2.0
onnxruntime>=1.15.0   # MIT

# OCR (Apache-2.0)
paddlepaddle>=2.5.0   # Apache-2.0
paddleocr>=2.7.0      # Apache-2.0

# Image Processing (BSD)
pillow>=10.0.0        # HPND (similar to BSD)
scikit-image>=0.21.0  # BSD-3

# Numerical (BSD)
numpy>=1.24.0         # BSD-3

# Utilities (MIT/Apache)
pyyaml>=6.0           # MIT
requests>=2.31.0      # Apache-2.0
fastapi>=0.100.0      # MIT
uvicorn>=0.23.0       # BSD-3

# Testing (MIT)
pytest>=7.4.0         # MIT
pytest-cov>=4.1.0     # MIT
```

### License Summary
| Component | License | Commercial |
|-----------|---------|------------|
| OpenCV | Apache-2.0 | ✅ |
| PyTorch | BSD-3 | ✅ |
| PaddleOCR | Apache-2.0 | ✅ |
| YuNet | Apache-2.0 | ✅ |
| YOLOX | Apache-2.0 | ✅ |
| NumPy | BSD-3 | ✅ |
| FastAPI | MIT | ✅ |

**All components are FREE for commercial use!**

---

## 6. Implementation Plan with Validation

### Phase 1: Environment & Document Detection (Week 1)

**Step 1.1: Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python -c "import cv2; import torch; import paddle; print('OK')"
```

**Step 1.2: Document Detection with YOLOX**
```python
# Test on sample images
# Validate: IoU > 0.8 on 10+ test images
```

**Colab Training:**
- Notebook: `colab/yolox_document_detection.ipynb`
- Dataset: COCO format with Algerian ID annotations
- Export: ONNX for inference

---

### Phase 2: MRZ Detection & OCR (Week 2)

**Step 2.1: MRZ Region Detection**
```python
# Use PaddleOCR text detection
# Validate: Text box IoU > 0.9 on MRZ region
```

**Step 2.2: MRZ OCR**
```python
# PaddleOCR recognition with custom model
# Validate: Character accuracy > 99%
```

**Colab Training:**
- Notebook: `colab/paddleocr_mrz_training.ipynb`
- Dataset: Synthetic MRZ (OCR-B font) + real samples
- Languages: Latin (MRZ) + Arabic (front text)

---

### Phase 3: Face Detection & Security Features (Week 3)

**Step 3.1: Face Detection with YuNet**
```python
# OpenCV DNN face detection
# Validate: Detection rate > 99%
```

**Step 3.2: Security Feature Detection**
```python
# Custom CNN for hologram/laser detection
# Validate: Feature detection > 90%
```

---

### Phase 4: Integration & API (Week 4)

**Step 4.1: Pipeline Integration**
```python
# End-to-end processing
# Validate: Full pipeline < 500ms per image
```

**Step 4.2: API Development**
```python
# FastAPI endpoints
# Validate: API response < 1s
```

---

## 7. Testing Strategy

### Unit Tests (Each Module)
```python
# tests/test_document_detection.py
def test_document_detection_iou():
    image = load_test_image("id_card_001.jpg")
    bbox = detect_document(image)
    assert iou(bbox, ground_truth) > 0.8

# tests/test_mrz_ocr.py
def test_mrz_character_accuracy():
    image = load_test_image("mrz_001.jpg")
    text = extract_mrz(image)
    assert character_accuracy(text, ground_truth) > 0.99
```

### Integration Tests
```python
# tests/test_pipeline.py
def test_full_pipeline():
    result = verify_id_card("test_image.jpg")
    assert result["mrz"]["valid"] == True
    assert result["face"]["detected"] == True
```

---

## 8. Deliverables

1. **Source Code:** GitHub repository with MIT license
2. **Trained Models:** ONNX format (framework-agnostic)
3. **Colab Notebooks:** Training pipelines
4. **API Server:** FastAPI with OpenAPI docs
5. **Docker Image:** Ready-to-deploy container
6. **Documentation:** Technical docs + API reference

---

## 9. License

**Project License:** MIT License (Free for commercial use)

**Component Licenses:**
- All dependencies: Apache-2.0, BSD-3, or MIT
- No AGPL, GPL, or copyleft restrictions

---

**Document Version:** 2.0  
**Date:** 2026-04-12  
**Status:** Ready for Implementation
