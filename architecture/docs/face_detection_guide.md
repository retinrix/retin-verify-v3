# Face Detection Module Guide

## Overview

The Face Detection module extracts and aligns face photos from Algerian ID cards for biometric verification.

**License:** Apache-2.0 (Free for commercial use)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Face Detection │────▶│  Face Alignment  │────▶│  Face Comparison│
│  (YuNet/Haar)   │     │  (Landmark-based)│     │  (Histogram/ORB)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Components

1. **FaceDetector** - Detects faces using YuNet DNN or Haar Cascade fallback
2. **FaceAligner** - Aligns faces using 5-point facial landmarks
3. **FacePipeline** - Complete pipeline from ID card to normalized face

## Installation

### Prerequisites

```bash
# OpenCV (includes DNN module)
pip install opencv-python opencv-contrib-python

# Download YuNet model
mkdir -p models
wget -O models/face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Usage

### Basic Face Extraction

```python
from face_detection import FacePipeline

# Initialize pipeline
pipeline = FacePipeline(
    detector_model_path="models/face_detection_yunet.onnx",
    target_size=(112, 112)
)

# Extract face from ID card
result = pipeline.extract_face_from_file("path/to/id_card.jpg")

if result.success:
    # Save extracted face
    cv2.imwrite("extracted_face.jpg", result.face_image)
    
    print(f"Face confidence: {result.confidence:.2f}")
    print(f"Alignment score: {result.alignment.alignment_score:.3f}")
else:
    print(f"Extraction failed: {result.error_message}")
```

### With Document Detection

```python
from document_detection import DocumentDetector
from face_detection import FacePipeline

# Detect document
detector = DocumentDetector()
doc_result = detector.detect(image)

if doc_result.success:
    # Extract face using document bbox
    pipeline = FacePipeline()
    face_result = pipeline.extract_face(
        image,
        document_bbox=doc_result.bbox
    )
```

### Face Comparison

```python
# Compare face from ID with live photo
similarity = pipeline.compare_faces(
    id_face,
    live_photo_face,
    method="histogram"  # or "template", "orb"
)

if similarity > 0.7:
    print("Faces match!")
else:
    print("Faces do not match")
```

## Face Alignment

The aligner uses 5 facial landmarks to normalize face position:

```
Right Eye ─────┐
               │
Left Eye  ─────┤───▶ Aligned to standard template
               │
Nose      ─────┤
               │
Right Mouth ───┤
               │
Left Mouth ────┘
```

### Standard Template (112x112)

```python
from face_detection import FaceAligner

aligner = FaceAligner(target_size=(112, 112))

# Align using landmarks
aligned = aligner.align(image, landmarks)

# Result contains:
# - aligned.image: Normalized face image
# - aligned.transformation_matrix: Affine transform
# - aligned.alignment_score: Quality score (0-1)
```

## Face Detection Models

### YuNet (Primary)

- **Model:** `face_detection_yunet_2023mar.onnx`
- **Input:** 320x320 or original size
- **Output:** Bounding box + 5 landmarks + confidence
- **Speed:** ~10ms on CPU
- **License:** Apache-2.0

### Haar Cascade (Fallback)

- Built into OpenCV
- No model download required
- Less accurate but always available

## ID Card Face Position

For Algerian ID cards, the face is typically positioned:

```
┌─────────────────────────────┐
│                             │
│      ┌─────────────┐        │  ▲
│      │             │        │  │
│      │    FACE     │        │  │ 40-50% of card height
│      │             │        │  │
│      └─────────────┘        │  ▼
│                             │
│      TEXT INFORMATION       │
│                             │
│  ┌───────────────────────┐  │
│  │        MRZ            │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

The pipeline automatically searches the upper portion of the ID card.

## Quality Checks

The pipeline performs automatic quality checks:

| Check | Threshold | Description |
|-------|-----------|-------------|
| Confidence | > 0.7 | Detection confidence |
| Size | > 80px | Minimum face size |
| Aspect Ratio | 0.6-1.0 | Face shape |
| Brightness | 0.3-0.7 | Not too dark/bright |
| Contrast | > 0.3 | Sufficient detail |
| Sharpness | > 0.3 | Not blurry |

## API Reference

### FaceDetector

```python
class FaceDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        input_size: Tuple[int, int] = (320, 320),
    )
    
    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]
    def detect_largest_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]
    def detect_in_region(self, image: np.ndarray, region_bbox: Tuple) -> List[FaceDetectionResult]
```

### FaceAligner

```python
class FaceAligner:
    def __init__(self, target_size: Tuple[int, int] = (112, 112))
    
    def align(self, image: np.ndarray, landmarks: np.ndarray) -> AlignedFace
    def extract_face_chip(self, image: np.ndarray, bbox: Tuple, padding: float = 0.2) -> np.ndarray
    def normalize_face(self, face_image: np.ndarray, method: str = "standard") -> np.ndarray
```

### FacePipeline

```python
class FacePipeline:
    def __init__(
        self,
        detector_model_path: Optional[str] = None,
        target_size: Tuple[int, int] = (112, 112),
        min_confidence: float = 0.7,
        min_face_size: int = 80,
    )
    
    def extract_face(
        self,
        image: np.ndarray,
        document_bbox: Optional[Tuple] = None
    ) -> FaceExtractionResult
    
    def compare_faces(
        self,
        face1: np.ndarray,
        face2: np.ndarray,
        method: str = "histogram"
    ) -> float
```

## Performance

| Metric | Value |
|--------|-------|
| Detection Time | 10-30ms (CPU) |
| Alignment Time | 5ms |
| Output Size | 112x112 pixels |
| Model Size | 340KB (YuNet) |

## Troubleshooting

### Face Not Detected

1. **Check image quality** - Ensure face is clearly visible
2. **Verify face position** - Face should be in upper portion
3. **Adjust confidence threshold** - Lower if needed
4. **Use Haar fallback** - Works without model download

### Poor Alignment

1. **Check landmarks** - Ensure all 5 landmarks detected
2. **Verify face angle** - Extreme angles may fail
3. **Use fallback extraction** - Simple crop without alignment

### Low Comparison Score

1. **Normalize lighting** - Ensure similar conditions
2. **Check alignment** - Both faces should be aligned
3. **Try different method** - histogram/template/orb

## References

- [YuNet Paper](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html)
- [Face Alignment Methods](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html)
