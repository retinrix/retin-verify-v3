# Security Features & Tampering Detection Guide

## Overview

This module provides comprehensive security feature analysis and tampering detection for Algerian ID card verification.

**License:** Apache-2.0 (Free for commercial use)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Analysis                         │
├─────────────────────────────────────────────────────────────┤
│  Hologram Detection  │  Laser Detection  │  Print Quality   │
│  - Color variance    │  - Edge analysis  │  - Resolution    │
│  - Iridescence       │  - Texture        │  - Sharpness     │
│  - Pattern type      │  - Relief         │  - Compression   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Tampering Detection                        │
├─────────────────────────────────────────────────────────────┤
│  Error Level Analysis    │    Copy-Move Detection           │
│  - Recompression diff    │    - Block matching              │
│  - Error patterns        │    - Similarity detection        │
│  - Suspicious regions    │    - Cluster analysis            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Hologram Detection

Detects holographic security features:

```python
from security_features import HologramDetector

detector = HologramDetector()
regions = detector.detect(image)

for region in regions:
    print(f"Type: {region.pattern_type.value}")
    print(f"Confidence: {region.confidence:.2f}")
    print(f"Color variance: {region.color_variance:.1f}")
```

**Hologram Types:**
- Dot Matrix
- True Color
- Kinegram
- Guoches

### 2. Laser Detection

Detects laser-engraved features:

```python
from security_features import LaserDetector

detector = LaserDetector()
is_auth, details = detector.verify_photo_authenticity(image, face_bbox)
```

**Feature Types:**
- Photo (laser-engraved portrait)
- Text (microtext)
- Guilloche (patterns)
- Microtext

### 3. Security Analyzer

Complete security analysis:

```python
from security_features import SecurityAnalyzer

analyzer = SecurityAnalyzer()
result = analyzer.analyze(image, face_bbox)

print(f"Authentic: {result.is_authentic}")
print(f"Overall Score: {result.overall_score:.2f}")
print(f"Hologram: {result.hologram_result['authentic']}")
print(f"Laser: {result.laser_result['authentic']}")
```

### 4. ELA (Error Level Analysis)

Detects image manipulation:

```python
from tampering_detection import ELAAnalyzer

analyzer = ELAAnalyzer(quality=95)
result = analyzer.analyze(image)

print(f"Tampered: {result.is_tampered}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Error ratio: {result.error_ratio:.4f}")

# Visualize
vis = analyzer.visualize_result(image, result)
```

### 5. Copy-Move Detection

Detects copy-move forgery:

```python
from tampering_detection import CopyMoveDetector

detector = CopyMoveDetector(
    block_size=8,
    similarity_threshold=0.9
)

result = detector.detect(image)

if result['forgery_detected']:
    print(f"Found {result['num_matches']} matching blocks")
    print(f"Confidence: {result['confidence']:.2f}")
```

### 6. Tampering Pipeline

Complete tampering analysis:

```python
from tampering_detection import TamperingPipeline

pipeline = TamperingPipeline()
result = pipeline.analyze(image)

print(f"Tampered: {result.is_tampered}")
print(f"Confidence: {result.overall_confidence:.2f}")

for issue in result.inconsistencies:
    print(f"Issue: {issue}")

for rec in result.recommendations:
    print(f"Recommendation: {rec}")
```

## Detection Methods

### Error Level Analysis (ELA)

**How it works:**
1. Save image at known JPEG quality
2. Compare with original
3. Analyze error patterns

**Interpretation:**
- Authentic: Uniform error distribution
- Tampered: Localized error differences

```python
# High quality setting = more sensitive
analyzer = ELAAnalyzer(quality=95)

# Lower threshold = more sensitive
analyzer = ELAAnalyzer(error_threshold=20)
```

### Copy-Move Detection

**How it works:**
1. Divide image into blocks
2. Compute features (DCT or pixels)
3. Find similar blocks
4. Cluster matches

**Parameters:**
- `block_size`: 8-16 pixels
- `overlap`: 4-8 pixels
- `similarity_threshold`: 0.85-0.95

## Security Features on Algerian ID Cards

### Hologram Locations

```
┌─────────────────────────────┐
│  ┌───┐                      │
│  │ H │  Hologram            │  <- Top corner
│  └───┘                      │
│                             │
│      ┌─────────┐            │
│      │  FACE   │            │  <- Laser photo
│      └─────────┘            │
│                             │
│  TEXT INFORMATION           │
│                             │
│  ┌─────────────────────┐    │
│  │        MRZ          │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
```

### Typical Security Features

| Feature | Location | Detection Method |
|---------|----------|------------------|
| Hologram | Top corner | Color variance |
| Laser photo | Center | Edge/texture |
| Microtext | Various | High-res scan |
| UV features | Invisible | UV light |
| Guilloche | Background | Pattern matching |

## Usage Examples

### Complete Verification

```python
from document_detection import DocumentDetector
from face_detection import FacePipeline
from mrz_ocr import MRZPipeline
from security_features import SecurityAnalyzer
from tampering_detection import TamperingPipeline

# Load image
image = cv2.imread("id_card.jpg")

# 1. Detect document
doc_detector = DocumentDetector()
doc_result = doc_detector.detect(image)

if not doc_result.success:
    print("Document not detected")
    exit()

# Extract document
doc_image = doc_detector.extract_document(image, doc_result.bbox)

# 2. Extract face
face_pipeline = FacePipeline()
face_result = face_pipeline.extract_face(doc_image)

# 3. Read MRZ
mrz_pipeline = MRZPipeline()
mrz_result = mrz_pipeline.process(doc_image)

# 4. Security analysis
security_analyzer = SecurityAnalyzer()
security_result = security_analyzer.analyze(
    doc_image,
    face_result.face_bbox if face_result.success else None
)

# 5. Tampering detection
tampering_pipeline = TamperingPipeline()
tampering_result = tampering_pipeline.analyze(doc_image)

# Final decision
is_valid = (
    doc_result.success and
    face_result.success and
    mrz_result.success and
    security_result.is_authentic and
    not tampering_result.is_tampered
)

print(f"ID Card Valid: {is_valid}")
```

### Batch Processing

```python
import glob

pipeline = TamperingPipeline()

for image_path in glob.glob("id_cards/*.jpg"):
    result = pipeline.analyze_file(image_path)
    
    status = "TAMPERED" if result.is_tampered else "OK"
    print(f"{image_path}: {status} ({result.overall_confidence:.2f})")
```

## Performance

| Method | Time | Accuracy |
|--------|------|----------|
| Hologram Detection | 50ms | 75% |
| Laser Detection | 100ms | 80% |
| ELA | 200ms | 85% |
| Copy-Move | 500ms | 90% |
| Full Pipeline | 1000ms | - |

## Troubleshooting

### False Positives

**Hologram detection:**
- Adjust `color_variance_threshold`
- Check for reflective surfaces

**ELA:**
- Use appropriate quality setting
- Account for multiple compressions

**Copy-move:**
- Increase `min_distance`
- Adjust `similarity_threshold`

### False Negatives

- Check image quality (resolution, blur)
- Verify correct ROI
- Adjust thresholds

## References

- [ELA Paper](http://blackhat.com/presentations/bh-dc-08/Krawetz/Whitepaper/bh-dc-08-krawetz-WP.pdf)
- [Copy-Move Detection Survey](https://www.sciencedirect.com/science/article/pii/S1742287613001023)
- [ID Card Security Features](https://www.interpol.int/How-we-work/Forensics/ID-documents)
