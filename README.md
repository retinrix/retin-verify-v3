# Retin-Verify V3

Computer Vision-Based Algerian ID Card Verification

## Overview

Retin-Verify V3 is a pure computer vision solution for verifying Algerian ID cards without chip reading. It extracts and validates MRZ (Machine Readable Zone), detects security features, and performs tampering detection.

**License:** MIT (Free for commercial use)

## Features

- ✅ **Document Detection** - Automatic ID card localization
- ✅ **MRZ Extraction & OCR** - Supports Arabic and Latin text
- ✅ **MRZ Validation** - ICAO 9303 check digit verification
- ✅ **Face Detection** - Extract face photo for biometric comparison
- ✅ **Security Feature Analysis** - Hologram, laser photo detection
- ✅ **Tampering Detection** - Error level analysis, copy-move detection

## Technology Stack

All components are open-source with **non-AGPL licenses** (free for commercial use):

| Component | License | Purpose |
|-----------|---------|---------|
| PaddleOCR | Apache-2.0 | OCR (Arabic + Latin support) |
| OpenCV | Apache-2.0 | Computer vision |
| PyTorch | BSD-3 | Deep learning |
| YuNet | Apache-2.0 | Face detection |
| FastAPI | MIT | Web API |

## Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/retinverify/retin-verify-v3.git
cd retin-verify-v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Usage

### Command Line

```python
from src.utils.mrz_utils import parse_mrz

# Parse MRZ from OCR output
mrz_lines = [
    "IDDZA1154521568<<<<<<<<<<<<<<<",
    "6703174M2908236DZA<<<<<<<<<<<8",
    "BOUTIGHANE<<MOHAMED<NAAMAN<<<<"
]

mrz_data = parse_mrz(mrz_lines)
print(f"Name: {mrz_data.surname} {mrz_data.given_names}")
print(f"Document: {mrz_data.document_number}")
print(f"Valid: {mrz_data.valid}")
```

### API

```bash
# Start API server
uvicorn api.main:app --reload

# Send request
curl -X POST "http://localhost:8000/api/v3/verify" \
  -F "image=@id_card.jpg" \
  -F "side=auto"
```

## Project Structure

```
retin-verify-v3/
├── src/                    # Source code
│   ├── utils/             # Utility functions
│   ├── document_detection/# Document detection module
│   ├── mrz_ocr/           # MRZ OCR module
│   ├── face_detection/    # Face detection module
│   ├── security_features/ # Security feature analysis
│   └── tampering_detection/# Tampering detection
├── tests/                 # Unit tests
├── colab/                 # Google Colab notebooks
├── models/                # Trained models
├── data/                  # Dataset
└── api/                   # REST API
```

## Training on Google Colab

- [Document Detection Training](colab/yolox_document_detection.ipynb)
- [MRZ OCR Training](colab/paddleocr_mrz_training.ipynb)

## Implementation Status

### Phase 1: Foundation ✅
- [x] Project setup
- [x] Utility functions (image, MRZ, validation)
- [x] MRZ validation (ICAO 9303)
- [x] Unit tests

### Phase 2: Document Detection ✅
- [x] YOLOX-based document detector
- [x] ONNX Runtime inference
- [x] Colab training notebook
- [x] Visualization tools

### Phase 3: MRZ OCR ✅
- [x] MRZ region detection (heuristic-based)
- [x] Text recognition (PaddleOCR/EasyOCR fallback)
- [x] Arabic + Latin character support
- [x] Complete pipeline integration
- [x] Colab training notebook

### Phase 4: Face Detection ✅
- [x] YuNet face detector (OpenCV DNN)
- [x] Haar Cascade fallback
- [x] 5-point facial landmark detection
- [x] Face alignment and normalization
- [x] Face comparison (histogram/template/ORB)
- [x] Quality checks
- [x] Colab download notebook

### Phase 5: Security & Tampering ✅
- [x] Hologram detection (color variance, iridescence)
- [x] Laser-engraved feature detection
- [x] Print quality analysis
- [x] Error Level Analysis (ELA)
- [x] Copy-move forgery detection
- [x] Complete security analyzer
- [x] Tampering detection pipeline

### Phase 6: API & Deployment ✅
- [x] FastAPI REST API with async support
- [x] WebSocket for real-time updates
- [x] Mobile-first Web UI with camera capture
- [x] PostgreSQL database with SQLAlchemy
- [x] Redis caching
- [x] MinIO/S3 object storage
- [x] Docker containerization
- [x] Docker Compose orchestration
- [x] Nginx reverse proxy
- [x] Health checks and monitoring

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

All dependencies use permissive licenses (Apache-2.0, BSD-3, MIT) and are free for commercial use.

## Contact

For questions or support, please open an issue on GitHub.
