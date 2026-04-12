"""Configuration settings for Retin-Verify V3."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model paths
MODEL_PATHS = {
    "document_detection": MODELS_DIR / "yolox_document.onnx",
    "mrz_detection": MODELS_DIR / "paddleocr_mrz",
    "face_detection": MODELS_DIR / "yunet_face.onnx",
}

# Image processing settings
IMAGE_SETTINGS = {
    "max_size": 1920,  # Max dimension for input images
    "min_size": 640,   # Min dimension for input images
    "target_size": (640, 640),  # Target size for document detection
}

# MRZ settings (ICAO 9303 TD1 format)
MRZ_SETTINGS = {
    "line_length": 30,
    "num_lines": 3,
    "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<",
    "check_digit_weights": [7, 3, 1],
}

# Face detection settings
FACE_SETTINGS = {
    "confidence_threshold": 0.7,
    "min_face_size": 80,  # Minimum face size in pixels
    "padding": 0.2,  # Padding around detected face
}

# Security feature settings
SECURITY_SETTINGS = {
    "hologram_threshold": 0.8,
    "laser_photo_threshold": 0.75,
    "authenticity_threshold": 70,  # Minimum score for authenticity
}

# API settings
API_SETTINGS = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".bmp", ".tiff"},
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
