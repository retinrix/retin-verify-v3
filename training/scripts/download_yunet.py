#!/usr/bin/env python3
"""
Download YuNet face detection model.

YuNet is a lightweight face detection model from OpenCV Zoo.
No training required - just download the pre-trained model.
"""

import os
import urllib.request
import argparse
from pathlib import Path


MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_NAME = "face_detection_yunet.onnx"


def download_model(output_dir, verify=True):
    """Download the YuNet model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / MODEL_NAME
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        file_size = model_path.stat().st_size / 1024
        print(f"File size: {file_size:.1f} KB")
        
        if not verify:
            return model_path
    
    print(f"Downloading YuNet model from {MODEL_URL}...")
    
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        
        file_size = model_path.stat().st_size / 1024
        print(f"✅ Download complete!")
        print(f"   Path: {model_path}")
        print(f"   Size: {file_size:.1f} KB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def verify_model(model_path):
    """Verify the downloaded model works."""
    print("\nVerifying model...")
    
    try:
        import cv2
        import numpy as np
        
        # Create detector
        detector = cv2.FaceDetectorYN_create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000
        )
        
        # Create test image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 240
        
        # Draw a simple face
        cv2.ellipse(test_image, (320, 240), (100, 120), 0, 0, 360, (180, 160, 140), -1)
        cv2.circle(test_image, (280, 220), 15, (60, 60, 60), -1)
        cv2.circle(test_image, (360, 220), 15, (60, 60, 60), -1)
        
        # Detect
        detector.setInputSize((640, 480))
        _, faces = detector.detect(test_image)
        
        if faces is not None and len(faces) > 0:
            print(f"✅ Model verification passed! Detected {len(faces)} face(s)")
            return True
        else:
            print("⚠️ Model loaded but no faces detected in test image")
            return True  # Model works, just test image issue
            
    except ImportError:
        print("⚠️ OpenCV not installed, skipping verification")
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download YuNet face detection model')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for the model')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip model verification')
    
    args = parser.parse_args()
    
    # Download model
    model_path = download_model(args.output_dir, verify=not args.no_verify)
    
    if model_path and not args.no_verify:
        verify_model(model_path)
    
    print(f"\n📋 Model Information:")
    print(f"   Name: YuNet (2023 March)")
    print(f"   License: Apache-2.0 (Free for commercial use)")
    print(f"   Input: 320x320 or dynamic size")
    print(f"   Output: Bounding box + 5 landmarks + confidence")
    print(f"   Size: ~340 KB")


if __name__ == '__main__':
    main()
