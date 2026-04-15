#!/usr/bin/env python3
"""
Simple inference app to test YOLOX document detection.

Usage:
    python test_inference_app.py --image path/to/image.jpg
    python test_inference_app.py --image path/to/image.jpg --model models/yolox_idcard.onnx
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from document_detection import DocumentDetector
except ImportError as e:
    print(f"Failed to import DocumentDetector: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test YOLOX document detection inference")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", "-o", default=None, help="Output image path (optional)")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\nPlease place your ONNX model at: {model_path.absolute()}")
        print("Or specify a different path with --model")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    detector = DocumentDetector(
        model_path=str(model_path),
        confidence_threshold=args.conf
    )

    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    print("Running inference...")
    results = detector.detect(image)

    print(f"\n{'='*50}")
    print(f"DETECTION RESULTS: {len(results)} object(s) found")
    print(f"{'='*50}")

    for i, det in enumerate(results, 1):
        print(f"\n  Detection #{i}")
        print(f"    Class: {det.class_name} (ID: {det.class_id})")
        print(f"    BBox:  ({det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]})")
        print(f"    Confidence: {det.confidence:.4f}")

    if not results:
        print("\n  ⚠️  No detections found.")
        print(f"     Try lowering confidence threshold with --conf 0.1")

    # Visualize
    vis_image = detector.visualize(image, results)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = image_path.stem
        output_path = image_path.parent / f"{stem}_detected.jpg"

    cv2.imwrite(str(output_path), vis_image)
    print(f"\n✅ Result saved to: {output_path}")


if __name__ == "__main__":
    main()
