#!/usr/bin/env python3
"""
Interactive inference app for YOLOX document detection.

Usage:
    python inference_app.py --image path/to/image.jpg
    python inference_app.py --image path/to/image.jpg --model models/yolox_idcard.onnx --conf 0.3
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

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from document_detection import DocumentDetector
except ImportError as e:
    print(f"Failed to import DocumentDetector: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="YOLOX Document Detection Inference App")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\nPlease download your trained ONNX model and place it at:")
        print(f"  {model_path.absolute()}")
        print(f"\nOr specify a different path with --model")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("YOLOX Document Detection - Inference")
    print(f"{'='*50}")
    print(f"Model:  {model_path}")
    print(f"Image:  {image_path}")
    print(f"Conf:   {args.conf}")
    print(f"{'='*50}\n")

    detector = DocumentDetector(
        model_path=str(model_path),
        confidence_threshold=args.conf
    )

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    print("Running inference...")
    results = detector.detect(image)

    print(f"\n{'='*50}")
    print(f"RESULTS: {len(results)} detection(s)")
    print(f"{'='*50}")

    for idx, det in enumerate(results, 1):
        print(f"\n  [{idx}] {det.class_name}")
        print(f"      Confidence: {det.confidence:.4f}")
        print(f"      BBox:       x1={det.bbox[0]}, y1={det.bbox[1]}, x2={det.bbox[2]}, y2={det.bbox[3]}")

    if not results:
        print("\n  ⚠️  No objects detected.")
        print("     Tip: Try lowering --conf (e.g., --conf 0.1)")

    # Visualize and save
    vis_image = detector.visualize(image, results)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = image_path.stem
        output_path = image_path.parent / f"{stem}_detected.jpg"

    cv2.imwrite(str(output_path), vis_image)
    print(f"\n✅ Output saved: {output_path}")


if __name__ == "__main__":
    main()
