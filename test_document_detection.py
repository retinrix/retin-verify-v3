#!/usr/bin/env python3
"""
Test script for document detection model.

Usage:
    python test_document_detection.py <image_path>
    python test_document_detection.py --model <model_path> <image_path>
    python test_document_detection.py --camera  # Use webcam
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_detection.detector import DocumentDetector


def test_image(image_path, model_path, output_path=None, conf_threshold=0.5):
    """Test detection on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing Document Detection")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    
    # Check files exist
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        print(f"   Please download the model from Google Drive and place it at:")
        print(f"   {model_path}")
        return None
    
    # Initialize detector
    print("\nInitializing detector...")
    try:
        detector = DocumentDetector(model_path=model_path)
        print("✅ Detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return None
    
    # Load image
    print(f"\nLoading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Detect
    print(f"\nRunning detection...")
    try:
        result = detector.detect(image_rgb, conf_threshold=conf_threshold)
        print("✅ Detection complete")
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Detection Results:")
    print(f"{'='*60}")
    
    if result.confidence > 0:
        print(f"  ✅ Document detected!")
        print(f"  Bounding Box: ({result.bbox[0]:.1f}, {result.bbox[1]:.1f}, {result.bbox[2]:.1f}, {result.bbox[3]:.1f})")
        print(f"  Confidence: {result.confidence:.3f} ({result.confidence*100:.1f}%)")
        print(f"  Inference Time: {result.inference_time*1000:.1f}ms")
        
        # Visualize
        vis_image = detector.visualize(image_rgb, result)
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"  Output saved to: {output_path}")
        else:
            # Generate output path
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_detected{ext}"
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"  Output saved to: {output_path}")
        
        # Display (if available)
        try:
            cv2.imshow("Document Detection", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass  # Headless environment
        
        return result
    else:
        print(f"  ❌ No document detected")
        print(f"  Confidence: {result.confidence:.3f}")
        return None


def test_camera(model_path, conf_threshold=0.5):
    """Test detection on live camera feed."""
    print(f"\n{'='*60}")
    print(f"Testing Document Detection - Camera Mode")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Press 'q' to quit, 's' to save screenshot")
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found: {model_path}")
        return
    
    # Initialize detector
    print("\nInitializing detector...")
    try:
        detector = DocumentDetector(model_path=model_path)
        print("✅ Detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    print("✅ Camera opened")
    print("\nStarting detection loop...")
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        try:
            result = detector.detect(frame_rgb, conf_threshold=conf_threshold)
            
            # Visualize
            if result.confidence > 0:
                vis_frame = detector.visualize(frame_rgb, result)
                
                # Add FPS info
                fps_text = f"Conf: {result.confidence:.2f}"
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                cv2.putText(vis_frame_bgr, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                vis_frame_bgr = frame
        except Exception as e:
            print(f"Detection error: {e}")
            vis_frame_bgr = frame
        
        # Show
        cv2.imshow("Document Detection - Press 'q' to quit", vis_frame_bgr)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(screenshot_path, vis_frame_bgr)
            print(f"Screenshot saved: {screenshot_path}")
            screenshot_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed")


def main():
    parser = argparse.ArgumentParser(description='Test document detection model')
    parser.add_argument('image', nargs='?', help='Path to test image')
    parser.add_argument('--model', type=str, default='models/yolox_idcard.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--camera', action='store_true',
                        help='Use camera for live detection')
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model):
        # Try alternative paths
        alt_paths = [
            'models/yolox_idcard.onnx',
            '../models/yolox_idcard.onnx',
            'yolox_idcard.onnx',
        ]
        for path in alt_paths:
            if os.path.exists(path):
                args.model = path
                break
    
    if args.camera:
        test_camera(args.model, args.conf)
    elif args.image:
        test_image(args.image, args.model, args.output, args.conf)
    else:
        # Test with sample image or show help
        sample_paths = [
            'test_images/id_card_sample.jpg',
            'data/test_image.jpg',
        ]
        
        found = False
        for path in sample_paths:
            if os.path.exists(path):
                test_image(path, args.model, args.output, args.conf)
                found = True
                break
        
        if not found:
            parser.print_help()
            print("\n❌ No test image provided and no sample images found.")
            print("Please provide an image path or use --camera for webcam.")


if __name__ == "__main__":
    main()
