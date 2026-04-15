#!/usr/bin/env python3
"""
Minimal Flask backend for JavaScript webcam app.
Receives frames from browser, runs YOLOX detection, returns results as JSON.

Usage:
    python webcam_backend.py --model models/yolox_idcard.onnx --conf 0.5

Then open webcam_js_app.html in your browser.
"""

import argparse
import base64
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import cv2
    import numpy as np
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install flask flask-cors opencv-python numpy")
    sys.exit(1)

try:
    from document_detection import DocumentDetector
except ImportError as e:
    print(f"Failed to import DocumentDetector: {e}")
    sys.exit(1)


def create_app(model_path, conf_threshold):
    app = Flask(__name__)
    CORS(app)  # Allow browser JS to call this backend
    
    print(f"Loading model: {model_path}")
    detector = DocumentDetector(
        model_path=model_path,
        confidence_threshold=conf_threshold
    )
    
    @app.route("/detect", methods=["POST"])
    def detect():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        start = time.time()
        results = detector.detect(image)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        
        detections = [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox)
            }
            for d in results
        ]
        
        return jsonify({
            "detections": detections,
            "inference_time_ms": elapsed_ms,
            "count": len(detections)
        })
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": Path(model_path).name})
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Flask backend for JS webcam detection")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    app = create_app(str(model_path), args.conf)
    print(f"\n🚀 Backend running at http://{args.host}:{args.port}")
    print("Open webcam_js_app.html in your browser")
    print("Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
