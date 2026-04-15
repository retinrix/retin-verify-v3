#!/usr/bin/env python3
"""
Simple web-based inference app for YOLOX document detection.
Upload an image and see the detection result in your browser.

Usage:
    python web_inference_app.py
    python web_inference_app.py --model models/yolox_idcard.onnx --port 8080

Then open http://localhost:8080 in your browser.
"""

import argparse
import base64
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import cv2
    import numpy as np
    from flask import Flask, render_template_string, request, jsonify
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install flask opencv-python numpy")
    sys.exit(1)

try:
    from document_detection import DocumentDetector
except ImportError as e:
    print(f"Failed to import DocumentDetector: {e}")
    sys.exit(1)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLOX Document Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-box:hover { border-color: #666; }
        input[type="file"] { display: none; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #0056b3; }
        .result { margin-top: 20px; }
        .result img { max-width: 100%; border: 1px solid #ddd; }
        .metrics { background: #f8f9fa; padding: 15px; margin-top: 15px; border-radius: 5px; }
        .metric { margin: 5px 0; }
        .no-detect { color: #856404; background: #fff3cd; padding: 15px; border-radius: 5px; }
        .error { color: #721c24; background: #f8d7da; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🆔 YOLOX Document Detection</h1>
    <p>Upload an image to detect ID cards.</p>
    
    <form method="post" enctype="multipart/form-data">
        <div class="upload-box">
            <label for="file" class="btn">Choose Image</label>
            <input type="file" id="file" name="image" accept="image/*" onchange="this.form.submit()">
            <p>or drag and drop an image here</p>
        </div>
    </form>
    
    {% if result %}
    <div class="result">
        <h2>Result</h2>
        {% if result.error %}
            <div class="error">{{ result.error }}</div>
        {% else %}
            <img src="data:image/jpeg;base64,{{ result.image }}" alt="Detection result">
            <div class="metrics">
                <div class="metric"><strong>Detections:</strong> {{ result.count }}</div>
                <div class="metric"><strong>Inference time:</strong> {{ result.time_ms }} ms</div>
                {% for det in result.detections %}
                <div class="metric">
                    📦 {{ det.class }} — confidence: {{ det.conf }}<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;bbox: ({{ det.x1 }}, {{ det.y1 }}, {{ det.x2 }}, {{ det.y2 }})
                </div>
                {% endfor %}
                {% if result.count == 0 %}
                <div class="no-detect">No objects detected. Try a different image or lower the confidence threshold.</div>
                {% endif %}
            </div>
        {% endif %}
    </div>
    {% endif %}
    
    <hr>
    <p><small>Model: {{ model_name }} | Conf threshold: {{ conf }}</small></p>
</body>
</html>
"""


def create_app(model_path, conf_threshold):
    app = Flask(__name__)
    
    print(f"Loading model: {model_path}")
    detector = DocumentDetector(
        model_path=model_path,
        confidence_threshold=conf_threshold
    )
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        
        if request.method == "POST":
            file = request.files.get("image")
            if not file:
                result = {"error": "No image uploaded"}
            else:
                try:
                    # Read image from upload
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        result = {"error": "Failed to decode image"}
                    else:
                        import time
                        start = time.time()
                        detections = detector.detect(image)
                        elapsed_ms = round((time.time() - start) * 1000, 2)
                        
                        # Visualize
                        vis_image = detector.visualize(image, detections)
                        
                        # Encode to base64
                        _, buffer = cv2.imencode(".jpg", vis_image)
                        img_b64 = base64.b64encode(buffer).decode("utf-8")
                        
                        result = {
                            "image": img_b64,
                            "count": len(detections),
                            "time_ms": elapsed_ms,
                            "detections": [
                                {
                                    "class": d.class_name,
                                    "conf": f"{d.confidence:.4f}",
                                    "x1": d.bbox[0],
                                    "y1": d.bbox[1],
                                    "x2": d.bbox[2],
                                    "y2": d.bbox[3],
                                }
                                for d in detections
                            ]
                        }
                except Exception as e:
                    result = {"error": str(e)}
        
        return render_template_string(
            HTML_PAGE,
            result=result,
            model_name=Path(model_path).name,
            conf=conf_threshold
        )
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Web inference app for YOLOX document detection")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\nPlease download your trained ONNX model and place it at:")
        print(f"  {model_path.absolute()}")
        print(f"\nOr specify a different path with --model")
        sys.exit(1)
    
    app = create_app(str(model_path), args.conf)
    print(f"\n🚀 Starting web server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
