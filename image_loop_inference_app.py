#!/usr/bin/env python3
"""
Loop a static image with live detection overlay as a simulated webcam stream.
Useful for testing when no camera or video is available.

Usage:
    python image_loop_inference_app.py --image path/to/image.jpg
    python image_loop_inference_app.py --image path/to/image.jpg --model models/yolox_idcard.onnx

Then open http://localhost:8080 in your browser.
"""

import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import cv2
    import numpy as np
    from flask import Flask, Response, render_template_string
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
    <title>YOLOX Live Detection (Image Loop)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; text-align: center; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .video-container { margin: 20px auto; border: 3px solid #00ff88; border-radius: 8px; display: inline-block; overflow: hidden; }
        .video-container img { max-width: 100%; height: auto; display: block; }
        .metrics { background: #2a2a2a; padding: 15px; margin-top: 15px; border-radius: 8px; display: inline-block; }
        .metric { margin: 8px 0; font-size: 16px; }
        .status { color: #00ff88; font-weight: bold; }
        .controls { margin: 20px 0; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; cursor: pointer; font-size: 14px; margin: 5px; border-radius: 5px; }
        .btn:hover { background: #0056b3; }
        .btn-capture { background: #28a745; }
        .btn-capture:hover { background: #218838; }
    </style>
</head>
<body>
    <h1>🖼️ YOLOX Live Detection (Image Loop)</h1>
    <p>Continuous inference on a static image</p>
    
    <div class="video-container">
        <img src="/video_feed" alt="Detection stream">
    </div>
    
    <div class="metrics">
        <div class="metric status">● Stream Active</div>
        <div class="metric">Model: <strong>{{ model_name }}</strong></div>
        <div class="metric">Confidence: <strong>{{ conf }}</strong></div>
        <div class="metric">FPS: <strong>~{{ fps }}</strong></div>
        <div class="metric">Frame: <strong id="frame">{{ frame_num }}</strong></div>
    </div>
    
    <div class="controls">
        <button class="btn btn-capture" onclick="captureFrame()">📸 Capture Frame</button>
    </div>
    
    <script>
        function captureFrame() {
            fetch('/capture')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const link = document.createElement('a');
                        link.href = 'data:image/jpeg;base64,' + data.image;
                        link.download = 'capture_' + Date.now() + '.jpg';
                        link.click();
                    } else {
                        alert('Capture failed: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(e => alert('Capture failed: ' + e));
        }
    </script>
</body>
</html>
"""


class ImageLoopDetector:
    def __init__(self, image_path, model_path, conf_threshold):
        self.detector = DocumentDetector(
            model_path=model_path,
            confidence_threshold=conf_threshold
        )
        
        self.base_image = cv2.imread(str(image_path))
        if self.base_image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        
        self.frame_num = 0
        self.fps = 0
        self.frame_time = 0
        self.running = True
    
    def get_processed_frame(self):
        frame = self.base_image.copy()
        self.frame_num += 1
        
        start = time.time()
        results = self.detector.detect(frame)
        inference_time = time.time() - start
        
        vis_frame = self.detector.visualize(frame, results)
        
        cv2.putText(vis_frame, f"Detections: {len(results)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Inference: {inference_time*1000:.1f} ms", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame: {self.frame_num}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        self.frame_time = inference_time
        self.fps = 1.0 / max(inference_time, 0.001)
        
        return vis_frame
    
    def generate_frames(self):
        while self.running:
            frame = self.get_processed_frame()
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.05)
    
    def get_current_frame_jpg(self):
        frame = self.base_image.copy()
        results = self.detector.detect(frame)
        vis_frame = self.detector.visualize(frame, results)
        
        _, buffer = cv2.imencode('.jpg', vis_frame)
        return buffer.tobytes()
    
    def stop(self):
        self.running = False


def create_app(image_path, model_path, conf_threshold):
    app = Flask(__name__)
    detector = ImageLoopDetector(image_path, model_path, conf_threshold)
    
    @app.route("/")
    def index():
        return render_template_string(
            HTML_PAGE,
            model_name=Path(model_path).name,
            conf=conf_threshold,
            fps=round(detector.fps),
            frame_num=detector.frame_num
        )
    
    @app.route("/video_feed")
    def video_feed():
        return Response(
            detector.generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    @app.route("/capture")
    def capture():
        jpg = detector.get_current_frame_jpg()
        if jpg is None:
            return {"success": False, "error": "No frame available"}
        return {
            "success": True,
            "image": base64.b64encode(jpg).decode("utf-8")
        }
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Image loop inference for YOLOX document detection")
    parser.add_argument("--image", "-i", required=True, help="Path to image file")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
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
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("YOLOX Image Loop Detection Stream")
    print(f"{'='*50}")
    print(f"Image:  {image_path}")
    print(f"Model:  {model_path}")
    print(f"Conf:   {args.conf}")
    print(f"{'='*50}")
    
    try:
        app = create_app(str(image_path), str(model_path), args.conf)
        print(f"\n🚀 Open http://{args.host}:{args.port} in your browser")
        print("Press Ctrl+C to stop\n")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
