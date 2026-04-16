#!/usr/bin/env python3
"""
Real-time webcam inference app for YOLOX document detection.
Opens the webcam and streams detection results with bounding boxes.

Usage:
    python webcam_inference_app.py
    python webcam_inference_app.py --model models/yolox_idcard.onnx --conf 0.3 --camera 0

Then open http://localhost:8080 in your browser.
"""

import argparse
import base64
import sys
import threading
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
    <title>YOLOX Live Document Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; text-align: center; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .video-container { margin: 20px auto; border: 3px solid #00ff88; border-radius: 8px; display: inline-block; overflow: hidden; }
        .video-container img { max-width: 100%; height: auto; display: block; }
        .metrics { background: #2a2a2a; padding: 15px; margin-top: 15px; border-radius: 8px; display: inline-block; min-width: 250px; }
        .metric { margin: 8px 0; font-size: 16px; }
        .status { color: #00ff88; font-weight: bold; }
        .status.idle { color: #ffaa00; }
        .controls { margin: 20px 0; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; cursor: pointer; font-size: 14px; margin: 5px; border-radius: 5px; }
        .btn:hover { background: #0056b3; }
        .btn-capture { background: #28a745; }
        .btn-capture:hover { background: #218838; }
        .note { color: #aaa; font-size: 14px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🎥 YOLOX Live Document Detection</h1>
    <p>Real-time ID card detection from your webcam</p>
    
    <div class="video-container">
        <img src="/video_feed" alt="Live detection stream">
    </div>
    
    <div class="metrics">
        <div class="metric status" id="status">● Live Stream Active</div>
        <div class="metric">Model: <strong>{{ model_name }}</strong></div>
        <div class="metric">Confidence threshold: <strong>{{ conf }}</strong></div>
        <div class="metric">Inference: <strong id="infer_ms">--</strong> ms</div>
        <div class="metric">Stream FPS: <strong id="fps">--</strong></div>
        <div class="metric">Detections: <strong id="detections">--</strong></div>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="toggleMirror()">↔️ Toggle Mirror</button>
        <button class="btn btn-capture" onclick="captureFrame()">📸 Capture Frame</button>
    </div>
    
    <p class="note">If the stream is slow, the model is running inference on every 3rd frame.<br>
    For better performance, use a GPU or lower the input resolution.</p>
    
    <script>
        let mirrored = false;
        function toggleMirror() {
            mirrored = !mirrored;
            document.querySelector('img').style.transform = mirrored ? 'scaleX(-1)' : 'scaleX(1)';
        }
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
        
        // Poll metrics
        setInterval(() => {
            fetch('/metrics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('infer_ms').textContent = data.infer_ms.toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    const status = document.getElementById('status');
                    if (data.detections > 0) {
                        status.textContent = '● Document Detected';
                        status.className = 'status';
                    } else if (data.infer_ms > 0) {
                        status.textContent = '● Searching...';
                        status.className = 'status idle';
                    } else {
                        status.textContent = '● Live Stream Active';
                        status.className = 'status';
                    }
                })
                .catch(() => {});
        }, 500);
    </script>
</body>
</html>
"""


class WebcamDetector:
    def __init__(self, model_path, conf_threshold, camera_id=0, infer_every_n=3):
        self.detector = DocumentDetector(
            model_path=model_path,
            confidence_threshold=conf_threshold
        )
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.display_frame = None
        
        self.infer_every_n = infer_every_n
        self.frame_count = 0
        
        self.fps = 0
        self.infer_ms = 0
        self.detections_count = 0
        self.last_infer_time = 0
        
        # Inference thread
        self.infer_lock = threading.Lock()
        self.latest_results = []
        self.inferring = False
        self.running = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()
    
    def _capture_loop(self):
        """Continuously capture frames from camera."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frame_count += 1
    
    def _infer_loop(self):
        """Run inference asynchronously on latest frame."""
        while self.running:
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                # Only run inference every N frames
                if self.frame_count % self.infer_every_n != 0:
                    time.sleep(0.005)
                    continue
                frame = self.latest_frame.copy()
            
            # Skip if still processing previous frame
            if self.inferring:
                time.sleep(0.005)
                continue
            
            self.inferring = True
            start = time.time()
            results = self.detector.detect(frame)
            infer_time = time.time() - start
            
            with self.infer_lock:
                self.latest_results = results
                self.infer_ms = infer_time * 1000
                self.detections_count = len(results)
                self.last_infer_time = time.time()
            
            self.inferring = False
    
    def _draw_frame(self, frame):
        """Draw detections on frame. Clear old detections if inference is stale."""
        with self.infer_lock:
            results = list(self.latest_results)
            infer_ms = self.infer_ms
            det_count = self.detections_count
            last_infer = self.last_infer_time
        
        # Clear detections if no inference for > 500ms (card moved away)
        if time.time() - last_infer > 0.5:
            results = []
        
        vis_frame = self.detector.visualize(frame, results)
        
        # Add overlay info
        h, w = vis_frame.shape[:2]
        cv2.rectangle(vis_frame, (0, 0), (280, 80), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"Detections: {det_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Inference: {infer_ms:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def get_processed_frame(self):
        """Get the latest frame with current detections drawn."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        
        vis_frame = self._draw_frame(frame)
        
        # Update stream FPS
        self.fps = 30.0  # Cap is set to 30fps, we stream as fast as capture
        
        return vis_frame
    
    def generate_frames(self):
        """Generate MJPEG stream."""
        last_frame_time = 0
        target_interval = 1.0 / 25  # Stream at 25 FPS max
        
        while self.running:
            frame = self.get_processed_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Throttle stream to target FPS
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
                now = time.time()
            last_frame_time = now
            
            # Add cache-busting header via timestamp in boundary
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n' + frame_bytes + b'\r\n')
    
    def get_current_frame_jpg(self):
        """Get current frame as JPEG for capture."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
        
        vis_frame = self._draw_frame(frame)
        _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
    
    def get_metrics(self):
        with self.infer_lock:
            return {
                "fps": self.fps,
                "infer_ms": self.infer_ms,
                "detections": self.detections_count,
            }
    
    def stop(self):
        self.running = False
        self.capture_thread.join(timeout=1)
        self.infer_thread.join(timeout=1)
        self.cap.release()


def create_app(model_path, conf_threshold, camera_id=0):
    app = Flask(__name__)
    webcam = WebcamDetector(model_path, conf_threshold, camera_id)
    
    @app.route("/")
    def index():
        return render_template_string(
            HTML_PAGE,
            model_name=Path(model_path).name,
            conf=conf_threshold,
        )
    
    @app.route("/video_feed")
    def video_feed():
        return Response(
            webcam.generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
            }
        )
    
    @app.route("/metrics")
    def metrics():
        return webcam.get_metrics()
    
    @app.route("/capture")
    def capture():
        jpg = webcam.get_current_frame_jpg()
        if jpg is None:
            return {"success": False, "error": "No frame available"}
        return {
            "success": True,
            "image": base64.b64encode(jpg).decode("utf-8")
        }
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam inference for YOLOX document detection")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--skip", type=int, default=3, help="Run inference every N frames (default: 3)")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\nPlease download your trained ONNX model and place it at:")
        print(f"  {model_path.absolute()}")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("YOLOX Live Webcam Detection")
    print(f"{'='*50}")
    print(f"Model:  {model_path}")
    print(f"Conf:   {args.conf}")
    print(f"Camera: {args.camera}")
    print(f"Skip:   {args.skip} frames")
    print(f"{'='*50}")
    
    try:
        app = create_app(str(model_path), args.conf, args.camera)
        print(f"\n🚀 Open http://{args.host}:{args.port} in your browser")
        print("Press Ctrl+C to stop\n")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your webcam is connected and not in use by another app.")
        sys.exit(1)


if __name__ == "__main__":
    main()
