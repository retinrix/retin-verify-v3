#!/usr/bin/env python3
"""
Real-time video file inference app for YOLOX document detection.
Streams a video file with live detection bounding boxes to your browser.

Usage:
    python video_inference_app.py --video path/to/video.mp4
    python video_inference_app.py --video path/to/video.mp4 --model models/yolox_idcard.onnx --conf 0.3

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
    <title>YOLOX Video Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; text-align: center; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .video-container { margin: 20px auto; border: 3px solid #00ff88; border-radius: 8px; display: inline-block; overflow: hidden; }
        .video-container img { max-width: 100%; height: auto; display: block; }
        .metrics { background: #2a2a2a; padding: 15px; margin-top: 15px; border-radius: 8px; display: inline-block; min-width: 250px; }
        .metric { margin: 8px 0; font-size: 16px; }
        .status { color: #00ff88; font-weight: bold; }
        .controls { margin: 20px 0; }
        .btn { background: #007bff; color: white; padding: 12px 24px; border: none; cursor: pointer; font-size: 14px; margin: 5px; border-radius: 5px; }
        .btn:hover { background: #0056b3; }
        .btn-capture { background: #28a745; }
        .btn-capture:hover { background: #218838; }
        .note { color: #aaa; font-size: 14px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🎬 YOLOX Video Document Detection</h1>
    <p>Real-time detection streaming from video file</p>
    
    <div class="video-container">
        <img src="/video_feed" alt="Detection stream">
    </div>
    
    <div class="metrics">
        <div class="metric status" id="status">● Stream Active</div>
        <div class="metric">Model: <strong>{{ model_name }}</strong></div>
        <div class="metric">Confidence: <strong>{{ conf }}</strong></div>
        <div class="metric">Inference: <strong id="infer_ms">--</strong> ms</div>
        <div class="metric">Stream FPS: <strong id="fps">--</strong></div>
        <div class="metric">Frame: <strong id="frame">--</strong></div>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="toggleMirror()">↔️ Toggle Mirror</button>
        <button class="btn btn-capture" onclick="captureFrame()">📸 Capture Frame</button>
    </div>
    
    <p class="note">Inference runs asynchronously; displayed bboxes are always aligned<br>
    with the frame they were computed on.</p>
    
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
        
        setInterval(() => {
            fetch('/metrics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('infer_ms').textContent = data.infer_ms.toFixed(1);
                    document.getElementById('frame').textContent = data.frame_num;
                })
                .catch(() => {});
        }, 500);
    </script>
</body>
</html>
"""


class VideoDetector:
    def __init__(self, video_path, model_path, conf_threshold, loop=True, infer_every_n=2):
        self.detector = DocumentDetector(
            model_path=model_path,
            confidence_threshold=conf_threshold
        )
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self.video_path = str(video_path)
        self.loop = loop
        self.infer_every_n = max(1, infer_every_n)
        
        # Reduce internal buffer for lower latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_lock = threading.Lock()
        self.display_lock = threading.Lock()
        
        # Aligned display frame + results
        self.display_frame = None
        self.display_frame_num = 0
        self.display_infer_ms = 0.0
        self.display_timestamp = 0.0
        
        # Raw reader state
        self.latest_frame = None
        self.frame_num = 0
        
        self.fps = 0.0
        self.running = True
        self._last_stream_time = time.time()
        
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()
    
    def _read_loop(self):
        """Continuously read frames from video file."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_num = 0
                    continue
                else:
                    break
            
            with self.frame_lock:
                self.latest_frame = frame
                self.frame_num += 1
    
    def _infer_loop(self):
        """Run inference asynchronously. Update display only with the inferred frame."""
        local_count = 0
        while self.running:
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                current_count = self.frame_num
                frame = self.latest_frame.copy()
            
            if current_count - local_count < self.infer_every_n:
                time.sleep(0.002)
                continue
            local_count = current_count
            
            start = time.time()
            results = self.detector.detect(frame)
            infer_ms = (time.time() - start) * 1000
            
            vis_frame = self._draw_frame(frame, current_count, results, infer_ms)
            
            with self.display_lock:
                self.display_frame = vis_frame
                self.display_frame_num = current_count
                self.display_infer_ms = infer_ms
                self.display_timestamp = time.time()
    
    def _draw_frame(self, frame, frame_num, results, infer_ms):
        """Draw detections and overlay on a specific frame."""
        vis_frame = self.detector.visualize(frame, results)
        
        h, w = vis_frame.shape[:2]
        cv2.rectangle(vis_frame, (0, 0), (280, 110), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"Detections: {len(results)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Inference: {infer_ms:.1f} ms", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Frame: {frame_num}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def get_processed_frame(self):
        """Get the latest fully-processed frame (frame + bboxes are aligned)."""
        with self.display_lock:
            if self.display_frame is None:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        return self._draw_frame(self.latest_frame.copy(), self.frame_num, [], 0.0)
                return None
            
            # If inference stalled for > 500ms, show current raw frame without old bboxes
            stale = (time.time() - self.display_timestamp) > 0.5
            if stale:
                with self.frame_lock:
                    if self.latest_frame is not None:
                        return self._draw_frame(self.latest_frame.copy(), self.frame_num, [], 0.0)
            
            return self.display_frame.copy()
    
    def generate_frames(self):
        target_interval = 1.0 / 25
        last_frame_time = 0
        
        while self.running:
            frame = self.get_processed_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
                now = time.time()
            last_frame_time = now
            
            self.fps = 1.0 / max(elapsed, target_interval)
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n' + frame_bytes + b'\r\n')
    
    def get_current_frame_jpg(self):
        frame = self.get_processed_frame()
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
    
    def get_metrics(self):
        with self.display_lock:
            return {
                "fps": self.fps,
                "infer_ms": self.display_infer_ms,
                "frame_num": self.display_frame_num,
            }
    
    def stop(self):
        self.running = False
        self.reader_thread.join(timeout=1)
        self.infer_thread.join(timeout=1)
        self.cap.release()


def create_app(video_path, model_path, conf_threshold, loop=True, skip=2):
    app = Flask(__name__)
    detector = VideoDetector(video_path, model_path, conf_threshold, loop, infer_every_n=skip)
    
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
            detector.generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
            }
        )
    
    @app.route("/metrics")
    def metrics():
        return detector.get_metrics()
    
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
    parser = argparse.ArgumentParser(description="Video file inference for YOLOX document detection")
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--no-loop", action="store_true", help="Don't loop the video")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--skip", type=int, default=2, help="Run inference every N frames (default: 2)")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    model_path = Path(args.model)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\nPlease download your trained ONNX model and place it at:")
        print(f"  {model_path.absolute()}")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("YOLOX Video Detection Stream")
    print(f"{'='*50}")
    print(f"Video:  {video_path}")
    print(f"Model:  {model_path}")
    print(f"Conf:   {args.conf}")
    print(f"Loop:   {not args.no_loop}")
    print(f"Skip:   {args.skip} frames")
    print(f"{'='*50}")
    
    try:
        app = create_app(str(video_path), str(model_path), args.conf, loop=not args.no_loop, skip=args.skip)
        print(f"\n🚀 Open http://{args.host}:{args.port} in your browser")
        print("Press Ctrl+C to stop\n")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
