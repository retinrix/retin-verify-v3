#!/usr/bin/env python3
"""
Combined Flask server that serves both the HTML frontend and the detection backend.
Open http://localhost:5000 in your browser — no separate file server needed.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import cv2
    import numpy as np
    from flask import Flask, request, jsonify, render_template_string
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


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOX Webcam Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 { color: #00ff88; margin-bottom: 10px; font-size: 1.8rem; }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 0.95rem; }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            width: 100%;
            max-width: 1400px;
        }
        .panel {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 15px;
            border: 1px solid #333;
        }
        .video-panel {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 320px;
        }
        .video-wrapper {
            position: relative;
            width: 640px;
            height: 480px;
            max-width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }
        #webcam {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: cover;
            background: #111;
        }
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 10;
            pointer-events: none;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #444; cursor: not-allowed; }
        .btn-green { background: #28a745; }
        .btn-green:hover { background: #218838; }
        .btn-red { background: #dc3545; }
        .btn-red:hover { background: #c82333; }
        .metrics {
            min-width: 280px;
            max-width: 350px;
        }
        .metric {
            background: #252525;
            padding: 12px 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
        }
        .metric-label { color: #aaa; }
        .metric-value { color: #00ff88; font-weight: 600; }
        .detections-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .detection-item {
            background: #252525;
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            border-left: 3px solid #00ff88;
        }
        .detection-item .conf { color: #ffd700; font-size: 0.9rem; }
        .detection-item .bbox { color: #888; font-size: 0.85rem; margin-top: 4px; }
        .no-detections {
            color: #888;
            text-align: center;
            padding: 30px;
            font-style: italic;
        }
        .status-bar {
            margin-top: 15px;
            padding: 10px 20px;
            border-radius: 8px;
            background: #252525;
            text-align: center;
            min-width: 300px;
        }
        .status-active { color: #00ff88; }
        .status-inactive { color: #ff4444; }
        .hidden { display: none !important; }
        .settings {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #333;
            min-width: 300px;
        }
        .settings label {
            display: block;
            margin-bottom: 10px;
            color: #aaa;
            font-size: 0.9rem;
        }
        .settings input[type="range"] {
            width: 100%;
            margin-top: 5px;
        }
        .settings select {
            width: 100%;
            padding: 8px;
            background: #252525;
            color: #fff;
            border: 1px solid #444;
            border-radius: 6px;
            margin-top: 5px;
        }
        .mirror { transform: scaleX(-1); }
        @media (max-width: 768px) {
            .video-wrapper { width: 100%; height: auto; aspect-ratio: 4/3; }
        }
    </style>
</head>
<body>
    <h1>🎥 YOLOX Live Document Detection</h1>
    <p class="subtitle">JavaScript Webcam + Real-time YOLOX Detection</p>
    
    <div class="container">
        <div class="panel video-panel">
            <div class="video-wrapper">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="overlay"></canvas>
            </div>
            
            <div class="controls">
                <button id="btnStart" class="btn-green">▶ Start Camera</button>
                <button id="btnStop" class="btn-red" disabled>⏹ Stop</button>
                <button id="btnCapture" disabled>📸 Capture</button>
                <button id="btnBest" disabled>⭐ Save Best</button>
                <button id="btnMirror">↔️ Mirror</button>
            </div>
            
            <div class="status-bar">
                Status: <span id="statusText" class="status-inactive">● Camera off</span>
                <span id="fpsText" style="margin-left: 20px; color: #888;"></span>
              </div>
            
            <div class="settings">
                <label>
                    Confidence Threshold: <span id="confValue">0.10</span>
                    <input type="range" id="confThreshold" min="0.05" max="0.9" step="0.05" value="0.1">
                </label>
                <label>
                    Camera:
                    <select id="cameraSelect">
                        <option value="">Default camera</option>
                    </select>
                </label>
            </div>
        </div>
        
        <div class="panel metrics">
            <h3 style="margin-bottom: 15px; color: #00ff88;">📊 Metrics</h3>
            <div class="metric">
                <span class="metric-label">Detections</span>
                <span class="metric-value" id="detCount">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Inference Time</span>
                <span class="metric-value" id="infTime">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">FPS</span>
                <span class="metric-value" id="fpsMetric">—</span>
            </div>
            <div class="metric">
                <span class="metric-label">Last API Status</span>
                <span class="metric-value" id="apiStatus">—</span>
            </div>
            
            <h3 style="margin: 20px 0 15px; color: #00ff88;">📦 Detections</h3>
            <div class="detections-list" id="detectionsList">
                <div class="no-detections">Start camera to begin detection</div>
            </div>
        </div>
    </div>
    
    <a id="downloadLink" class="hidden"></a>

    <script>
        const API_URL = '/detect';
        
        const video = document.getElementById('webcam');
        const overlay = document.getElementById('overlay');
        const ctx = overlay.getContext('2d');
        const btnStart = document.getElementById('btnStart');
        const btnStop = document.getElementById('btnStop');
        const btnCapture = document.getElementById('btnCapture');
        const btnBest = document.getElementById('btnBest');
        const btnMirror = document.getElementById('btnMirror');
        const statusText = document.getElementById('statusText');
        const fpsText = document.getElementById('fpsText');
        const detCount = document.getElementById('detCount');
        const infTime = document.getElementById('infTime');
        const fpsMetric = document.getElementById('fpsMetric');
        const apiStatus = document.getElementById('apiStatus');
        const detectionsList = document.getElementById('detectionsList');
        const confThreshold = document.getElementById('confThreshold');
        const confValue = document.getElementById('confValue');
        const cameraSelect = document.getElementById('cameraSelect');
        const wrapper = document.querySelector('.video-wrapper');
        
        let stream = null;
        let isRunning = false;
        let mirrored = false;
        let intervalId = null;
        let frameCount = 0;
        let lastFpsTime = performance.now();
        let bestDetection = null;
        let bestFrameBlob = null;
        let requestId = 0;
        let lastProcessedId = 0;
        
        function resizeOverlay() {
            const rect = wrapper.getBoundingClientRect();
            overlay.width = Math.floor(rect.width);
            overlay.height = Math.floor(rect.height);
        }
        
        window.addEventListener('resize', resizeOverlay);
        resizeOverlay();
        
        async function enumerateCameras() {
            try {
                await navigator.mediaDevices.getUserMedia({ video: true });
                const devices = await navigator.mediaDevices.enumerateDevices();
                const cameras = devices.filter(d => d.kind === 'videoinput');
                cameraSelect.innerHTML = '<option value="">Default camera</option>';
                cameras.forEach((cam, idx) => {
                    const opt = document.createElement('option');
                    opt.value = cam.deviceId;
                    opt.text = cam.label || 'Camera ' + (idx + 1);
                    cameraSelect.appendChild(opt);
                });
            } catch (e) {
                console.error('Failed to enumerate cameras:', e);
            }
        }
        
        async function startCamera() {
            try {
                const constraints = {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined
                    },
                    audio: false
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                
                video.onloadedmetadata = () => {
                    video.play();
                    resizeOverlay();
                };
                
                isRunning = true;
                btnStart.disabled = true;
                btnStop.disabled = false;
                btnCapture.disabled = false;
                btnBest.disabled = false;
                statusText.textContent = '● Camera on — detecting...';
                statusText.className = 'status-active';
                
                intervalId = setInterval(processFrame, 600);
                requestAnimationFrame(updateFps);
            } catch (err) {
                alert('Camera error: ' + err.message);
                console.error(err);
            }
        }
        
        function stopCamera() {
            isRunning = false;
            if (stream) {
                stream.getTracks().forEach(t => t.stop());
                stream = null;
            }
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            video.srcObject = null;
            
            btnStart.disabled = false;
            btnStop.disabled = true;
            btnCapture.disabled = true;
            btnBest.disabled = true;
            statusText.textContent = '● Camera off';
            statusText.className = 'status-inactive';
            fpsText.textContent = '';
            detCount.textContent = '0';
            infTime.textContent = '—';
            fpsMetric.textContent = '—';
            apiStatus.textContent = '—';
            detectionsList.innerHTML = '<div class="no-detections">Start camera to begin detection</div>';
        }
        
        function updateFps() {
            if (!isRunning) return;
            
            const now = performance.now();
            frameCount++;
            if (now - lastFpsTime >= 1000) {
                const fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
                fpsMetric.textContent = fps + ' FPS';
                fpsText.textContent = '(' + fps + ' FPS)';
                frameCount = 0;
                lastFpsTime = now;
            }
            requestAnimationFrame(updateFps);
        }
        
        async function processFrame() {
            if (!isRunning || video.paused || video.ended || !video.videoWidth) return;
            
            const thisId = ++requestId;
            const blob = await captureFrameBlob();
            const start = performance.now();
            
            try {
                const formData = new FormData();
                formData.append('image', blob, 'frame.jpg');
                formData.append('conf', confThreshold.value);
                
                const res = await fetch(API_URL, { method: 'POST', body: formData });
                const data = await res.json();
                
                // Ignore out-of-order responses
                if (thisId < lastProcessedId) {
                    console.log('Ignoring out-of-order response', thisId);
                    return;
                }
                lastProcessedId = thisId;
                
                const elapsed = Math.round(performance.now() - start);
                infTime.textContent = elapsed + ' ms';
                apiStatus.textContent = res.status + ' OK';
                apiStatus.style.color = '#00ff88';

                console.log('API response:', data);

                // Track best detection for "Save Best" feature
                const dets = data.detections || [];
                if (dets.length > 0) {
                    const top = dets.reduce((a, b) => a.confidence > b.confidence ? a : b);
                    if (!bestDetection || top.confidence > bestDetection.confidence) {
                        bestDetection = top;
                        bestFrameBlob = blob;
                    }
                }

                drawDetections(dets);
                updateDetectionsList(dets);
            } catch (e) {
                console.error('API call failed:', e);
                infTime.textContent = 'ERR';
                apiStatus.textContent = 'FAIL';
                apiStatus.style.color = '#ff4444';
                ctx.clearRect(0, 0, overlay.width, overlay.height);
            }
        }
        
        function drawDetections(detections) {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            
            const scaleX = overlay.width / (video.videoWidth || 640);
            const scaleY = overlay.height / (video.videoHeight || 480);
            
            if (mirrored) {
                ctx.save();
                ctx.translate(overlay.width, 0);
                ctx.scale(-1, 1);
            }
            
            detections.forEach(det => {
                const x1 = det.bbox[0] * scaleX;
                const y1 = det.bbox[1] * scaleY;
                const x2 = det.bbox[2] * scaleX;
                const y2 = det.bbox[3] * scaleY;
                const w = x2 - x1;
                const h = y2 - y1;
                
                ctx.strokeStyle = '#00ff88';
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, w, h);
                
                const label = det.class + ' ' + Math.round(det.confidence * 100) + '%';
                ctx.font = 'bold 18px Arial';
                const textWidth = ctx.measureText(label).width;
                ctx.fillStyle = '#00ff88';
                ctx.fillRect(x1, y1 - 28, textWidth + 12, 28);
                
                ctx.fillStyle = '#000';
                ctx.fillText(label, x1 + 6, y1 - 7);
            });
            
            if (mirrored) ctx.restore();
            
            detCount.textContent = detections.length;
        }
        
        function updateDetectionsList(detections) {
            if (detections.length === 0) {
                detectionsList.innerHTML = '<div class="no-detections">No objects detected</div>';
                return;
            }
            
            var html = '';
            detections.forEach(det => {
                html += '<div class="detection-item">' +
                    '<div><strong>' + det.class + '</strong> <span class="conf">' + (det.confidence * 100).toFixed(1) + '%</span></div>' +
                    '<div class="bbox">bbox: (' + det.bbox.join(', ') + ')</div>' +
                '</div>';
            });
            detectionsList.innerHTML = html;
        }
        
        async function captureFrameBlob() {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = video.videoWidth || 640;
            tempCanvas.height = video.videoHeight || 480;
            const tctx = tempCanvas.getContext('2d');
            
            if (mirrored) {
                tctx.translate(tempCanvas.width, 0);
                tctx.scale(-1, 1);
            }
            
            tctx.drawImage(video, 0, 0);
            
            return new Promise(resolve => {
                tempCanvas.toBlob(resolve, 'image/jpeg', 0.85);
            });
        }
        
        function captureImage() {
            captureFrameBlob().then(blob => {
                const url = URL.createObjectURL(blob);
                const a = document.getElementById('downloadLink');
                a.href = url;
                a.download = 'capture_' + Date.now() + '.jpg';
                a.click();
                URL.revokeObjectURL(url);
            });
        }
        
        btnStart.addEventListener('click', startCamera);
        btnStop.addEventListener('click', stopCamera);
        btnCapture.addEventListener('click', captureImage);
        btnBest.addEventListener('click', () => {
            if (!bestFrameBlob) {
                alert('No detection captured yet.');
                return;
            }
            const url = URL.createObjectURL(bestFrameBlob);
            const a = document.getElementById('downloadLink');
            a.href = url;
            a.download = 'best_' + Date.now() + '_' + Math.round(bestDetection.confidence * 100) + 'pct.jpg';
            a.click();
            URL.revokeObjectURL(url);
        });
        btnMirror.addEventListener('click', () => {
            mirrored = !mirrored;
            video.classList.toggle('mirror', mirrored);
        });
        
        confThreshold.addEventListener('input', () => {
            confValue.textContent = parseFloat(confThreshold.value).toFixed(2);
        });
        
        enumerateCameras();
    </script>
</body>
</html>"""


def create_app(model_path, conf_threshold):
    app = Flask(__name__)
    CORS(app)
    
    print(f"Loading model: {model_path}")
    detector = DocumentDetector(
        model_path=model_path,
        input_size=(640, 640),
        confidence_threshold=conf_threshold,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.5,
        min_box_area=0.02,
    )
    
    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(HTML_PAGE)
    
    # Ensure logs directory exists
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)

    @app.route("/detect", methods=["POST"])
    def detect():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Read dynamic confidence threshold from frontend
        try:
            dynamic_conf = float(request.form.get("conf", detector.confidence_threshold))
        except (ValueError, TypeError):
            dynamic_conf = detector.confidence_threshold
        
        # Temporarily override threshold for this request
        original_conf = detector.confidence_threshold
        detector.confidence_threshold = dynamic_conf
        
        start = time.time()
        results = detector.detect(image)
        elapsed_ms = round((time.time() - start) * 1000, 2)
        
        # Restore original threshold
        detector.confidence_threshold = original_conf

        detections = [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox)
            }
            for d in results
        ]

        # Save debug image with bbox overlay for later review
        try:
            timestamp = int(time.time() * 1000)
            debug_image = detector.visualize(image, results)
            debug_path = LOG_DIR / f"debug_{timestamp}.jpg"
            cv2.imwrite(str(debug_path), debug_image)

            # Also log detection metadata to a JSONL file
            meta_path = LOG_DIR / "detections.jsonl"
            import json
            meta = {
                "timestamp": timestamp,
                "image_shape": list(image.shape),
                "inference_time_ms": elapsed_ms,
                "detections": detections
            }
            with open(meta_path, "a") as f:
                f.write(json.dumps(meta) + "\n")
        except Exception as e:
            print(f"Debug log error: {e}")

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
    parser = argparse.ArgumentParser(description="Combined Flask server for JS webcam detection")
    parser.add_argument("--model", "-m", default="models/yolox_idcard.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", "-c", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    app = create_app(str(model_path), args.conf)
    print(f"\n🚀 Server running at http://{args.host}:{args.port}")
    print("Open this URL in Chrome to use the webcam app")
    print("Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
