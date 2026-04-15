#!/usr/bin/env python3
"""
Data collection server for capturing training images.

Modes:
  /capture/front  - Capture id-front images
  /capture/back   - Capture id-back images
  /capture/none   - Capture no-card (negative) images

Images are saved to:
  data/collected/front/
  data/collected/back/
  data/collected/no-card/
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


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Capture Tool</title>
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
        h1 { color: #00ff88; margin-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 30px; font-size: 0.95rem; }
        .modes {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .mode-btn {
            padding: 20px 40px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        .mode-btn:hover { transform: scale(1.05); }
        .mode-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-front { background: #007bff; color: white; }
        .btn-back { background: #6f42c1; color: white; }
        .btn-none { background: #dc3545; color: white; }
        .video-panel {
            position: relative;
            width: 640px;
            height: 480px;
            max-width: 100%;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid #333;
        }
        #webcam {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .overlay-info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 15px 20px;
            border-radius: 8px;
            font-size: 1.2rem;
        }
        .overlay-info .label { color: #00ff88; font-weight: bold; }
        .overlay-info .timer { color: #ffd700; font-size: 2rem; margin-top: 5px; }
        .overlay-info .count { color: #aaa; font-size: 0.9rem; margin-top: 5px; }
        .status-bar {
            margin-top: 20px;
            padding: 15px 30px;
            background: #1a1a1a;
            border-radius: 8px;
            text-align: center;
            min-width: 300px;
        }
        .instructions {
            margin-top: 30px;
            max-width: 600px;
            background: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
            line-height: 1.6;
        }
        .instructions h3 { color: #00ff88; margin-bottom: 10px; }
        .instructions ul { margin-left: 20px; color: #aaa; }
        .instructions li { margin-bottom: 8px; }
    </style>
</head>
<body>
    <h1>📸 Dataset Capture Tool</h1>
    <p class="subtitle">Capture training images for ID card detection</p>
    
    <div class="modes">
        <button class="mode-btn btn-front" id="btnFront" onclick="startCapture('front')">🪪 Capture ID-Front</button>
        <button class="mode-btn btn-back" id="btnBack" onclick="startCapture('back')">🪪 Capture ID-Back</button>
        <button class="mode-btn btn-none" id="btnNone" onclick="startCapture('none')">❌ Capture No-Card</button>
    </div>
    
    <div class="video-panel">
        <video id="webcam" autoplay playsinline muted></video>
        <div class="overlay-info" id="overlayInfo" style="display: none;">
            <div class="label" id="modeLabel">Mode</div>
            <div class="timer" id="timer">60</div>
            <div class="count" id="count">Saved: 0 frames</div>
        </div>
    </div>
    
    <div class="status-bar" id="statusBar">
        Ready to capture. Click a mode above to start.
    </div>
    
    <div class="instructions">
        <h3>📋 How to Capture</h3>
        <ul>
            <li><strong>ID-Front:</strong> Hold the front side. Move it closer, farther, tilt it, rotate it slightly.</li>
            <li><strong>ID-Back:</strong> Same movements with the back side.</li>
            <li><strong>No-Card:</strong> Show your hand, face, wall, desk — anything EXCEPT an ID card.</li>
            <li>Each mode auto-captures for <strong>60 seconds</strong> (~120 frames).</li>
            <li>Images are saved to <code>data/collected/</code></li>
        </ul>
    </div>

    <script>
        const video = document.getElementById('webcam');
        const overlayInfo = document.getElementById('overlayInfo');
        const modeLabel = document.getElementById('modeLabel');
        const timerEl = document.getElementById('timer');
        const countEl = document.getElementById('count');
        const statusBar = document.getElementById('statusBar');
        const btns = {
            front: document.getElementById('btnFront'),
            back: document.getElementById('btnBack'),
            none: document.getElementById('btnNone')
        };
        
        let stream = null;
        let isCapturing = false;
        let captureInterval = null;
        let countdownInterval = null;
        let savedCount = 0;
        let timeLeft = 0;
        
        async function startCamera() {
            if (stream) return;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
            } catch (err) {
                alert('Camera error: ' + err.message);
            }
        }
        
        function getFrameBlob() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
        }
        
        async function sendFrame(mode) {
            try {
                const blob = await getFrameBlob();
                const formData = new FormData();
                formData.append('image', blob, 'frame.jpg');
                formData.append('mode', mode);
                
                const res = await fetch('/save', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.saved) {
                    savedCount++;
                    countEl.textContent = 'Saved: ' + savedCount + ' frames';
                } else {
                    console.error('Server rejected save:', data);
                    statusBar.textContent = '⚠️ Save failed: ' + (data.error || 'Unknown error');
                    statusBar.style.color = '#ff4444';
                }
            } catch (e) {
                console.error('Save failed:', e);
                statusBar.textContent = '⚠️ Network error — is the server running?';
                statusBar.style.color = '#ff4444';
            }
        }
        
        async function startCapture(mode) {
            if (isCapturing) return;
            await startCamera();
            
            isCapturing = true;
            savedCount = 0;
            timeLeft = 60;
            
            Object.values(btns).forEach(b => b.disabled = true);
            overlayInfo.style.display = 'block';
            modeLabel.textContent = mode === 'front' ? 'ID-FRONT' : mode === 'back' ? 'ID-BACK' : 'NO-CARD';
            modeLabel.style.color = mode === 'front' ? '#007bff' : mode === 'back' ? '#6f42c1' : '#dc3545';
            timerEl.textContent = timeLeft;
            countEl.textContent = 'Saved: 0 frames';
            statusBar.textContent = 'Capturing ' + mode + '... Move the card around!';
            
            // Capture every 500ms = ~120 frames in 60s
            captureInterval = setInterval(() => sendFrame(mode), 500);
            
            countdownInterval = setInterval(() => {
                timeLeft--;
                timerEl.textContent = timeLeft;
                if (timeLeft <= 0) {
                    stopCapture();
                }
            }, 1000);
        }
        
        function stopCapture() {
            isCapturing = false;
            clearInterval(captureInterval);
            clearInterval(countdownInterval);
            overlayInfo.style.display = 'none';
            Object.values(btns).forEach(b => b.disabled = false);
            if (savedCount === 0) {
                statusBar.textContent = '⚠️ Capture stopped. No frames were saved. Check server connection.';
                statusBar.style.color = '#ff4444';
            } else {
                statusBar.textContent = '✅ Capture complete! Saved ' + savedCount + ' frames to data/collected/';
                statusBar.style.color = '#00ff88';
            }
        }
        
        startCamera();
    </script>
</body>
</html>"""


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Ensure output directories exist
    BASE_DIR = Path("data/collected")
    for subdir in ["front", "back", "no-card"]:
        (BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)
    
    @app.route("/save", methods=["POST"])
    def save():
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400
        
        mode = request.form.get("mode", "none")
        if mode not in ["front", "back", "none"]:
            return jsonify({"error": "Invalid mode"}), 400
        
        # Map "none" to "no-card" for folder naming
        folder = "no-card" if mode == "none" else mode
        
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Failed to decode"}), 400
        
        # Save with timestamp
        timestamp = int(time.time() * 1000)
        filename = f"{mode}_{timestamp}.jpg"
        save_path = BASE_DIR / folder / filename
        cv2.imwrite(str(save_path), image)
        
        return jsonify({"saved": True, "path": str(save_path), "mode": mode})
    
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=5001, help="Port")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    args = parser.parse_args()
    
    app = create_app()
    print(f"\n🎥 Capture server running at http://{args.host}:{args.port}")
    print("Open this URL in Chrome to capture training images\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
