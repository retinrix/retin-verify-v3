/**
 * Retin-Verify V3 - Mobile ID Capture Application
 */

class IDCaptureApp {
    constructor() {
        this.video = document.getElementById('camera-feed');
        this.canvas = document.getElementById('overlay-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.ws = null;
        this.isProcessing = false;
        this.currentMode = 'auto';
        this.facingMode = 'environment';
        
        this.qualityMetrics = {
            blur: 0,
            lighting: 0,
            position: 0
        };
        
        this.init();
    }
    
    async init() {
        try {
            await this.setupCamera();
            this.setupEventListeners();
            this.startQualityMonitoring();
            console.log('ID Capture App initialized');
        } catch (err) {
            console.error('Initialization failed:', err);
            alert('Camera access is required for ID verification. Please allow camera access and refresh.');
        }
    }
    
    async setupCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: this.facingMode,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            await new Promise(resolve => {
                this.video.onloadedmetadata = () => {
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    resolve();
                };
            });
            
        } catch (err) {
            console.error('Camera setup failed:', err);
            throw err;
        }
    }
    
    setupEventListeners() {
        // Capture button
        document.getElementById('capture-btn').addEventListener('click', () => {
            this.capture();
        });
        
        // Flash toggle
        document.getElementById('flash-btn').addEventListener('click', () => {
            this.toggleFlash();
        });
        
        // Camera switch
        document.getElementById('switch-btn').addEventListener('click', () => {
            this.switchCamera();
        });
        
        // Mode selection
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setMode(e.target.dataset.mode);
            });
        });
        
        // File upload
        document.getElementById('file-input').addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
        
        // Results actions
        document.getElementById('close-results').addEventListener('click', () => {
            this.showCaptureView();
        });
        
        document.getElementById('verify-another').addEventListener('click', () => {
            this.showCaptureView();
        });
        
        document.getElementById('save-btn').addEventListener('click', () => {
            this.saveResults();
        });
        
        document.getElementById('share-btn').addEventListener('click', () => {
            this.shareResults();
        });
    }
    
    startQualityMonitoring() {
        // Real-time quality analysis
        const analyzeFrame = () => {
            if (!this.isProcessing) {
                this.analyzeQuality();
            }
            requestAnimationFrame(analyzeFrame);
        };
        requestAnimationFrame(analyzeFrame);
    }
    
    analyzeQuality() {
        if (this.video.readyState !== 4) return;
        
        // Draw current frame to canvas for analysis
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Get frame data
        const frameData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        
        // Calculate blur (Laplacian variance approximation)
        this.qualityMetrics.blur = this.calculateSharpness(frameData);
        
        // Calculate lighting
        this.qualityMetrics.lighting = this.calculateLighting(frameData);
        
        // Update UI
        this.updateQualityIndicators();
        
        // Draw document guide overlay
        this.drawDocumentGuide();
    }
    
    calculateSharpness(frameData) {
        const data = frameData.data;
        let sum = 0;
        let sumSq = 0;
        const step = 4; // Sample every 4th pixel
        const samples = Math.floor(data.length / (step * 4));
        
        for (let i = 0; i < data.length; i += step * 4) {
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            sum += gray;
            sumSq += gray * gray;
        }
        
        const mean = sum / samples;
        const variance = (sumSq / samples) - (mean * mean);
        
        // Normalize to 0-1
        return Math.min(1, variance / 500);
    }
    
    calculateLighting(frameData) {
        const data = frameData.data;
        let sum = 0;
        const step = 4;
        const samples = Math.floor(data.length / (step * 4));
        
        for (let i = 0; i < data.length; i += step * 4) {
            sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        }
        
        const mean = sum / samples;
        
        // Ideal range: 100-200 (out of 255)
        if (mean < 50 || mean > 250) return 0;
        return 1 - Math.abs(mean - 150) / 150;
    }
    
    updateQualityIndicators() {
        // Update blur indicator
        const blurBar = document.querySelector('#blur-indicator .fill');
        blurBar.style.width = `${this.qualityMetrics.blur * 100}%`;
        blurBar.className = 'fill ' + (this.qualityMetrics.blur > 0.7 ? 'good' : 'poor');
        
        // Update lighting indicator
        const lightBar = document.querySelector('#light-indicator .fill');
        lightBar.style.width = `${this.qualityMetrics.lighting * 100}%`;
        lightBar.className = 'fill ' + (this.qualityMetrics.lighting > 0.6 ? 'good' : 'poor');
        
        // Update position indicator (always good for now)
        const posBar = document.querySelector('#position-indicator .fill');
        posBar.style.width = '80%';
        posBar.className = 'fill good';
    }
    
    drawDocumentGuide() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Clear overlay
        this.ctx.clearRect(0, 0, w, h);
        
        // Draw document frame (ID card aspect ratio ~1.58)
        const frameW = Math.min(w * 0.8, h * 0.5 * 1.58);
        const frameH = frameW / 1.58;
        const x = (w - frameW) / 2;
        const y = (h - frameH) / 2;
        
        // Draw corner markers
        this.ctx.strokeStyle = this.isReadyToCapture() ? '#10b981' : 'rgba(255, 255, 255, 0.8)';
        this.ctx.lineWidth = 4;
        const cornerLen = 40;
        
        // Top-left
        this.ctx.beginPath();
        this.ctx.moveTo(x, y + cornerLen);
        this.ctx.lineTo(x, y);
        this.ctx.lineTo(x + cornerLen, y);
        this.ctx.stroke();
        
        // Top-right
        this.ctx.beginPath();
        this.ctx.moveTo(x + frameW - cornerLen, y);
        this.ctx.lineTo(x + frameW, y);
        this.ctx.lineTo(x + frameW, y + cornerLen);
        this.ctx.stroke();
        
        // Bottom-left
        this.ctx.beginPath();
        this.ctx.moveTo(x, y + frameH - cornerLen);
        this.ctx.lineTo(x, y + frameH);
        this.ctx.lineTo(x + cornerLen, y + frameH);
        this.ctx.stroke();
        
        // Bottom-right
        this.ctx.beginPath();
        this.ctx.moveTo(x + frameW - cornerLen, y + frameH);
        this.ctx.lineTo(x + frameW, y + frameH);
        this.ctx.lineTo(x + frameW, y + frameH - cornerLen);
        this.ctx.stroke();
    }
    
    isReadyToCapture() {
        return (
            this.qualityMetrics.blur > 0.7 &&
            this.qualityMetrics.lighting > 0.6
        );
    }
    
    async capture() {
        if (this.isProcessing) return;
        
        // Capture frame
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = this.video.videoWidth;
        captureCanvas.height = this.video.videoHeight;
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);
        
        // Convert to blob
        const blob = await new Promise(resolve => {
            captureCanvas.toBlob(resolve, 'image/jpeg', 0.95);
        });
        
        // Start verification
        this.startVerification(blob);
    }
    
    async handleFileUpload(file) {
        this.startVerification(file);
    }
    
    async startVerification(imageBlob) {
        this.isProcessing = true;
        
        // Show processing view
        this.showProcessingView();
        
        // Convert to base64
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            this.sendVerificationRequest(base64);
        };
        reader.readAsDataURL(imageBlob);
    }
    
    async sendVerificationRequest(base64Image) {
        try {
            // Use WebSocket for real-time updates if available
            if (window.WebSocket) {
                await this.sendWebSocketVerification(base64Image);
            } else {
                await this.sendHTTPVerification(base64Image);
            }
        } catch (err) {
            console.error('Verification failed:', err);
            alert('Verification failed. Please try again.');
            this.showCaptureView();
        }
    }
    
    async sendWebSocketVerification(base64Image) {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/verify`);
            
            this.ws.onopen = () => {
                this.ws.send(JSON.stringify({
                    type: 'start',
                    image: base64Image,
                    options: {
                        detect_document: true,
                        extract_mrz: true,
                        extract_face: true,
                        security_check: true,
                        tampering_check: true
                    }
                }));
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data, resolve);
            };
            
            this.ws.onerror = (err) => {
                reject(err);
            };
            
            this.ws.onclose = () => {
                this.ws = null;
            };
        });
    }
    
    async sendHTTPVerification(base64Image) {
        const response = await fetch('/api/v1/verify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64Image,
                side: this.currentMode,
                options: {
                    detect_document: true,
                    extract_mrz: true,
                    extract_face: true,
                    security_check: true,
                    tampering_check: true
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        this.showResults(result);
    }
    
    handleWebSocketMessage(data, resolve) {
        switch (data.type) {
            case 'progress':
                this.updateProgress(data);
                break;
            case 'stage_start':
                this.markStageActive(data.stage);
                break;
            case 'complete':
                this.showResults(data.result);
                resolve();
                break;
            case 'error':
                alert('Error: ' + data.message);
                this.showCaptureView();
                resolve();
                break;
        }
    }
    
    updateProgress(data) {
        const progressFill = document.getElementById('progress-fill');
        const stageText = document.getElementById('stage-text');
        
        progressFill.style.width = `${data.progress * 100}%`;
        stageText.textContent = data.message;
        
        if (data.stage) {
            this.markStageCompleted(data.stage);
        }
    }
    
    markStageActive(stage) {
        const stageEl = document.querySelector(`[data-stage="${stage}"]`);
        if (stageEl) {
            stageEl.classList.add('active');
        }
    }
    
    markStageCompleted(stage) {
        const stageEl = document.querySelector(`[data-stage="${stage}"]`);
        if (stageEl) {
            stageEl.classList.remove('active');
            stageEl.classList.add('completed');
        }
    }
    
    showResults(result) {
        this.isProcessing = false;
        
        // Switch to results view
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById('results-view').classList.add('active');
        
        // Update status badge
        const statusBadge = document.getElementById('status-badge');
        const isSuccess = result.status === 'success';
        statusBadge.className = 'status-badge ' + (isSuccess ? 'success' : 'failed');
        statusBadge.innerHTML = `
            <span class="icon">${isSuccess ? '✓' : '✗'}</span>
            <span class="text">${isSuccess ? 'VERIFIED' : 'FAILED'}</span>
        `;
        
        // Update extracted data
        if (result.results?.mrz_extraction?.parsed) {
            const mrz = result.results.mrz_extraction.parsed;
            document.getElementById('doc-number').textContent = mrz.document_number || '-';
            document.getElementById('dob').textContent = mrz.date_of_birth || '-';
            document.getElementById('expiry').textContent = mrz.date_of_expiry || '-';
            document.getElementById('name').textContent = 
                `${mrz.surname || ''} ${mrz.given_names || ''}`.trim() || '-';
        }
        
        // Update processing details
        document.getElementById('proc-time').textContent = 
            result.processing_time_ms ? `${result.processing_time_ms}ms` : '-';
        document.getElementById('doc-conf').textContent = 
            result.results?.document_detection?.confidence ? 
            `${Math.round(result.results.document_detection.confidence * 100)}%` : '-';
        document.getElementById('mrz-valid').textContent = 
            result.results?.mrz_extraction?.valid ? 'Yes' : 'No';
        document.getElementById('face-quality').textContent = 
            result.results?.face_extraction?.quality_score ? 
            `${Math.round(result.results.face_extraction.quality_score * 100)}%` : '-';
        
        // Update security score
        const securityScore = result.results?.security_analysis?.overall_score || 0;
        document.getElementById('security-score').textContent = 
            Math.round(securityScore * 100) + '%';
        document.getElementById('score-circle').setAttribute(
            'stroke-dasharray', 
            `${securityScore * 100}, 100`
        );
        
        // Update security checks
        this.updateSecurityCheck('hologram', 
            result.results?.security_analysis?.hologram?.authentic);
        this.updateSecurityCheck('laser', 
            result.results?.security_analysis?.laser?.authentic);
        this.updateSecurityCheck('ela', 
            !result.results?.tampering_detection?.ela?.tampered);
        this.updateSecurityCheck('copymove', 
            !result.results?.tampering_detection?.copy_move?.detected);
        
        // Store result for save/share
        this.lastResult = result;
    }
    
    updateSecurityCheck(id, passed) {
        const el = document.getElementById(`${id}-check`);
        if (!el) return;
        
        const isPass = passed === true;
        el.className = 'security-item ' + (isPass ? 'passed' : 'failed');
        el.querySelector('.icon').textContent = isPass ? '✓' : '✗';
    }
    
    showCaptureView() {
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById('capture-view').classList.add('active');
        
        // Reset stages
        document.querySelectorAll('.stage').forEach(s => {
            s.classList.remove('active', 'completed');
        });
        
        // Reset progress
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('stage-text').textContent = 'Initializing...';
    }
    
    showProcessingView() {
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById('processing-view').classList.add('active');
    }
    
    async toggleFlash() {
        const track = this.stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();
        
        if (capabilities.torch) {
            const settings = track.getSettings();
            try {
                await track.applyConstraints({
                    advanced: [{ torch: !settings.torch }]
                });
            } catch (err) {
                console.error('Flash toggle failed:', err);
            }
        }
    }
    
    async switchCamera() {
        this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
        
        // Stop current stream
        this.stream.getTracks().forEach(track => track.stop());
        
        try {
            await this.setupCamera();
        } catch (err) {
            console.error('Camera switch failed:', err);
            // Revert facing mode
            this.facingMode = this.facingMode === 'environment' ? 'user' : 'environment';
        }
    }
    
    setMode(mode) {
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        this.currentMode = mode;
    }
    
    saveResults() {
        if (!this.lastResult) return;
        
        const dataStr = JSON.stringify(this.lastResult, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `verification-${this.lastResult.verification_id}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    }
    
    async shareResults() {
        if (!this.lastResult) return;
        
        const shareData = {
            title: 'ID Verification Result',
            text: `Verification ${this.lastResult.status === 'success' ? 'successful' : 'failed'}`,
            url: window.location.href
        };
        
        if (navigator.share) {
            try {
                await navigator.share(shareData);
            } catch (err) {
                console.error('Share failed:', err);
            }
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(JSON.stringify(this.lastResult, null, 2));
            alert('Results copied to clipboard');
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new IDCaptureApp();
});

// Register service worker for PWA
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/js/sw.js')
        .then(reg => console.log('Service Worker registered'))
        .catch(err => console.error('Service Worker registration failed:', err));
}
