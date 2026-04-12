# Phase 6: API & Deployment - Architecture Specification

## Executive Summary

This document specifies the complete architecture for Phase 6 of Retin-Verify V3, including:
- REST API with FastAPI
- Web UI for mobile/handheld ID capture
- Real-time pipeline execution
- Docker containerization
- Production deployment

**Version:** 3.0.0  
**Date:** 2026-04-11  
**License:** MIT / Apache-2.0

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [API Layer Specification](#2-api-layer-specification)
3. [Web UI Specification](#3-web-ui-specification)
4. [Pipeline Integration](#4-pipeline-integration)
5. [Database Layer](#5-database-layer)
6. [Deployment Architecture](#6-deployment-architecture)
7. [Security Considerations](#7-security-considerations)
8. [Performance Requirements](#8-performance-requirements)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Web UI     │  │  Mobile App  │  │   API Client │  │   Admin UI   │    │
│  │  (Browser)   │  │  (Handheld)  │  │  (External)  │  │   (Web)      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │                 │
          └─────────────────┴─────────────────┴─────────────────┘
                              │
                              ▼ HTTPS/WSS
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│                    (Nginx / Traefik / Kong)                                  │
│         Rate Limiting │ SSL Termination │ Load Balancing │ Auth              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         FastAPI Application                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   REST API  │  │  WebSocket  │  │   GraphQL   │  │   Web UI    │  │   │
│  │  │   Router    │  │   Handler   │  │   Endpoint  │  │  (Static)   │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │   │
│  │         └─────────────────┴─────────────────┴─────────────────┘       │   │
│  │                              │                                        │   │
│  │                    ┌─────────┴─────────┐                              │   │
│  │                    │  Pipeline Manager │                              │   │
│  │                    │   (Async Queue)   │                              │   │
│  │                    └─────────┬─────────┘                              │   │
│  └──────────────────────────────┼───────────────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  DOCUMENT       │  │  FACE           │  │  SECURITY       │
│  DETECTION      │  │  DETECTION      │  │  ANALYSIS       │
│  SERVICE        │  │  SERVICE        │  │  SERVICE        │
│  (YOLOX/ONNX)   │  │  (YuNet/Haar)   │  │  (ELA/CopyMove) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  PostgreSQL  │  │    Redis     │  │  MinIO/S3    │  │   Elasticsearch│   │
│  │  (Primary)   │  │   (Cache)    │  │  (Storage)   │  │   (Search)     │   │
│  │              │  │              │  │              │  │                │   │
│  │ - Verifications│ - Sessions   │  │ - Images     │  │ - Logs         │   │
│  │ - Users      │  │ - Rate Limit │  │ - Results    │  │ - Analytics    │   │
│  │ - Audit Log  │  │ - Queue      │  │ - Models     │  │ - Metrics      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Client  │────▶│   API    │────▶│ Pipeline │────▶│ Services │────▶│  Storage │
│  (UI)    │◀────│ Gateway  │◀────│ Manager  │◀────│ (ML)     │◀────│ (DB/S3)  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                │                │                │                │
     │ 1. Upload      │                │                │                │
     │───────────────▶│                │                │                │
     │                │ 2. Queue Job   │                │                │
     │                │───────────────▶│                │                │
     │                │                │ 3. Execute     │                │
     │                │                │───────────────▶│                │
     │                │                │                │ 4. Process     │
     │                │                │                │───────────────▶│
     │                │                │                │ 5. Store       │
     │                │                │                │◀───────────────│
     │                │                │ 6. Results     │                │
     │                │                │◀───────────────│                │
     │                │ 7. Response    │                │                │
     │                │◀───────────────│                │                │
     │ 8. Display     │                │                │                │
     │◀───────────────│                │                │                │
```

---

## 2. API Layer Specification

### 2.1 FastAPI Application Structure

```
api/
├── __init__.py
├── main.py                    # Application entry point
├── config.py                  # API configuration
├── dependencies.py            # Dependency injection
├── middleware/
│   ├── __init__.py
│   ├── auth.py               # Authentication middleware
│   ├── rate_limit.py         # Rate limiting
│   ├── logging.py            # Request logging
│   └── cors.py               # CORS handling
├── routers/
│   ├── __init__.py
│   ├── verification.py       # ID verification endpoints
│   ├── health.py             # Health checks
│   ├── admin.py              # Admin endpoints
│   └── websocket.py          # WebSocket handlers
├── models/
│   ├── __init__.py
│   ├── requests.py           # Request schemas
│   ├── responses.py          # Response schemas
│   └── database.py           # Database models
├── services/
│   ├── __init__.py
│   ├── pipeline_service.py   # Pipeline orchestration
│   ├── storage_service.py    # File storage
│   └── cache_service.py      # Caching
└── static/
    ├── index.html            # Web UI
    ├── css/
    ├── js/
    └── assets/
```

### 2.2 Core API Endpoints

#### Verification Endpoints

```python
# POST /api/v1/verify
# Single image verification
{
    "request": {
        "image": "base64_encoded_image",
        "side": "auto|front|back",
        "options": {
            "detect_document": true,
            "extract_mrz": true,
            "extract_face": true,
            "security_check": true,
            "tampering_check": true
        }
    },
    "response": {
        "verification_id": "uuid",
        "status": "success|failed|partial",
        "confidence": 0.95,
        "results": {
            "document": {
                "detected": true,
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.98
            },
            "mrz": {
                "detected": true,
                "raw_lines": [...],
                "parsed": {
                    "document_number": "...",
                    "date_of_birth": "...",
                    "date_of_expiry": "...",
                    "surname": "...",
                    "given_names": "..."
                },
                "valid": true
            },
            "face": {
                "detected": true,
                "bbox": [x1, y1, x2, y2],
                "image_url": "...",
                "quality_score": 0.85
            },
            "security": {
                "hologram_detected": true,
                "laser_features_detected": true,
                "print_quality_score": 0.90,
                "authentic": true
            },
            "tampering": {
                "is_tampered": false,
                "ela_score": 0.15,
                "copy_move_detected": false,
                "confidence": 0.95
            }
        },
        "processing_time_ms": 1250,
        "timestamp": "2026-04-11T18:35:36Z"
    }
}

# POST /api/v1/verify/full
# Front + back verification with face matching
{
    "request": {
        "front_image": "base64_encoded",
        "back_image": "base64_encoded",
        "selfie_image": "base64_encoded",  # Optional
        "options": { ... }
    },
    "response": {
        "verification_id": "uuid",
        "status": "success",
        "results": {
            "front": { ... },
            "back": { ... },
            "face_matching": {
                "matched": true,
                "similarity": 0.87,
                "confidence": "high"
            },
            "cross_validation": {
                "mrz_matches_front": true,
                "document_numbers_match": true
            }
        }
    }
}

# WebSocket /ws/verify
# Real-time verification with progress updates
{
    "type": "progress",
    "stage": "document_detection",
    "progress": 0.25,
    "message": "Detecting document..."
}
```

#### Health & Admin Endpoints

```python
# GET /health
{
    "status": "healthy",
    "version": "3.0.0",
    "services": {
        "database": "connected",
        "cache": "connected",
        "storage": "connected"
    },
    "models": {
        "document_detection": "loaded",
        "face_detection": "loaded",
        "mrz_ocr": "loaded"
    }
}

# GET /api/v1/admin/stats
{
    "total_verifications": 15420,
    "success_rate": 0.94,
    "average_processing_time_ms": 1200,
    "daily_stats": [...]
}
```

### 2.3 Pipeline Service Class

```python
# api/services/pipeline_service.py

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import asyncio
from datetime import datetime
import uuid

@dataclass
class PipelineStage:
    name: str
    weight: float  # Progress contribution
    status: str = "pending"  # pending|running|completed|failed
    result: Optional[Dict] = None
    error: Optional[str] = None

class PipelineManager:
    """
    Orchestrates the complete verification pipeline.
    
    Stages:
    1. Document Detection (15%)
    2. MRZ Extraction (25%)
    3. Face Extraction (20%)
    4. Security Analysis (20%)
    5. Tampering Detection (20%)
    """
    
    STAGES = [
        ("document_detection", 0.15),
        ("mrz_extraction", 0.25),
        ("face_extraction", 0.20),
        ("security_analysis", 0.20),
        ("tampering_detection", 0.20),
    ]
    
    def __init__(
        self,
        doc_detector,
        mrz_pipeline,
        face_pipeline,
        security_analyzer,
        tampering_pipeline
    ):
        self.doc_detector = doc_detector
        self.mrz_pipeline = mrz_pipeline
        self.face_pipeline = face_pipeline
        self.security_analyzer = security_analyzer
        self.tampering_pipeline = tampering_pipeline
        
        self.active_jobs: Dict[str, Dict] = {}
    
    async def verify(
        self,
        image: np.ndarray,
        options: Dict[str, bool],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute complete verification pipeline.
        
        Args:
            image: Input image
            options: Feature flags for each stage
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete verification results
        """
        verification_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        stages = {
            name: PipelineStage(name, weight)
            for name, weight in self.STAGES
        }
        
        results = {
            "verification_id": verification_id,
            "status": "running",
            "stages": stages,
            "results": {}
        }
        
        self.active_jobs[verification_id] = results
        
        try:
            # Stage 1: Document Detection
            if options.get("detect_document", True):
                await self._run_stage(
                    verification_id, "document_detection",
                    self._detect_document, image, stages, progress_callback
                )
            
            # Stage 2: MRZ Extraction
            if options.get("extract_mrz", True):
                doc_image = self._get_document_image(image, results)
                await self._run_stage(
                    verification_id, "mrz_extraction",
                    self._extract_mrz, doc_image, stages, progress_callback
                )
            
            # Stage 3: Face Extraction
            if options.get("extract_face", True):
                doc_image = self._get_document_image(image, results)
                await self._run_stage(
                    verification_id, "face_extraction",
                    self._extract_face, doc_image, stages, progress_callback
                )
            
            # Stage 4: Security Analysis
            if options.get("security_check", True):
                doc_image = self._get_document_image(image, results)
                face_bbox = self._get_face_bbox(results)
                await self._run_stage(
                    verification_id, "security_analysis",
                    self._analyze_security, doc_image, face_bbox,
                    stages, progress_callback
                )
            
            # Stage 5: Tampering Detection
            if options.get("tampering_check", True):
                doc_image = self._get_document_image(image, results)
                await self._run_stage(
                    verification_id, "tampering_detection",
                    self._detect_tampering, doc_image, stages, progress_callback
                )
            
            # Calculate final status
            results["status"] = self._calculate_final_status(results)
            results["processing_time_ms"] = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Verification {verification_id} failed: {e}")
        
        finally:
            del self.active_jobs[verification_id]
        
        return results
    
    async def _run_stage(
        self,
        verification_id: str,
        stage_name: str,
        stage_func,
        *args,
        stages: Dict[str, PipelineStage],
        progress_callback: Optional[Callable] = None
    ):
        """Execute a pipeline stage with progress tracking."""
        stage = stages[stage_name]
        stage.status = "running"
        
        if progress_callback:
            await progress_callback({
                "type": "stage_start",
                "stage": stage_name,
                "message": f"Starting {stage_name}..."
            })
        
        try:
            # Run stage (potentially in thread pool for CPU-bound tasks)
            result = await asyncio.get_event_loop().run_in_executor(
                None, stage_func, *args
            )
            
            stage.status = "completed"
            stage.result = result
            
            # Update results
            self.active_jobs[verification_id]["results"][stage_name] = result
            
        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            logger.error(f"Stage {stage_name} failed: {e}")
        
        # Calculate progress
        completed_weight = sum(
            s.weight for s in stages.values()
            if s.status == "completed"
        )
        progress = completed_weight
        
        if progress_callback:
            await progress_callback({
                "type": "progress",
                "stage": stage_name,
                "progress": progress,
                "status": stage.status,
                "message": f"{stage_name} {stage.status}"
            })
    
    def _detect_document(self, image: np.ndarray) -> Dict:
        """Run document detection."""
        result = self.doc_detector.detect(image)
        return {
            "detected": result.success,
            "bbox": result.bbox if result.success else None,
            "confidence": result.confidence if result.success else 0,
            "image": result.cropped_image if result.success else None
        }
    
    def _extract_mrz(self, image: np.ndarray) -> Dict:
        """Run MRZ extraction."""
        result = self.mrz_pipeline.process(image)
        return {
            "detected": result.success,
            "raw_lines": result.raw_lines,
            "parsed": result.mrz_data.__dict__ if result.mrz_data else None,
            "valid": result.mrz_data.valid if result.mrz_data else False,
            "confidence": min(result.confidences) if result.confidences else 0
        }
    
    def _extract_face(self, image: np.ndarray) -> Dict:
        """Run face extraction."""
        result = self.face_pipeline.extract_face(image)
        return {
            "detected": result.success,
            "bbox": result.face_bbox,
            "quality_score": self._calculate_face_quality(result),
            "image": result.face_image
        }
    
    def _analyze_security(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple]
    ) -> Dict:
        """Run security analysis."""
        result = self.security_analyzer.analyze(image, face_bbox)
        return {
            "authentic": result.is_authentic,
            "overall_score": result.overall_score,
            "hologram": result.hologram_result,
            "laser": result.laser_result,
            "print_quality": result.print_quality
        }
    
    def _detect_tampering(self, image: np.ndarray) -> Dict:
        """Run tampering detection."""
        result = self.tampering_pipeline.analyze(image)
        return {
            "is_tampered": result.is_tampered,
            "confidence": result.overall_confidence,
            "ela": {
                "tampered": result.ela_result.is_tampered if result.ela_result else None,
                "confidence": result.ela_result.confidence if result.ela_result else 0
            },
            "copy_move": {
                "detected": result.copy_move_result.get('forgery_detected', False) if result.copy_move_result else False,
                "confidence": result.copy_move_result.get('confidence', 0) if result.copy_move_result else 0
            },
            "inconsistencies": result.inconsistencies
        }
    
    def _calculate_final_status(self, results: Dict) -> str:
        """Calculate final verification status."""
        r = results["results"]
        
        # Check critical failures
        if r.get("tampering", {}).get("is_tampered", False):
            return "rejected"
        
        if not r.get("document", {}).get("detected", False):
            return "failed"
        
        # Check partial success
        critical_stages = ["mrz", "face"]
        failed_critical = sum(
            1 for stage in critical_stages
            if not r.get(stage, {}).get("detected", False)
        )
        
        if failed_critical == len(critical_stages):
            return "failed"
        elif failed_critical > 0:
            return "partial"
        
        return "success"
```

---

## 3. Web UI Specification

### 3.1 UI Architecture

```
Web UI (Mobile-First Design)
│
├── Capture Interface
│   ├── Camera Component
│   │   ├── Live Preview
│   │   ├── Auto-focus
│   │   ├── Flash Control
│   │   └── Resolution Selection
│   │
│   ├── Capture Modes
│   │   ├── Single (Auto-detect side)
│   │   ├── Manual (Front/Back selection)
│   │   ├── Guided (Step-by-step)
│   │   └── Batch (Multiple IDs)
│   │
│   └── Quality Feedback
│       ├── Blur Detection
│       ├── Lighting Check
│       ├── Document Position
│       └── Real-time Guidance
│
├── Processing Interface
│   ├── Progress Display
│   │   ├── Stage Indicators
│   │   ├── Progress Bar
│   │   ├── Time Estimate
│   │   └── Cancel Option
│   │
│   └── Real-time Updates (WebSocket)
│
├── Results Interface
│   ├── Document Preview
│   ├── Extracted Data
│   │   ├── MRZ Fields
│   │   ├── Face Photo
│   │   └── Raw Image
│   │
│   ├── Verification Status
│   │   ├── Overall Result
│   │   ├── Security Score
│   │   ├── Tampering Check
│   │   └── Authenticity Badge
│   │
│   └── Action Buttons
│       ├── Save
│       ├── Share
│       ├── Print
│       └── Re-verify
│
└── History Interface
    ├── Past Verifications
    ├── Search/Filter
    └── Export
```

### 3.2 UI Components

#### Camera Capture Component

```html
<!-- Web UI - Camera Capture -->
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retin-Verify - ID Capture</title>
    <link rel="stylesheet" href="/static/css/capture.css">
</head>
<body>
    <div id="app">
        <!-- Header -->
        <header class="capture-header">
            <h1>ID Card Verification</h1>
            <div class="mode-selector">
                <button class="mode-btn active" data-mode="auto">Auto</button>
                <button class="mode-btn" data-mode="manual">Manual</button>
                <button class="mode-btn" data-mode="guided">Guided</button>
            </div>
        </header>

        <!-- Camera View -->
        <main class="capture-main">
            <div class="camera-container">
                <video id="camera-feed" autoplay playsinline></video>
                <canvas id="overlay-canvas"></canvas>
                
                <!-- Document Overlay Guide -->
                <div class="document-guide" id="doc-guide">
                    <div class="guide-frame"></div>
                    <p class="guide-text">Position ID card within frame</p>
                </div>
                
                <!-- Quality Indicators -->
                <div class="quality-indicators">
                    <div class="indicator" id="blur-indicator">
                        <span class="icon">📷</span>
                        <span class="label">Focus</span>
                        <div class="bar"><div class="fill"></div></div>
                    </div>
                    <div class="indicator" id="light-indicator">
                        <span class="icon">💡</span>
                        <span class="label">Light</span>
                        <div class="bar"><div class="fill"></div></div>
                    </div>
                    <div class="indicator" id="position-indicator">
                        <span class="icon">🎯</span>
                        <span class="label">Position</span>
                        <div class="bar"><div class="fill"></div></div>
                    </div>
                </div>
            </div>
            
            <!-- Capture Button -->
            <div class="capture-controls">
                <button id="flash-btn" class="control-btn">
                    <span class="icon">⚡</span>
                </button>
                <button id="capture-btn" class="capture-btn">
                    <div class="shutter"></div>
                </button>
                <button id="switch-btn" class="control-btn">
                    <span class="icon">🔄</span>
                </button>
            </div>
        </main>

        <!-- Processing Overlay -->
        <div id="processing-overlay" class="overlay hidden">
            <div class="processing-content">
                <div class="spinner"></div>
                <h2>Verifying ID...</h2>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <p class="stage-text" id="stage-text">Initializing...</p>
                </div>
                <div class="stage-indicators">
                    <div class="stage" data-stage="document">
                        <span class="icon">📄</span>
                        <span class="label">Document</span>
                    </div>
                    <div class="stage" data-stage="mrz">
                        <span class="icon">🔤</span>
                        <span class="label">MRZ</span>
                    </div>
                    <div class="stage" data-stage="face">
                        <span class="icon">👤</span>
                        <span class="label">Face</span>
                    </div>
                    <div class="stage" data-stage="security">
                        <span class="icon">🔒</span>
                        <span class="label">Security</span>
                    </div>
                    <div class="stage" data-stage="tampering">
                        <span class="icon">🔍</span>
                        <span class="label">Tampering</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Overlay -->
        <div id="results-overlay" class="overlay hidden">
            <div class="results-content">
                <div class="result-header">
                    <div class="status-badge" id="status-badge">
                        <span class="icon">✓</span>
                        <span class="text">VERIFIED</span>
                    </div>
                    <button class="close-btn" id="close-results">×</button>
                </div>
                
                <div class="result-body">
                    <!-- Document Preview -->
                    <div class="document-preview">
                        <img id="result-image" src="" alt="ID Card">
                        <div class="face-extract">
                            <img id="face-image" src="" alt="Face">
                        </div>
                    </div>
                    
                    <!-- Extracted Data -->
                    <div class="extracted-data">
                        <h3>Extracted Information</h3>
                        <div class="data-grid">
                            <div class="data-item">
                                <label>Document Number</label>
                                <span id="doc-number">-</span>
                            </div>
                            <div class="data-item">
                                <label>Date of Birth</label>
                                <span id="dob">-</span>
                            </div>
                            <div class="data-item">
                                <label>Expiry Date</label>
                                <span id="expiry">-</span>
                            </div>
                            <div class="data-item">
                                <label>Name</label>
                                <span id="name">-</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Security Analysis -->
                    <div class="security-analysis">
                        <h3>Security Analysis</h3>
                        <div class="security-grid">
                            <div class="security-item" id="hologram-check">
                                <span class="icon">✓</span>
                                <span class="label">Hologram</span>
                            </div>
                            <div class="security-item" id="laser-check">
                                <span class="icon">✓</span>
                                <span class="label">Laser Features</span>
                            </div>
                            <div class="security-item" id="ela-check">
                                <span class="icon">✓</span>
                                <span class="label">ELA Check</span>
                            </div>
                            <div class="security-item" id="copymove-check">
                                <span class="icon">✓</span>
                                <span class="label">Copy-Move</span>
                            </div>
                        </div>
                        <div class="security-score">
                            <div class="score-circle">
                                <svg viewBox="0 0 36 36">
                                    <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                                    <path class="circle-progress" id="score-circle" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                                </svg>
                                <div class="score-value" id="security-score">0%</div>
                            </div>
                            <span class="score-label">Security Score</span>
                        </div>
                    </div>
                </div>
                
                <div class="result-actions">
                    <button class="action-btn secondary" id="save-btn">
                        <span class="icon">💾</span> Save
                    </button>
                    <button class="action-btn secondary" id="share-btn">
                        <span class="icon">📤</span> Share
                    </button>
                    <button class="action-btn primary" id="verify-another">
                        <span class="icon">📷</span> Verify Another
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/js/capture.js"></script>
</body>
</html>
```

#### JavaScript Capture Controller

```javascript
// static/js/capture.js

class IDCaptureController {
    constructor() {
        this.video = document.getElementById('camera-feed');
        this.canvas = document.getElementById('overlay-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.ws = null;
        this.isProcessing = false;
        
        this.qualityMetrics = {
            blur: 0,
            lighting: 0,
            position: 0
        };
        
        this.init();
    }
    
    async init() {
        await this.setupCamera();
        this.setupEventListeners();
        this.startQualityMonitoring();
        this.connectWebSocket();
    }
    
    async setupCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            // Wait for video to be ready
            await new Promise(resolve => {
                this.video.onloadedmetadata = () => {
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    resolve();
                };
            });
            
        } catch (err) {
            console.error('Camera access failed:', err);
            alert('Camera access is required for ID verification');
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
    }
    
    startQualityMonitoring() {
        // Real-time quality analysis
        setInterval(() => {
            this.analyzeQuality();
        }, 200);
    }
    
    analyzeQuality() {
        if (!this.video.readyState === 4) return;
        
        // Draw current frame to canvas for analysis
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Get frame data
        const frameData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        
        // Calculate blur (Laplacian variance)
        this.qualityMetrics.blur = this.calculateSharpness(frameData);
        
        // Calculate lighting (average brightness)
        this.qualityMetrics.lighting = this.calculateLighting(frameData);
        
        // Update UI
        this.updateQualityIndicators();
        
        // Draw document guide overlay
        this.drawDocumentGuide();
    }
    
    calculateSharpness(frameData) {
        // Simplified sharpness calculation
        const data = frameData.data;
        let sum = 0;
        let sumSq = 0;
        const step = 4; // Sample every 4th pixel
        
        for (let i = 0; i < data.length; i += step * 4) {
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            sum += gray;
            sumSq += gray * gray;
        }
        
        const mean = sum / (data.length / (step * 4));
        const variance = (sumSq / (data.length / (step * 4))) - (mean * mean);
        
        // Normalize to 0-1
        return Math.min(1, variance / 500);
    }
    
    calculateLighting(frameData) {
        const data = frameData.data;
        let sum = 0;
        const step = 4;
        
        for (let i = 0; i < data.length; i += step * 4) {
            sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        }
        
        const mean = sum / (data.length / (step * 4));
        
        // Ideal range: 100-200
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
        lightBar.className = 'fill ' + (this.qualityMetrics.lighting > 0.7 ? 'good' : 'poor');
    }
    
    drawDocumentGuide() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Clear overlay
        ctx.clearRect(0, 0, w, h);
        
        // Draw document frame (ID card aspect ratio ~1.58)
        const frameW = Math.min(w * 0.8, h * 0.5 * 1.58);
        const frameH = frameW / 1.58;
        const x = (w - frameW) / 2;
        const y = (h - frameH) / 2;
        
        // Draw corner markers
        ctx.strokeStyle = this.isReadyToCapture() ? '#00ff00' : '#ffffff';
        ctx.lineWidth = 4;
        const cornerLen = 40;
        
        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        ctx.stroke();
        
        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + frameW - cornerLen, y);
        ctx.lineTo(x + frameW, y);
        ctx.lineTo(x + frameW, y + cornerLen);
        ctx.stroke();
        
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + frameH - cornerLen);
        ctx.lineTo(x, y + frameH);
        ctx.lineTo(x + cornerLen, y + frameH);
        ctx.stroke();
        
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + frameW - cornerLen, y + frameH);
        ctx.lineTo(x + frameW, y + frameH);
        ctx.lineTo(x + frameW, y + frameH - cornerLen);
        ctx.stroke();
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
    
    async startVerification(imageBlob) {
        this.isProcessing = true;
        
        // Show processing overlay
        document.getElementById('processing-overlay').classList.remove('hidden');
        
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
            // Connect WebSocket for real-time updates
            this.connectWebSocket();
            
            const response = await fetch('/api/v1/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64Image,
                    side: 'auto',
                    options: {
                        detect_document: true,
                        extract_mrz: true,
                        extract_face: true,
                        security_check: true,
                        tampering_check: true
                    }
                })
            });
            
            const result = await response.json();
            this.showResults(result);
            
        } catch (err) {
            console.error('Verification failed:', err);
            alert('Verification failed. Please try again.');
            this.hideProcessing();
        }
    }
    
    connectWebSocket() {
        if (this.ws) return;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/verify`);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'progress':
                this.updateProgress(data);
                break;
            case 'stage_complete':
                this.markStageComplete(data.stage);
                break;
            case 'complete':
                this.showResults(data.result);
                break;
            case 'error':
                this.handleError(data.error);
                break;
        }
    }
    
    updateProgress(data) {
        const progressFill = document.getElementById('progress-fill');
        const stageText = document.getElementById('stage-text');
        
        progressFill.style.width = `${data.progress * 100}%`;
        stageText.textContent = data.message;
        
        // Update stage indicators
        if (data.stage) {
            const stageEl = document.querySelector(`[data-stage="${data.stage}"]`);
            if (stageEl) {
                stageEl.classList.add('active');
            }
        }
    }
    
    markStageComplete(stage) {
        const stageEl = document.querySelector(`[data-stage="${stage}"]`);
        if (stageEl) {
            stageEl.classList.remove('active');
            stageEl.classList.add('completed');
        }
    }
    
    showResults(result) {
        this.hideProcessing();
        
        const overlay = document.getElementById('results-overlay');
        overlay.classList.remove('hidden');
        
        // Update status badge
        const statusBadge = document.getElementById('status-badge');
        const isSuccess = result.status === 'success';
        statusBadge.className = 'status-badge ' + (isSuccess ? 'success' : 'failed');
        statusBadge.innerHTML = `
            <span class="icon">${isSuccess ? '✓' : '✗'}</span>
            <span class="text">${isSuccess ? 'VERIFIED' : 'FAILED'}</span>
        `;
        
        // Update extracted data
        if (result.results?.mrz?.parsed) {
            const mrz = result.results.mrz.parsed;
            document.getElementById('doc-number').textContent = mrz.document_number || '-';
            document.getElementById('dob').textContent = mrz.date_of_birth || '-';
            document.getElementById('expiry').textContent = mrz.date_of_expiry || '-';
            document.getElementById('name').textContent = 
                `${mrz.surname || ''} ${mrz.given_names || ''}`.trim() || '-';
        }
        
        // Update security score
        const securityScore = result.results?.security?.overall_score || 0;
        document.getElementById('security-score').textContent = 
            Math.round(securityScore * 100) + '%';
        document.getElementById('score-circle').setAttribute(
            'stroke-dasharray', 
            `${securityScore * 100}, 100`
        );
        
        // Update security checks
        this.updateSecurityCheck('hologram', result.results?.security?.hologram?.authentic);
        this.updateSecurityCheck('laser', result.results?.security?.laser?.authentic);
        this.updateSecurityCheck('ela', !result.results?.tampering?.ela?.tampered);
        this.updateSecurityCheck('copymove', !result.results?.tampering?.copy_move?.detected);
    }
    
    updateSecurityCheck(id, passed) {
        const el = document.getElementById(`${id}-check`);
        el.className = 'security-item ' + (passed ? 'passed' : 'failed');
        el.querySelector('.icon').textContent = passed ? '✓' : '✗';
    }
    
    hideProcessing() {
        document.getElementById('processing-overlay').classList.add('hidden');
        this.isProcessing = false;
    }
    
    handleError(error) {
        console.error('Verification error:', error);
        alert('Error during verification: ' + error);
        this.hideProcessing();
    }
    
    async toggleFlash() {
        const track = this.stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();
        
        if (capabilities.torch) {
            const settings = track.getSettings();
            await track.applyConstraints({
                advanced: [{ torch: !settings.torch }]
            });
        }
    }
    
    async switchCamera() {
        // Toggle between front and back camera
        const currentFacing = this.stream.getVideoTracks()[0].getSettings().facingMode;
        const newFacing = currentFacing === 'environment' ? 'user' : 'environment';
        
        this.stream.getTracks().forEach(track => track.stop());
        
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: newFacing }
            });
            this.video.srcObject = this.stream;
        } catch (err) {
            console.error('Failed to switch camera:', err);
        }
    }
    
    setMode(mode) {
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        // Update UI based on mode
        this.currentMode = mode;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    new IDCaptureController();
});
```

---

## 4. Pipeline Integration

### 4.1 Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VERIFICATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: ID Card Image                                                        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 1: DOCUMENT DETECTION (15%)                                   │    │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │ │ Preprocess  │───▶│   YOLOX     │───▶│  Extract    │              │    │
│  │ │  Image      │    │  Inference  │    │  Document   │              │    │
│  │ └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │ Output: document_bbox, confidence, cropped_image                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 2: MRZ EXTRACTION (25%)                                       │    │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │ │  Detect MRZ │───▶│  OCR Text   │───▶│   Parse     │              │    │
│  │ │   Region    │    │  (Paddle)   │    │   MRZ       │              │    │
│  │ └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │ Output: mrz_lines, parsed_data, validation                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 3: FACE EXTRACTION (20%)                                      │    │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │ │   Detect    │───▶│   Align     │───▶│  Normalize  │              │    │
│  │ │    Face     │    │   Face      │    │   Face      │              │    │
│  │ └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │ Output: face_bbox, landmarks, normalized_face                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 4: SECURITY ANALYSIS (20%)                                    │    │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │ │  Hologram   │───▶│    Laser    │───▶│   Print     │              │    │
│  │ │  Detection  │    │  Detection  │    │   Quality   │              │    │
│  │ └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │ Output: security_score, authenticity, features                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 5: TAMPERING DETECTION (20%)                                  │    │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │    │
│  │ │    ELA      │───▶│  Copy-Move  │───▶│ Consistency │              │    │
│  │ │  Analysis   │    │  Detection  │    │    Check    │              │    │
│  │ └─────────────┘    └─────────────┘    └─────────────┘              │    │
│  │ Output: tampering_score, suspicious_regions, confidence             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FINAL DECISION ENGINE                            │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  Rules:                                                     │   │    │
│  │  │  - If tampering detected: REJECTED                          │   │    │
│  │  │  - If document not detected: FAILED                         │   │    │
│  │  │  - If MRZ invalid: PARTIAL                                  │   │    │
│  │  │  - If all checks pass: VERIFIED                             │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  OUTPUT: Verification Result                                                 │
│       ├── status: success|partial|failed|rejected                           │
│       ├── confidence: 0.0-1.0                                               │
│       ├── extracted_data: {mrz, face, document}                             │
│       ├── security_analysis: {score, checks}                                │
│       └── tampering_analysis: {score, flags}                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Pipeline Manager Implementation

```python
# api/services/pipeline_manager.py

import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    enable_document_detection: bool = True
    enable_mrz_extraction: bool = True
    enable_face_extraction: bool = True
    enable_security_analysis: bool = True
    enable_tampering_detection: bool = True
    timeout_seconds: int = 30
    max_retries: int = 2


@dataclass
class PipelineStageResult:
    """Result from a pipeline stage."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
    
    async def execute(self, input_data: Dict[str, Any]) -> PipelineStageResult:
        """Execute the stage."""
        raise NotImplementedError


class DocumentDetectionStage(PipelineStage):
    """Document detection stage."""
    
    def __init__(self, detector):
        super().__init__("document_detection", 0.15)
        self.detector = detector
    
    async def execute(self, input_data: Dict[str, Any]) -> PipelineStageResult:
        start = datetime.utcnow()
        
        try:
            image = input_data.get('image')
            if image is None:
                return PipelineStageResult(
                    success=False,
                    error="No image provided"
                )
            
            # Run detection in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.detector.detect, image
            )
            
            processing_time = (datetime.utcnow() - start).total_seconds() * 1000
            
            return PipelineStageResult(
                success=result.success,
                data={
                    'detected': result.success,
                    'bbox': result.bbox if result.success else None,
                    'confidence': result.confidence if result.success else 0,
                    'cropped_image': result.cropped_image if result.success else None
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Document detection failed: {e}")
            return PipelineStageResult(
                success=False,
                error=str(e)
            )


class MRZExtractionStage(PipelineStage):
    """MRZ extraction stage."""
    
    def __init__(self, pipeline):
        super().__init__("mrz_extraction", 0.25)
        self.pipeline = pipeline
    
    async def execute(self, input_data: Dict[str, Any]) -> PipelineStageResult:
        start = datetime.utcnow()
        
        try:
            # Use cropped document if available
            image = input_data.get('cropped_image') or input_data.get('image')
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.pipeline.process, image
            )
            
            processing_time = (datetime.utcnow() - start).total_seconds() * 1000
            
            return PipelineStageResult(
                success=result.success,
                data={
                    'detected': result.success,
                    'raw_lines': result.raw_lines,
                    'parsed': result.mrz_data.__dict__ if result.mrz_data else None,
                    'valid': result.mrz_data.valid if result.mrz_data else False,
                    'confidences': result.confidences
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"MRZ extraction failed: {e}")
            return PipelineStageResult(
                success=False,
                error=str(e)
            )


class FaceExtractionStage(PipelineStage):
    """Face extraction stage."""
    
    def __init__(self, pipeline):
        super().__init__("face_extraction", 0.20)
        self.pipeline = pipeline
    
    async def execute(self, input_data: Dict[str, Any]) -> PipelineStageResult:
        start = datetime.utcnow()
        
        try:
            image = input_data.get('cropped_image') or input_data.get('image')
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.pipeline.extract_face, image
            )
            
            processing_time = (datetime.utcnow() - start).total_seconds() * 1000
            
            return PipelineStageResult(
                success=result.success,
                data={
                    'detected': result.success,
                    'bbox': result.face_bbox,
                    'image': result.face_image,
                    'confidence': result.confidence,
                    'alignment_score': result.alignment.alignment_score if result.alignment else 0
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return PipelineStageResult(
                success=False,
                error=str(e)
            )


class SecurityAnalysisStage(PipelineStage):
    """Security analysis stage."""
    
    def __init__(self, analyzer):
        super().__init__("security_analysis", 0.20)
        self.analyzer = analyzer
    
    async def execute(self, input_data: Dict[str, Any]) -> PipelineStageResult:
        start = datetime.utcnow()
        
        try:
            image = input_data.get('cropped_image') or input_data.get('image')
            face_bbox = input_data.get('face_bbox')
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.analyzer.analyze, image, face_bbox
            )
            
            processing_time = (datetime.utcnow() - start).total_seconds() * 1000
            
            return PipelineStageResult(
                success=True,
                data={
                    'authentic': result.is_authentic,
                    'overall_score': result.overall_score,
                    'hologram': result.hologram_result,
                    'laser': result.laser_result,
                    'print_quality': result.print_quality
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return PipelineStageResult(
                success=False,
                error=str(e)
            )


class TamperingDetectionStage(PipelineStage):
    """Tampering detection stage."""
    
    def __init__(self, pipeline):
        super().__init__("tampering_detection", 0.20)
        self.pipeline = pipeline
    
    async def execute(self, input_data: Dict[str, Any]) -> PipelineStageResult:
        start = datetime.utcnow()
        
        try:
            image = input_data.get('cropped_image') or input_data.get('image')
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.pipeline.analyze, image
            )
            
            processing_time = (datetime.utcnow() - start).total_seconds() * 1000
            
            return PipelineStageResult(
                success=True,
                data={
                    'is_tampered': result.is_tampered,
                    'confidence': result.overall_confidence,
                    'ela': {
                        'tampered': result.ela_result.is_tampered if result.ela_result else None,
                        'confidence': result.ela_result.confidence if result.ela_result else 0
                    },
                    'copy_move': {
                        'detected': result.copy_move_result.get('forgery_detected', False) if result.copy_move_result else False,
                        'confidence': result.copy_move_result.get('confidence', 0) if result.copy_move_result else 0
                    },
                    'inconsistencies': result.inconsistencies
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Tampering detection failed: {e}")
            return PipelineStageResult(
                success=False,
                error=str(e)
            )


class VerificationPipeline:
    """
    Complete verification pipeline manager.
    """
    
    def __init__(
        self,
        doc_detector,
        mrz_pipeline,
        face_pipeline,
        security_analyzer,
        tampering_pipeline
    ):
        self.stages = [
            DocumentDetectionStage(doc_detector),
            MRZExtractionStage(mrz_pipeline),
            FaceExtractionStage(face_pipeline),
            SecurityAnalysisStage(security_analyzer),
            TamperingDetectionStage(tampering_pipeline)
        ]
    
    async def run(
        self,
        image: np.ndarray,
        config: PipelineConfig,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run complete verification pipeline.
        """
        verification_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        results = {
            'verification_id': verification_id,
            'status': 'running',
            'stages': {},
            'results': {}
        }
        
        context = {'image': image}
        completed_weight = 0.0
        
        # Determine which stages to run
        stage_enabled = {
            'document_detection': config.enable_document_detection,
            'mrz_extraction': config.enable_mrz_extraction,
            'face_extraction': config.enable_face_extraction,
            'security_analysis': config.enable_security_analysis,
            'tampering_detection': config.enable_tampering_detection
        }
        
        for stage in self.stages:
            if not stage_enabled.get(stage.name, True):
                continue
            
            # Report stage start
            if progress_callback:
                await progress_callback({
                    'type': 'stage_start',
                    'stage': stage.name,
                    'message': f'Starting {stage.name}...'
                })
            
            # Execute stage
            stage_result = await stage.execute(context)
            
            # Store result
            results['stages'][stage.name] = {
                'success': stage_result.success,
                'processing_time_ms': stage_result.processing_time_ms,
                'error': stage_result.error
            }
            
            if stage_result.success:
                results['results'][stage.name] = stage_result.data
                
                # Update context for next stages
                context.update(stage_result.data)
                
                # Update progress
                completed_weight += stage.weight
                
                if progress_callback:
                    await progress_callback({
                        'type': 'progress',
                        'stage': stage.name,
                        'progress': completed_weight,
                        'message': f'{stage.name} completed'
                    })
            else:
                # Stage failed
                results['stages'][stage.name]['status'] = 'failed'
                
                if progress_callback:
                    await progress_callback({
                        'type': 'error',
                        'stage': stage.name,
                        'message': stage_result.error
                    })
                
                # Continue with other stages unless critical
                if stage.name == 'document_detection':
                    break
        
        # Calculate final status
        results['status'] = self._calculate_final_status(results)
        results['total_processing_time_ms'] = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000
        
        return results
    
    def _calculate_final_status(self, results: Dict[str, Any]) -> str:
        """Calculate final verification status."""
        r = results['results']
        
        # Check tampering first
        if r.get('tampering_detection', {}).get('is_tampered', False):
            return 'rejected'
        
        # Check document detection
        if not r.get('document_detection', {}).get('detected', False):
            return 'failed'
        
        # Check MRZ and face
        mrz_ok = r.get('mrz_extraction', {}).get('valid', False)
        face_ok = r.get('face_extraction', {}).get('detected', False)
        
        if mrz_ok and face_ok:
            return 'success'
        elif mrz_ok or face_ok:
            return 'partial'
        else:
            return 'failed'
```

---

## 5. Database Layer

### 5.1 Database Schema

```sql
-- PostgreSQL Schema

-- Verifications table
CREATE TABLE verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) NOT NULL, -- success, partial, failed, rejected
    confidence DECIMAL(3,2),
    processing_time_ms INTEGER,
    
    -- Document info
    document_detected BOOLEAN,
    document_bbox INTEGER[],
    document_confidence DECIMAL(3,2),
    
    -- MRZ info
    mrz_detected BOOLEAN,
    mrz_valid BOOLEAN,
    document_number VARCHAR(50),
    date_of_birth DATE,
    date_of_expiry DATE,
    surname VARCHAR(100),
    given_names VARCHAR(100),
    nationality VARCHAR(10),
    
    -- Face info
    face_detected BOOLEAN,
    face_confidence DECIMAL(3,2),
    face_quality_score DECIMAL(3,2),
    
    -- Security analysis
    security_authentic BOOLEAN,
    security_score DECIMAL(3,2),
    hologram_detected BOOLEAN,
    laser_detected BOOLEAN,
    
    -- Tampering analysis
    tampering_detected BOOLEAN,
    tampering_confidence DECIMAL(3,2),
    ela_score DECIMAL(3,2),
    copy_move_detected BOOLEAN,
    
    -- Storage references
    image_path VARCHAR(500),
    face_image_path VARCHAR(500),
    
    -- Metadata
    client_ip INET,
    user_agent TEXT,
    source VARCHAR(50) -- web, mobile, api
);

-- Create indexes
CREATE INDEX idx_verifications_created_at ON verifications(created_at);
CREATE INDEX idx_verifications_status ON verifications(status);
CREATE INDEX idx_verifications_document_number ON verifications(document_number);

-- Audit log table
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    verification_id UUID REFERENCES verifications(id),
    action VARCHAR(50) NOT NULL,
    details JSONB,
    ip_address INET,
    user_id UUID
);

-- API keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    key_hash VARCHAR(256) NOT NULL,
    name VARCHAR(100),
    permissions VARCHAR(50)[],
    rate_limit INTEGER DEFAULT 100,
    active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP WITH TIME ZONE
);
```

### 5.2 Database Models (SQLAlchemy)

```python
# api/models/database.py

from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, 
    Numeric, ARRAY, JSON, ForeignKey, create_engine
)
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


class Verification(Base):
    __tablename__ = 'verifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), nullable=False)
    confidence = Column(Numeric(3, 2))
    processing_time_ms = Column(Integer)
    
    # Document
    document_detected = Column(Boolean)
    document_bbox = Column(ARRAY(Integer))
    document_confidence = Column(Numeric(3, 2))
    
    # MRZ
    mrz_detected = Column(Boolean)
    mrz_valid = Column(Boolean)
    document_number = Column(String(50))
    date_of_birth = Column(DateTime)
    date_of_expiry = Column(DateTime)
    surname = Column(String(100))
    given_names = Column(String(100))
    nationality = Column(String(10))
    
    # Face
    face_detected = Column(Boolean)
    face_confidence = Column(Numeric(3, 2))
    face_quality_score = Column(Numeric(3, 2))
    
    # Security
    security_authentic = Column(Boolean)
    security_score = Column(Numeric(3, 2))
    hologram_detected = Column(Boolean)
    laser_detected = Column(Boolean)
    
    # Tampering
    tampering_detected = Column(Boolean)
    tampering_confidence = Column(Numeric(3, 2))
    ela_score = Column(Numeric(3, 2))
    copy_move_detected = Column(Boolean)
    
    # Storage
    image_path = Column(String(500))
    face_image_path = Column(String(500))
    
    # Metadata
    client_ip = Column(INET)
    user_agent = Column(String)
    source = Column(String(50))
    
    audit_logs = relationship("AuditLog", back_populates="verification")
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status,
            'confidence': float(self.confidence) if self.confidence else None,
            'processing_time_ms': self.processing_time_ms,
            'document': {
                'detected': self.document_detected,
                'confidence': float(self.document_confidence) if self.document_confidence else None
            },
            'mrz': {
                'detected': self.mrz_detected,
                'valid': self.mrz_valid,
                'document_number': self.document_number,
                'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
                'date_of_expiry': self.date_of_expiry.isoformat() if self.date_of_expiry else None,
                'surname': self.surname,
                'given_names': self.given_names
            },
            'face': {
                'detected': self.face_detected,
                'confidence': float(self.face_confidence) if self.face_confidence else None
            },
            'security': {
                'authentic': self.security_authentic,
                'score': float(self.security_score) if self.security_score else None
            },
            'tampering': {
                'detected': self.tampering_detected,
                'confidence': float(self.tampering_confidence) if self.tampering_confidence else None
            }
        }


class AuditLog(Base):
    __tablename__ = 'audit_log'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    verification_id = Column(UUID(as_uuid=True), ForeignKey('verifications.id'))
    action = Column(String(50), nullable=False)
    details = Column(JSON)
    ip_address = Column(INET)
    user_id = Column(UUID(as_uuid=True))
    
    verification = relationship("Verification", back_populates="audit_logs")


# Database connection
engine = create_engine('postgresql://user:pass@localhost/retinverify')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## 6. Deployment Architecture

### 6.1 Docker Configuration

```dockerfile
# Dockerfile

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/models /app/uploads /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/retinverify
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    volumes:
      - ./models:/app/models
      - uploads:/app/uploads
    depends_on:
      - db
      - redis
      - minio
    networks:
      - retinverify

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=retinverify
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - retinverify

  redis:
    image: redis:7-alpine
    networks:
      - retinverify

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    networks:
      - retinverify

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    networks:
      - retinverify

volumes:
  postgres_data:
  minio_data:
  uploads:

networks:
  retinverify:
    driver: bridge
```

### 6.2 Kubernetes Deployment

```yaml
# k8s-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: retinverify-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: retinverify-api
  template:
    metadata:
      labels:
        app: retinverify-api
    spec:
      containers:
      - name: api
        image: retinverify/api:v3.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: retinverify-api
spec:
  selector:
    app: retinverify-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: retinverify-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt
spec:
  tls:
  - hosts:
    - api.retinverify.com
    secretName: retinverify-tls
  rules:
  - host: api.retinverify.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: retinverify-api
            port:
              number: 80
```

---

## 7. Security Considerations

### 7.1 Security Measures

| Layer | Measures |
|-------|----------|
| Transport | TLS 1.3, HSTS, Certificate pinning |
| Authentication | API keys, JWT tokens, OAuth 2.0 |
| Authorization | Role-based access control (RBAC) |
| Input Validation | Schema validation, file type checking, size limits |
| Data Protection | Encryption at rest (AES-256), PII masking |
| Rate Limiting | 100 req/min per API key |
| Audit Logging | All requests logged with IP, timestamp, result |

### 7.2 API Security Implementation

```python
# api/middleware/auth.py

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import hashlib
import hmac

security = HTTPBearer()

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def create_api_key(self, name: str, permissions: list) -> str:
        """Generate new API key."""
        api_key = f"rv_{secrets.token_urlsafe(32)}"
        
        # Store hash in database
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store in DB
        db_api_key = APIKey(
            key_hash=key_hash,
            name=name,
            permissions=permissions
        )
        db.add(db_api_key)
        db.commit()
        
        return api_key
    
    async def verify_api_key(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ):
        """Verify API key from request."""
        api_key = credentials.credentials
        
        # Check format
        if not api_key.startswith("rv_"):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        # Hash and lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        db_key = db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.active == True
        ).first()
        
        if not db_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Update last used
        db_key.last_used_at = datetime.utcnow()
        db.commit()
        
        return db_key
    
    def check_rate_limit(self, api_key_id: str) -> bool:
        """Check if request is within rate limit."""
        key = f"rate_limit:{api_key_id}"
        current = redis.get(key)
        
        if current is None:
            redis.setex(key, 60, 1)
            return True
        
        if int(current) >= 100:  # 100 requests per minute
            return False
        
        redis.incr(key)
        return True

# Dependency for protected routes
async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
):
    auth_manager = AuthManager(settings.SECRET_KEY)
    api_key = await auth_manager.verify_api_key(credentials)
    
    if not auth_manager.check_rate_limit(str(api_key.id)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return api_key
```

---

## 8. Performance Requirements

### 8.1 Target Performance Metrics

| Metric | Target | Maximum |
|--------|--------|---------|
| API Response Time (p95) | < 2s | 5s |
| Document Detection | < 500ms | 1s |
| MRZ Extraction | < 800ms | 2s |
| Face Extraction | < 600ms | 1.5s |
| Security Analysis | < 1s | 3s |
| Tampering Detection | < 1.5s | 4s |
| Throughput | 10 req/s | - |
| Availability | 99.9% | - |

### 8.2 Performance Optimization

```python
# api/services/cache_service.py

import redis
import pickle
from functools import wraps
from typing import Optional

class CacheService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        data = self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache."""
        data = pickle.dumps(value)
        self.redis.setex(key, ttl or self.default_ttl, data)
    
    async def delete(self, key: str):
        """Delete value from cache."""
        self.redis.delete(key)
    
    def cached(self, ttl: int = None, key_prefix: str = ""):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args))}"
                
                # Try cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator

# Usage example
@cache_service.cached(ttl=300, key_prefix="verification")
async def get_verification_result(verification_id: str):
    return db.query(Verification).filter(Verification.id == verification_id).first()
```

---

## Appendix A: API Endpoint Summary

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | /api/v1/verify | Single image verification | API Key |
| POST | /api/v1/verify/full | Full verification (front+back) | API Key |
| WS | /ws/verify | WebSocket real-time verification | API Key |
| GET | /api/v1/verify/{id} | Get verification result | API Key |
| GET | /health | Health check | None |
| GET | /api/v1/admin/stats | Admin statistics | Admin |
| GET | /api/v1/admin/verifications | List verifications | Admin |

## Appendix B: Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL connection string | - |
| REDIS_URL | Redis connection string | - |
| MINIO_ENDPOINT | MinIO/S3 endpoint | - |
| SECRET_KEY | Application secret key | - |
| LOG_LEVEL | Logging level | INFO |
| MAX_IMAGE_SIZE | Max upload size (MB) | 10 |
| WORKERS | Number of API workers | 4 |

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-11  
**Author:** Retin-Verify Team
