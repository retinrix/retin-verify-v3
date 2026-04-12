"""API response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class VerificationStatus(str, Enum):
    """Verification status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    REJECTED = "rejected"
    RUNNING = "running"


class DocumentResult(BaseModel):
    """Document detection result."""
    detected: bool
    bbox: Optional[List[int]] = None
    confidence: Optional[float] = None


class MRZResult(BaseModel):
    """MRZ extraction result."""
    detected: bool
    raw_lines: Optional[List[str]] = None
    parsed: Optional[Dict[str, Any]] = None
    valid: bool = False
    confidence: Optional[float] = None


class FaceResult(BaseModel):
    """Face extraction result."""
    detected: bool
    bbox: Optional[List[int]] = None
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    image_url: Optional[str] = None


class SecurityResult(BaseModel):
    """Security analysis result."""
    authentic: bool
    overall_score: float
    hologram: Optional[Dict[str, Any]] = None
    laser: Optional[Dict[str, Any]] = None
    print_quality: Optional[Dict[str, Any]] = None


class TamperingResult(BaseModel):
    """Tampering detection result."""
    is_tampered: bool
    confidence: float
    ela: Optional[Dict[str, Any]] = None
    copy_move: Optional[Dict[str, Any]] = None
    inconsistencies: List[str] = Field(default_factory=list)


class VerificationResponse(BaseModel):
    """Verification response."""
    verification_id: str
    status: VerificationStatus
    confidence: Optional[float] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "verification_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "confidence": 0.95,
                "results": {
                    "document": {"detected": True, "confidence": 0.98},
                    "mrz": {"detected": True, "valid": True},
                    "face": {"detected": True, "confidence": 0.92},
                    "security": {"authentic": True, "overall_score": 0.88},
                    "tampering": {"is_tampered": False, "confidence": 0.95}
                },
                "processing_time_ms": 1250,
                "timestamp": "2026-04-11T18:35:36Z"
            }
        }


class FaceMatchResponse(BaseModel):
    """Face matching response."""
    matched: bool
    similarity: float
    confidence: str
    threshold: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)
    models: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
