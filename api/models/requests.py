"""API request models."""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class IDSide(str, Enum):
    """ID card side."""
    AUTO = "auto"
    FRONT = "front"
    BACK = "back"


class VerificationOptions(BaseModel):
    """Verification options."""
    detect_document: bool = Field(default=True, description="Enable document detection")
    extract_mrz: bool = Field(default=True, description="Enable MRZ extraction")
    extract_face: bool = Field(default=True, description="Enable face extraction")
    security_check: bool = Field(default=True, description="Enable security analysis")
    tampering_check: bool = Field(default=True, description="Enable tampering detection")


class VerifyRequest(BaseModel):
    """Single image verification request."""
    image: str = Field(..., description="Base64 encoded image")
    side: IDSide = Field(default=IDSide.AUTO, description="ID card side")
    options: VerificationOptions = Field(default_factory=VerificationOptions)
    
    class Config:
        json_schema_extra = {
            "example": {
                "image": "/9j/4AAQSkZJRgABAQ...",
                "side": "auto",
                "options": {
                    "detect_document": True,
                    "extract_mrz": True,
                    "extract_face": True,
                    "security_check": True,
                    "tampering_check": True
                }
            }
        }


class VerifyFullRequest(BaseModel):
    """Full verification request (front + back)."""
    front_image: str = Field(..., description="Base64 encoded front image")
    back_image: str = Field(..., description="Base64 encoded back image")
    selfie_image: Optional[str] = Field(None, description="Optional selfie for face matching")
    options: VerificationOptions = Field(default_factory=VerificationOptions)


class BatchVerifyRequest(BaseModel):
    """Batch verification request."""
    images: List[str] = Field(..., description="List of base64 encoded images")
    options: VerificationOptions = Field(default_factory=VerificationOptions)


class FaceMatchRequest(BaseModel):
    """Face matching request."""
    id_face_image: str = Field(..., description="Base64 encoded ID face image")
    selfie_image: str = Field(..., description="Base64 encoded selfie image")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Matching threshold")
