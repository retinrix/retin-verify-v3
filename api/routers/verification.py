"""Verification API routes."""

import base64
import io
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from sqlalchemy.orm import Session
import cv2
import logging

from api.models.requests import VerifyRequest, VerifyFullRequest, FaceMatchRequest
from api.models.responses import (
    VerificationResponse, FaceMatchResponse, VerificationStatus
)
from api.models.database import get_db, Verification as VerificationModel, AuditLog
from api.services.pipeline_manager import VerificationPipeline, PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter()


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@router.post("/verify", response_model=VerificationResponse)
async def verify_id(
    request: VerifyRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    http_request: Request = None
):
    """
    Verify ID card from single image.
    
    This endpoint performs complete verification including:
    - Document detection
    - MRZ extraction
    - Face extraction
    - Security analysis
    - Tampering detection
    """
    try:
        # Decode image
        image = decode_base64_image(request.image)
        
        # Create pipeline config
        config = PipelineConfig(
            enable_document_detection=request.options.detect_document,
            enable_mrz_extraction=request.options.extract_mrz,
            enable_face_extraction=request.options.extract_face,
            enable_security_analysis=request.options.security_check,
            enable_tampering_detection=request.options.tampering_check
        )
        
        # Get pipeline from app state
        pipeline = http_request.app.state.pipeline
        
        # Run verification
        result = await pipeline.run(image, config)
        
        # Save to database
        verification = VerificationModel(
            status=result['status'],
            confidence=result.get('confidence'),
            processing_time_ms=result.get('total_processing_time_ms'),
            client_ip=http_request.client.host if http_request else None,
            source='api'
        )
        
        # Add document info
        if 'document_detection' in result['results']:
            doc = result['results']['document_detection']
            verification.document_detected = doc.get('detected')
            verification.document_bbox = doc.get('bbox')
            verification.document_confidence = doc.get('confidence')
        
        # Add MRZ info
        if 'mrz_extraction' in result['results']:
            mrz = result['results']['mrz_extraction']
            verification.mrz_detected = mrz.get('detected')
            verification.mrz_valid = mrz.get('valid')
            if mrz.get('parsed'):
                parsed = mrz['parsed']
                verification.document_number = parsed.get('document_number')
                verification.surname = parsed.get('surname')
                verification.given_names = parsed.get('given_names')
        
        # Add face info
        if 'face_extraction' in result['results']:
            face = result['results']['face_extraction']
            verification.face_detected = face.get('detected')
            verification.face_confidence = face.get('confidence')
            verification.face_quality_score = face.get('quality_score')
        
        # Add security info
        if 'security_analysis' in result['results']:
            sec = result['results']['security_analysis']
            verification.security_authentic = sec.get('authentic')
            verification.security_score = sec.get('overall_score')
            if sec.get('hologram'):
                verification.hologram_detected = sec['hologram'].get('authentic')
            if sec.get('laser'):
                verification.laser_detected = sec['laser'].get('authentic')
        
        # Add tampering info
        if 'tampering_detection' in result['results']:
            tam = result['results']['tampering_detection']
            verification.tampering_detected = tam.get('is_tampered')
            verification.tampering_confidence = tam.get('confidence')
            if tam.get('ela'):
                verification.ela_score = tam['ela'].get('confidence')
            if tam.get('copy_move'):
                verification.copy_move_detected = tam['copy_move'].get('detected')
        
        db.add(verification)
        db.commit()
        db.refresh(verification)
        
        # Add audit log
        audit = AuditLog(
            verification_id=verification.id,
            action='verify',
            ip_address=http_request.client.host if http_request else None
        )
        db.add(audit)
        db.commit()
        
        return VerificationResponse(
            verification_id=str(verification.id),
            status=VerificationStatus(result['status']),
            results=result['results'],
            processing_time_ms=result.get('total_processing_time_ms')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.post("/verify/full", response_model=VerificationResponse)
async def verify_full(
    request: VerifyFullRequest,
    db: Session = Depends(get_db)
):
    """
    Verify ID card from front and back images.
    
    Optional selfie image for face matching.
    """
    # TODO: Implement full verification
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.post("/face-match", response_model=FaceMatchResponse)
async def match_faces(request: FaceMatchRequest):
    """
    Match ID face with selfie.
    
    Returns similarity score and match result.
    """
    try:
        # Decode images
        id_face = decode_base64_image(request.id_face_image)
        selfie = decode_base64_image(request.selfie_image)
        
        # Simple comparison using histogram
        id_gray = cv2.cvtColor(id_face, cv2.COLOR_BGR2GRAY)
        selfie_gray = cv2.cvtColor(selfie, cv2.COLOR_BGR2GRAY)
        
        # Resize to same size
        id_resized = cv2.resize(id_gray, (112, 112))
        selfie_resized = cv2.resize(selfie_gray, (112, 112))
        
        # Calculate histogram similarity
        hist1 = cv2.calcHist([id_resized], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([selfie_resized], [0], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        similarity = max(0, similarity)
        
        matched = similarity >= request.threshold
        
        confidence = "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
        
        return FaceMatchResponse(
            matched=matched,
            similarity=float(similarity),
            confidence=confidence,
            threshold=request.threshold
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face match error: {e}")
        raise HTTPException(status_code=500, detail=f"Face matching failed: {str(e)}")


@router.get("/verify/{verification_id}")
async def get_verification(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """Get verification result by ID."""
    from uuid import UUID
    
    try:
        verification = db.query(VerificationModel).filter(
            VerificationModel.id == UUID(verification_id)
        ).first()
        
        if not verification:
            raise HTTPException(status_code=404, detail="Verification not found")
        
        return verification.to_dict()
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid verification ID")
