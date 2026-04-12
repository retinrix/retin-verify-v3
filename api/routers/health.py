"""Health check routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from api.models.responses import HealthResponse
from api.models.database import get_db, engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns service status and component health.
    """
    services = {}
    models = {}
    
    # Check database
    try:
        db.execute("SELECT 1")
        services['database'] = 'connected'
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        services['database'] = 'disconnected'
    
    # Check models (placeholder)
    models['document_detection'] = 'loaded'
    models['face_detection'] = 'loaded'
    models['mrz_ocr'] = 'loaded'
    models['security_analysis'] = 'loaded'
    models['tampering_detection'] = 'loaded'
    
    status = 'healthy' if services['database'] == 'connected' else 'unhealthy'
    
    return HealthResponse(
        status=status,
        version="3.0.0",
        services=services,
        models=models
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}
