"""Admin routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from typing import Optional

from api.models.database import get_db, Verification

router = APIRouter()


@router.get("/stats")
async def get_stats(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get verification statistics.
    
    Args:
        days: Number of days to include in stats
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    # Total verifications
    total = db.query(Verification).filter(
        Verification.created_at >= since
    ).count()
    
    # By status
    status_counts = db.query(
        Verification.status,
        func.count(Verification.id)
    ).filter(
        Verification.created_at >= since
    ).group_by(Verification.status).all()
    
    # Average processing time
    avg_time = db.query(
        func.avg(Verification.processing_time_ms)
    ).filter(
        Verification.created_at >= since,
        Verification.processing_time_ms.isnot(None)
    ).scalar()
    
    # Success rate
    successful = db.query(Verification).filter(
        Verification.created_at >= since,
        Verification.status.in_(['success', 'partial'])
    ).count()
    
    success_rate = successful / total if total > 0 else 0
    
    return {
        "total_verifications": total,
        "success_rate": round(success_rate, 2),
        "average_processing_time_ms": round(avg_time, 2) if avg_time else None,
        "by_status": {status: count for status, count in status_counts},
        "period_days": days
    }


@router.get("/verifications")
async def list_verifications(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List recent verifications."""
    query = db.query(Verification)
    
    if status:
        query = query.filter(Verification.status == status)
    
    verifications = query.order_by(
        Verification.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    return {
        "verifications": [v.to_dict() for v in verifications],
        "total": query.count()
    }
