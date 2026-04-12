"""Database models."""

from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, 
    Numeric, ARRAY, JSON, ForeignKey, create_engine, Text
)
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


class Verification(Base):
    """Verification record."""
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
    user_agent = Column(Text)
    source = Column(String(50))
    
    audit_logs = relationship("AuditLog", back_populates="verification")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status,
            'confidence': float(self.confidence) if self.confidence else None,
            'document': {
                'detected': self.document_detected,
                'confidence': float(self.document_confidence) if self.document_confidence else None
            },
            'mrz': {
                'detected': self.mrz_detected,
                'valid': self.mrz_valid,
                'document_number': self.document_number,
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
    """Audit log entry."""
    __tablename__ = 'audit_log'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    verification_id = Column(UUID(as_uuid=True), ForeignKey('verifications.id'))
    action = Column(String(50), nullable=False)
    details = Column(JSON)
    ip_address = Column(INET)
    user_id = Column(UUID(as_uuid=True))
    
    verification = relationship("Verification", back_populates="audit_logs")


class APIKey(Base):
    """API key record."""
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    key_hash = Column(String(256), nullable=False)
    name = Column(String(100))
    permissions = Column(ARRAY(String))
    rate_limit = Column(Integer, default=100)
    active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True))


# Database connection
engine = create_engine(
    'postgresql://postgres:password@localhost:5432/retinverify',
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
