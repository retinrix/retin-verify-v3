"""Tampering detection module for Algerian ID card verification."""

from .ela_analyzer import ELAAnalyzer
from .copy_move_detector import CopyMoveDetector
from .tampering_pipeline import TamperingPipeline, TamperingResult

__all__ = [
    'ELAAnalyzer',
    'CopyMoveDetector',
    'TamperingPipeline',
    'TamperingResult',
]
