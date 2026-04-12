"""Pipeline manager service."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np

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
            face_bbox = input_data.get('bbox')
            
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
    """Complete verification pipeline manager."""
    
    def __init__(self):
        """Initialize pipeline with all components."""
        # Import here to avoid circular imports
        from document_detection import DocumentDetector
        from mrz_ocr import MRZPipeline
        from face_detection import FacePipeline
        from security_features import SecurityAnalyzer
        from tampering_detection import TamperingPipeline
        
        self.doc_detector = DocumentDetector()
        self.mrz_pipeline = MRZPipeline()
        self.face_pipeline = FacePipeline()
        self.security_analyzer = SecurityAnalyzer()
        self.tampering_pipeline = TamperingPipeline()
        
        self.stages = [
            DocumentDetectionStage(self.doc_detector),
            MRZExtractionStage(self.mrz_pipeline),
            FaceExtractionStage(self.face_pipeline),
            SecurityAnalysisStage(self.security_analyzer),
            TamperingDetectionStage(self.tampering_pipeline)
        ]
    
    async def run(
        self,
        image: np.ndarray,
        config: PipelineConfig,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run complete verification pipeline."""
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
                results['results'][stage.name.replace('_', '_')] = stage_result.data
                
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
        results['total_processing_time_ms'] = int(
            (datetime.utcnow() - start_time).total_seconds() * 1000
        )
        
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
