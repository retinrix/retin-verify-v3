"""Complete tampering detection pipeline for ID card verification."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .ela_analyzer import ELAAnalyzer, ELAResult
from .copy_move_detector import CopyMoveDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TamperingResult:
    """Complete tampering detection result."""
    is_tampered: bool
    overall_confidence: float
    ela_result: Optional[ELAResult]
    copy_move_result: Optional[Dict]
    metadata_analysis: Dict
    inconsistencies: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_tampered': self.is_tampered,
            'overall_confidence': self.overall_confidence,
            'ela': {
                'is_tampered': self.ela_result.is_tampered if self.ela_result else None,
                'confidence': self.ela_result.confidence if self.ela_result else 0.0,
                'error_ratio': self.ela_result.error_ratio if self.ela_result else 0.0,
                'suspicious_regions': len(self.ela_result.suspicious_regions) if self.ela_result else 0,
            },
            'copy_move': {
                'forgery_detected': self.copy_move_result.get('forgery_detected', False) if self.copy_move_result else False,
                'confidence': self.copy_move_result.get('confidence', 0.0) if self.copy_move_result else 0.0,
                'num_matches': self.copy_move_result.get('num_matches', 0) if self.copy_move_result else 0,
            },
            'metadata_analysis': self.metadata_analysis,
            'inconsistencies': self.inconsistencies,
            'recommendations': self.recommendations
        }


class TamperingPipeline:
    """
    Complete tampering detection pipeline.
    
    Combines multiple detection methods:
    1. Error Level Analysis (ELA)
    2. Copy-move forgery detection
    3. Metadata analysis
    4. Consistency checks
    
    Provides a comprehensive assessment of image authenticity.
    """
    
    def __init__(
        self,
        ela_analyzer: Optional[ELAAnalyzer] = None,
        copy_move_detector: Optional[CopyMoveDetector] = None,
    ):
        """
        Initialize tampering pipeline.
        
        Args:
            ela_analyzer: ELA analyzer instance
            copy_move_detector: Copy-move detector instance
        """
        self.ela_analyzer = ela_analyzer or ELAAnalyzer()
        self.copy_move_detector = copy_move_detector or CopyMoveDetector()
    
    def analyze(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None
    ) -> TamperingResult:
        """
        Perform complete tampering analysis.
        
        Args:
            image: Input image
            image_path: Optional path to image file (for metadata)
            
        Returns:
            Tampering detection result
        """
        inconsistencies = []
        recommendations = []
        
        # 1. ELA Analysis
        try:
            ela_result = self.ela_analyzer.analyze(image)
            if ela_result.is_tampered:
                inconsistencies.append(f"ELA detected tampering (confidence: {ela_result.confidence:.2f})")
                recommendations.append("Review ELA heatmap for suspicious regions")
        except Exception as e:
            logger.warning(f"ELA analysis failed: {e}")
            ela_result = None
            inconsistencies.append("ELA analysis failed")
        
        # 2. Copy-move detection
        try:
            copy_move_result = self.copy_move_detector.detect(image)
            if copy_move_result.get('forgery_detected', False):
                inconsistencies.append(f"Copy-move forgery detected (confidence: {copy_move_result.get('confidence', 0):.2f})")
                recommendations.append("Check for duplicated regions in the image")
        except Exception as e:
            logger.warning(f"Copy-move detection failed: {e}")
            copy_move_result = None
            inconsistencies.append("Copy-move detection failed")
        
        # 3. Metadata analysis
        try:
            metadata_analysis = self._analyze_metadata(image, image_path)
            if metadata_analysis.get('suspicious_software'):
                inconsistencies.append(f"Suspicious software detected: {metadata_analysis['suspicious_software']}")
        except Exception as e:
            logger.warning(f"Metadata analysis failed: {e}")
            metadata_analysis = {}
        
        # 4. Consistency checks
        consistency_checks = self._perform_consistency_checks(image)
        for check_name, passed, message in consistency_checks:
            if not passed:
                inconsistencies.append(message)
        
        # Calculate overall result
        is_tampered, confidence = self._calculate_overall_result(
            ela_result, copy_move_result, inconsistencies
        )
        
        # Add recommendations based on result
        if is_tampered:
            recommendations.append("Image shows signs of manipulation - manual review recommended")
        else:
            recommendations.append("No obvious tampering detected, but verify with other checks")
        
        return TamperingResult(
            is_tampered=is_tampered,
            overall_confidence=confidence,
            ela_result=ela_result,
            copy_move_result=copy_move_result,
            metadata_analysis=metadata_analysis,
            inconsistencies=inconsistencies,
            recommendations=recommendations
        )
    
    def analyze_file(self, image_path: str) -> TamperingResult:
        """
        Analyze image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tampering detection result
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return TamperingResult(
                is_tampered=False,
                overall_confidence=0.0,
                ela_result=None,
                copy_move_result=None,
                metadata_analysis={'error': 'Cannot load image'},
                inconsistencies=["Failed to load image"],
                recommendations=["Check file path and format"]
            )
        
        return self.analyze(image, image_path)
    
    def _analyze_metadata(
        self,
        image: np.ndarray,
        image_path: Optional[str]
    ) -> Dict:
        """Analyze image metadata."""
        results = {
            'has_metadata': False,
            'software': None,
            'suspicious_software': None,
            'compression_quality': None,
            'dimensions': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        }
        
        if image_path is None:
            return results
        
        try:
            # Try to read EXIF data
            from PIL import Image as PILImage
            with PILImage.open(image_path) as pil_img:
                results['has_metadata'] = True
                
                # Extract EXIF
                exif = pil_img._getexif()
                if exif:
                    # Check for software
                    software_tags = [0x0131, 0x013C]  # Software, HostComputer
                    for tag in software_tags:
                        if tag in exif:
                            software = exif[tag]
                            results['software'] = software
                            
                            # Check for suspicious software
                            suspicious = ['photoshop', 'gimp', 'paint', 'editor']
                            if any(s in software.lower() for s in suspicious):
                                results['suspicious_software'] = software
                
                # Check format
                results['format'] = pil_img.format
                
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        return results
    
    def _perform_consistency_checks(
        self,
        image: np.ndarray
    ) -> List[Tuple[str, bool, str]]:
        """Perform consistency checks on image."""
        checks = []
        
        # Check 1: Color consistency
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:, :, 0])
        color_consistent = hue_std < 60  # Not too much variation
        checks.append((
            "color_consistency",
            color_consistent,
            "Inconsistent color distribution detected" if not color_consistent else ""
        ))
        
        # Check 2: Noise consistency
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Divide into regions and compare noise levels
        h, w = gray.shape
        regions = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:]
        ]
        
        noise_levels = [np.std(r) for r in regions]
        noise_consistent = max(noise_levels) - min(noise_levels) < 20
        checks.append((
            "noise_consistency",
            noise_consistent,
            "Inconsistent noise levels across image" if not noise_consistent else ""
        ))
        
        # Check 3: Compression artifacts
        # Check for JPEG blocking artifacts
        block_size = 8
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        if h_blocks > 0 and w_blocks > 0:
            block_variances = []
            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[
                        i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size
                    ]
                    block_variances.append(np.var(block))
            
            # Check for periodic patterns (blocking artifacts)
            variance_std = np.std(block_variances)
            low_artifacts = variance_std < 500
            checks.append((
                "compression_artifacts",
                low_artifacts,
                "Suspicious compression artifacts detected" if not low_artifacts else ""
            ))
        
        return [c for c in checks if c[2]]  # Return only failed checks with messages
    
    def _calculate_overall_result(
        self,
        ela_result: Optional[ELAResult],
        copy_move_result: Optional[Dict],
        inconsistencies: List[str]
    ) -> Tuple[bool, float]:
        """Calculate overall tampering result."""
        scores = []
        
        # ELA score
        if ela_result:
            if ela_result.is_tampered:
                scores.append(ela_result.confidence * 0.4)
            else:
                scores.append(0.0)
        else:
            scores.append(0.1)  # Neutral if failed
        
        # Copy-move score
        if copy_move_result:
            if copy_move_result.get('forgery_detected', False):
                scores.append(copy_move_result.get('confidence', 0) * 0.4)
            else:
                scores.append(0.0)
        else:
            scores.append(0.1)
        
        # Inconsistency score
        inconsistency_score = min(0.2, len(inconsistencies) * 0.05)
        scores.append(inconsistency_score)
        
        # Calculate final
        confidence = sum(scores)
        is_tampered = confidence > 0.5
        
        return is_tampered, confidence
    
    def visualize_result(
        self,
        image: np.ndarray,
        result: TamperingResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create comprehensive visualization of tampering analysis.
        
        Args:
            image: Original image
            result: Tampering result
            save_path: Optional path to save
            
        Returns:
            Visualization image
        """
        h, w = image.shape[:2]
        
        # Create multi-panel visualization
        panel_height = h
        panel_width = w
        
        # Layout: Original | ELA | Copy-Move | Summary
        canvas_width = panel_width * 3
        canvas_height = panel_height + 200  # Extra space for summary
        
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Panel 1: Original
        canvas[:h, :w] = image
        cv2.putText(
            canvas,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Panel 2: ELA
        if result.ela_result:
            ela_vis = self.ela_analyzer.visualize_result(image, result.ela_result)
            ela_panel = ela_vis[:h, w:]  # Take right half (ELA)
            ela_resized = cv2.resize(ela_panel, (w, h))
            canvas[:h, w:2*w] = ela_resized
        else:
            canvas[:h, w:2*w] = 128
            cv2.putText(
                canvas,
                "ELA Failed",
                (w + 50, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # Panel 3: Copy-move
        if result.copy_move_result:
            cm_vis = self.copy_move_detector.visualize_result(
                image, result.copy_move_result
            )
            cm_resized = cv2.resize(cm_vis, (w, h))
            canvas[:h, 2*w:] = cm_resized
        else:
            canvas[:h, 2*w:] = 128
            cv2.putText(
                canvas,
                "Copy-Move Failed",
                (2*w + 50, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # Summary panel
        y_offset = h + 40
        
        # Status
        status = "TAMPERED" if result.is_tampered else "AUTHENTIC"
        status_color = (0, 0, 255) if result.is_tampered else (0, 255, 0)
        
        cv2.putText(
            canvas,
            f"Status: {status} (Confidence: {result.overall_confidence:.2f})",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )
        
        y_offset += 40
        
        # Inconsistencies
        if result.inconsistencies:
            cv2.putText(
                canvas,
                "Issues Found:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 100, 255),
                2
            )
            y_offset += 30
            
            for issue in result.inconsistencies[:3]:
                cv2.putText(
                    canvas,
                    f"  - {issue[:60]}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 100, 255),
                    1
                )
                y_offset += 25
        
        if save_path is not None:
            cv2.imwrite(save_path, canvas)
        
        return canvas


if __name__ == "__main__":
    # Test tampering pipeline
    print("Tampering Pipeline Test")
    print("=" * 50)
    
    pipeline = TamperingPipeline()
    print("Pipeline initialized!")
    
    # Create test image with tampering
    cv2 = __import__('cv2')
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Add content
    cv2.rectangle(image, (50, 50), (150, 150), (150, 150, 150), -1)
    cv2.circle(image, (300, 200), 50, (100, 100, 100), -1)
    
    # Add copy-move forgery
    pattern = np.random.randint(100, 200, (40, 40, 3), dtype=np.uint8)
    image[100:140, 100:140] = pattern
    image[250:290, 400:440] = pattern
    
    # Add ELA tampering (inconsistent compression)
    tampered = image[300:380, 200:400].copy()
    _, buf = cv2.imencode('.jpg', tampered, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    tampered = cv2.imdecode(buf, 1)
    image[300:380, 200:400] = tampered
    
    # Analyze
    result = pipeline.analyze(image)
    
    print(f"\nAnalysis Results:")
    print(f"  Is Tampered: {result.is_tampered}")
    print(f"  Overall Confidence: {result.overall_confidence:.2f}")
    print(f"  ELA: {result.ela_result.is_tampered if result.ela_result else 'Failed'}")
    print(f"  Copy-Move: {result.copy_move_result.get('forgery_detected', False) if result.copy_move_result else 'Failed'}")
    print(f"  Inconsistencies: {len(result.inconsistencies)}")
    
    for issue in result.inconsistencies:
        print(f"    - {issue}")
    
    print("\nTest completed!")
