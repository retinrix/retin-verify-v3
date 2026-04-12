"""Complete security feature analyzer for ID card verification."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .hologram_detector import HologramDetector, HologramRegion
from .laser_detector import LaserDetector, LaserRegion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityResult:
    """Complete security analysis result."""
    is_authentic: bool
    overall_score: float
    hologram_result: Dict
    laser_result: Dict
    print_quality: Dict
    error_checks: Dict
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'is_authentic': self.is_authentic,
            'overall_score': self.overall_score,
            'hologram': self.hologram_result,
            'laser': self.laser_result,
            'print_quality': self.print_quality,
            'error_checks': self.error_checks,
            'warnings': self.warnings
        }


class SecurityAnalyzer:
    """
    Complete security feature analyzer for Algerian ID cards.
    
    Analyzes:
    1. Hologram authenticity
    2. Laser-engraved features
    3. Print quality
    4. Error level analysis (ELA)
    5. Copy-move detection
    
    This provides a comprehensive security assessment of the ID card.
    """
    
    def __init__(
        self,
        hologram_detector: Optional[HologramDetector] = None,
        laser_detector: Optional[LaserDetector] = None,
    ):
        """
        Initialize security analyzer.
        
        Args:
            hologram_detector: Hologram detector instance
            laser_detector: Laser detector instance
        """
        self.hologram_detector = hologram_detector or HologramDetector()
        self.laser_detector = laser_detector or LaserDetector()
    
    def analyze(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> SecurityResult:
        """
        Perform complete security analysis.
        
        Args:
            image: Input ID card image
            face_bbox: Face region bbox (if known)
            
        Returns:
            Security analysis result
        """
        warnings = []
        
        # 1. Hologram analysis
        try:
            hologram_auth, hologram_details = self.hologram_detector.verify_authenticity(image)
        except Exception as e:
            logger.warning(f"Hologram analysis failed: {e}")
            hologram_auth = False
            hologram_details = {'error': str(e)}
            warnings.append("Hologram analysis failed")
        
        # 2. Laser feature analysis
        try:
            laser_auth, laser_details = self.laser_detector.verify_photo_authenticity(image, face_bbox)
        except Exception as e:
            logger.warning(f"Laser analysis failed: {e}")
            laser_auth = False
            laser_details = {'error': str(e)}
            warnings.append("Laser analysis failed")
        
        # 3. Print quality analysis
        try:
            print_quality = self._analyze_print_quality(image)
        except Exception as e:
            logger.warning(f"Print quality analysis failed: {e}")
            print_quality = {'error': str(e)}
            warnings.append("Print quality analysis failed")
        
        # 4. Error level analysis
        try:
            error_checks = self._error_level_analysis(image)
        except Exception as e:
            logger.warning(f"Error level analysis failed: {e}")
            error_checks = {'error': str(e)}
            warnings.append("Error level analysis failed")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            hologram_auth, hologram_details,
            laser_auth, laser_details,
            print_quality, error_checks
        )
        
        # Determine authenticity
        is_authentic = self._determine_authenticity(
            hologram_auth, laser_auth, print_quality, error_checks
        )
        
        return SecurityResult(
            is_authentic=is_authentic,
            overall_score=overall_score,
            hologram_result={
                'authentic': hologram_auth,
                'details': hologram_details
            },
            laser_result={
                'authentic': laser_auth,
                'details': laser_details
            },
            print_quality=print_quality,
            error_checks=error_checks,
            warnings=warnings
        )
    
    def _analyze_print_quality(self, image: np.ndarray) -> Dict:
        """
        Analyze print quality characteristics.
        
        Checks:
        - Resolution
        - Sharpness
        - Color consistency
        - Compression artifacts
        """
        results = {}
        
        # Resolution check
        h, w = image.shape[:2]
        results['resolution'] = {'width': w, 'height': h}
        results['resolution_check'] = w >= 600 and h >= 400
        
        # Sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        results['sharpness'] = float(sharpness)
        results['sharpness_check'] = sharpness > 100
        
        # Color consistency
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:, :, 0])
        results['color_consistency'] = float(hue_std)
        results['color_check'] = hue_std < 50  # Not too much variation
        
        # Compression artifacts (blocking)
        # Convert to YUV and check DCT coefficients
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        
        # Check for 8x8 blocking artifacts
        block_size = 8
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        block_variances = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = y_channel[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                block_variances.append(np.var(block))
        
        # High variance between blocks indicates compression artifacts
        variance_of_variances = np.var(block_variances)
        results['compression_score'] = float(variance_of_variances)
        results['compression_check'] = variance_of_variances < 1000
        
        return results
    
    def _error_level_analysis(self, image: np.ndarray) -> Dict:
        """
        Perform Error Level Analysis (ELA).
        
        ELA helps detect image manipulation by comparing the original
        image with a recompressed version.
        """
        results = {}
        
        # Save and reload at specific quality to simulate recompression
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encoded = cv2.imencode('.jpg', image, encode_params)
        recompressed = cv2.imdecode(encoded, 1)
        
        # Calculate difference
        diff = cv2.absdiff(image, recompressed)
        
        # Convert to grayscale for analysis
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Statistics
        results['mean_error'] = float(np.mean(diff_gray))
        results['max_error'] = float(np.max(diff_gray))
        results['std_error'] = float(np.std(diff_gray))
        
        # Error distribution
        _, thresh = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        error_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        results['error_ratio'] = float(error_pixels / total_pixels)
        
        # Check for suspicious patterns
        # High error in specific regions may indicate tampering
        results['suspicious'] = results['error_ratio'] > 0.1
        
        # Store ELA image for visualization
        results['ela_image'] = diff
        
        return results
    
    def detect_copy_move(
        self,
        image: np.ndarray,
        block_size: int = 8,
        threshold: float = 0.9
    ) -> Dict:
        """
        Detect copy-move forgery.
        
        Copy-move forgery detection using block matching.
        
        Args:
            image: Input image
            block_size: Size of blocks to compare
            threshold: Similarity threshold
            
        Returns:
            Detection results
        """
        results = {
            'forgery_detected': False,
            'suspicious_regions': [],
            'similarity_map': None
        }
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Extract blocks
        blocks = []
        positions = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                
                # Normalize block
                block = block.astype(np.float32)
                block = (block - np.mean(block)) / (np.std(block) + 1e-7)
                
                blocks.append(block.flatten())
                positions.append((i, j))
        
        if len(blocks) < 2:
            return results
        
        # Compare blocks (simplified - in production use more efficient method)
        blocks_array = np.array(blocks)
        n_blocks = len(blocks)
        
        # Calculate pairwise similarities (upper triangle only)
        similar_pairs = []
        
        for i in range(n_blocks):
            for j in range(i + 10, n_blocks):  # Skip nearby blocks
                similarity = np.dot(blocks_array[i], blocks_array[j])
                
                if similarity > threshold:
                    similar_pairs.append((i, j, similarity))
        
        # Check for suspicious patterns
        if len(similar_pairs) > 5:
            results['forgery_detected'] = True
            
            # Get suspicious regions
            for i, j, sim in similar_pairs[:10]:  # Top 10
                y1, x1 = positions[i]
                y2, x2 = positions[j]
                
                results['suspicious_regions'].append({
                    'region1': (x1, y1, x1 + block_size, y1 + block_size),
                    'region2': (x2, y2, x2 + block_size, y2 + block_size),
                    'similarity': float(sim)
                })
        
        return results
    
    def _calculate_overall_score(
        self,
        hologram_auth: bool,
        hologram_details: Dict,
        laser_auth: bool,
        laser_details: Dict,
        print_quality: Dict,
        error_checks: Dict
    ) -> float:
        """Calculate overall security score."""
        scores = []
        
        # Hologram score
        if 'overall_confidence' in hologram_details:
            scores.append(hologram_details['overall_confidence'] * 0.25)
        elif hologram_auth:
            scores.append(0.25)
        
        # Laser score
        if 'overall_confidence' in laser_details:
            scores.append(laser_details['overall_confidence'] * 0.25)
        elif laser_auth:
            scores.append(0.25)
        
        # Print quality score
        quality_checks = [
            print_quality.get('resolution_check', False),
            print_quality.get('sharpness_check', False),
            print_quality.get('color_check', False),
            print_quality.get('compression_check', False)
        ]
        quality_score = sum(quality_checks) / len(quality_checks) * 0.25
        scores.append(quality_score)
        
        # Error check score
        if 'suspicious' in error_checks:
            error_score = 0.25 if not error_checks['suspicious'] else 0.0
            scores.append(error_score)
        else:
            scores.append(0.125)  # Neutral if check failed
        
        return float(sum(scores))
    
    def _determine_authenticity(
        self,
        hologram_auth: bool,
        laser_auth: bool,
        print_quality: Dict,
        error_checks: Dict
    ) -> bool:
        """Determine overall authenticity."""
        # Must have at least one security feature
        has_security_feature = hologram_auth or laser_auth
        
        # Print quality must be acceptable
        quality_ok = (
            print_quality.get('resolution_check', True) and
            print_quality.get('sharpness_check', True)
        )
        
        # No suspicious error patterns
        no_tampering = not error_checks.get('suspicious', False)
        
        return has_security_feature and quality_ok and no_tampering
    
    def visualize_analysis(
        self,
        image: np.ndarray,
        result: SecurityResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize security analysis results.
        
        Args:
            image: Original image
            result: Security analysis result
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Create summary image
        h, w = image.shape[:2]
        
        # Create canvas
        canvas_h = h + 200
        canvas = np.ones((canvas_h, w, 3), dtype=np.uint8) * 255
        
        # Place original image
        canvas[:h, :w] = image
        
        # Add text summary
        y_offset = h + 30
        
        # Authenticity status
        status_color = (0, 255, 0) if result.is_authentic else (0, 0, 255)
        status_text = "AUTHENTIC" if result.is_authentic else "SUSPICIOUS"
        cv2.putText(
            canvas,
            f"Status: {status_text} (Score: {result.overall_score:.2f})",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
        
        y_offset += 30
        
        # Hologram status
        holo_status = "✓" if result.hologram_result.get('authentic', False) else "✗"
        cv2.putText(
            canvas,
            f"Hologram: {holo_status}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        
        y_offset += 25
        
        # Laser status
        laser_status = "✓" if result.laser_result.get('authentic', False) else "✗"
        cv2.putText(
            canvas,
            f"Laser: {laser_status}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        
        y_offset += 25
        
        # Print quality
        pq = result.print_quality
        quality_score = sum([
            pq.get('resolution_check', False),
            pq.get('sharpness_check', False),
            pq.get('color_check', False),
            pq.get('compression_check', False)
        ])
        cv2.putText(
            canvas,
            f"Print Quality: {quality_score}/4 checks passed",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        
        # Warnings
        if result.warnings:
            y_offset += 30
            cv2.putText(
                canvas,
                "Warnings:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 100, 255),
                1
            )
            for warning in result.warnings[:3]:  # Show first 3
                y_offset += 20
                cv2.putText(
                    canvas,
                    f"  - {warning[:50]}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 100, 255),
                    1
                )
        
        if save_path is not None:
            cv2.imwrite(save_path, canvas)
        
        return canvas


if __name__ == "__main__":
    # Test security analyzer
    print("Security Analyzer Test")
    print("=" * 50)
    
    analyzer = SecurityAnalyzer()
    print("Analyzer initialized!")
    
    # Create synthetic ID card image
    cv2 = __import__('cv2')
    image = np.ones((600, 900, 3), dtype=np.uint8) * 240
    
    # Add face region
    cv2.ellipse(image, (450, 200), (80, 100), 0, 0, 360, (180, 160, 140), -1)
    
    # Add hologram-like region
    y, x = np.ogrid[50:120, 700:800]
    hue = ((x - 700) / 100 * 180).astype(np.uint8)
    sat = np.full((70, 100), 255, dtype=np.uint8)
    val = np.full((70, 100), 200, dtype=np.uint8)
    hsv = cv2.merge([hue, sat, val])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image[50:120, 700:800] = bgr
    
    # Analyze
    result = analyzer.analyze(image)
    
    print(f"\nAnalysis Results:")
    print(f"  Authentic: {result.is_authentic}")
    print(f"  Overall Score: {result.overall_score:.2f}")
    print(f"  Hologram: {result.hologram_result.get('authentic', False)}")
    print(f"  Laser: {result.laser_result.get('authentic', False)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    print("\nTest completed!")
