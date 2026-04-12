"""Error Level Analysis (ELA) for tampering detection."""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ELAResult:
    """ELA analysis result."""
    ela_image: np.ndarray
    mean_error: float
    max_error: float
    std_error: float
    error_ratio: float
    suspicious_regions: list
    is_tampered: bool
    confidence: float


class ELAAnalyzer:
    """
    Error Level Analysis for detecting image manipulation.
    
    ELA works by:
    1. Saving the image at a known quality level
    2. Comparing with the original
    3. Analyzing error patterns
    
    Manipulated regions typically show different error levels
    compared to authentic regions.
    
    Reference: http://blackhat.com/presentations/bh-dc-08/Krawetz/Whitepaper/bh-dc-08-krawetz-WP.pdf
    """
    
    def __init__(
        self,
        quality: int = 95,
        error_threshold: int = 30,
        min_region_size: int = 100,
    ):
        """
        Initialize ELA analyzer.
        
        Args:
            quality: JPEG quality for recompression (higher = more sensitive)
            error_threshold: Pixel error threshold for suspicious regions
            min_region_size: Minimum size of suspicious regions
        """
        self.quality = quality
        self.error_threshold = error_threshold
        self.min_region_size = min_region_size
    
    def analyze(self, image: np.ndarray) -> ELAResult:
        """
        Perform ELA on image.
        
        Args:
            image: Input image
            
        Returns:
            ELA analysis result
        """
        # Ensure image is in BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Recompress image
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded = cv2.imencode('.jpg', image, encode_params)
        recompressed = cv2.imdecode(encoded, 1)
        
        # Calculate absolute difference
        diff = cv2.absdiff(image, recompressed)
        
        # Enhance differences for visualization
        ela_image = self._enhance_ela(diff)
        
        # Calculate statistics
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        mean_error = float(np.mean(diff_gray))
        max_error = float(np.max(diff_gray))
        std_error = float(np.std(diff_gray))
        
        # Calculate error ratio
        error_mask = diff_gray > self.error_threshold
        error_pixels = np.sum(error_mask)
        total_pixels = diff_gray.size
        error_ratio = float(error_pixels / total_pixels)
        
        # Detect suspicious regions
        suspicious_regions = self._detect_suspicious_regions(diff_gray)
        
        # Determine if tampered
        is_tampered, confidence = self._assess_tampering(
            error_ratio, std_error, len(suspicious_regions)
        )
        
        return ELAResult(
            ela_image=ela_image,
            mean_error=mean_error,
            max_error=max_error,
            std_error=std_error,
            error_ratio=error_ratio,
            suspicious_regions=suspicious_regions,
            is_tampered=is_tampered,
            confidence=confidence
        )
    
    def _enhance_ela(self, diff: np.ndarray) -> np.ndarray:
        """Enhance ELA image for better visualization."""
        # Scale differences
        enhanced = diff * 15  # Amplify differences
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Apply color map for better visualization
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        colorized = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        return colorized
    
    def _detect_suspicious_regions(
        self,
        diff_gray: np.ndarray
    ) -> list:
        """Detect suspicious regions with high error levels."""
        # Threshold error image
        _, thresh = cv2.threshold(
            diff_gray,
            self.error_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_region_size:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate average error in region
            region = diff_gray[y:y+h, x:x+w]
            avg_error = float(np.mean(region))
            max_error = float(np.max(region))
            
            regions.append({
                'bbox': (x, y, x+w, y+h),
                'area': int(area),
                'avg_error': avg_error,
                'max_error': max_error
            })
        
        # Sort by average error (descending)
        regions.sort(key=lambda r: r['avg_error'], reverse=True)
        
        return regions
    
    def _assess_tampering(
        self,
        error_ratio: float,
        std_error: float,
        num_suspicious_regions: int
    ) -> Tuple[bool, float]:
        """
        Assess whether image is likely tampered.
        
        Returns:
            Tuple of (is_tampered, confidence)
        """
        # Scoring factors
        score = 0.0
        
        # Error ratio factor
        if error_ratio > 0.15:
            score += 0.4
        elif error_ratio > 0.10:
            score += 0.2
        
        # Standard deviation factor (high std indicates localized errors)
        if std_error > 10:
            score += 0.3
        elif std_error > 5:
            score += 0.15
        
        # Suspicious regions factor
        if num_suspicious_regions > 5:
            score += 0.3
        elif num_suspicious_regions > 2:
            score += 0.15
        
        # Determine result
        is_tampered = score > 0.5
        confidence = min(1.0, score)
        
        return is_tampered, confidence
    
    def compare_regions(
        self,
        image: np.ndarray,
        region1: Tuple[int, int, int, int],
        region2: Tuple[int, int, int, int]
    ) -> Dict:
        """
        Compare error levels between two regions.
        
        Args:
            image: Input image
            region1: First region bbox (x1, y1, x2, y2)
            region2: Second region bbox (x1, y1, x2, y2)
            
        Returns:
            Comparison results
        """
        # Perform ELA
        result = self.analyze(image)
        
        # Extract regions from ELA image
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        ela_gray = cv2.cvtColor(result.ela_image, cv2.COLOR_BGR2GRAY)
        
        region1_ela = ela_gray[y1_1:y2_1, x1_1:x2_1]
        region2_ela = ela_gray[y1_2:y2_2, x1_2:x2_2]
        
        # Calculate statistics
        r1_mean = float(np.mean(region1_ela))
        r1_std = float(np.std(region1_ela))
        r2_mean = float(np.mean(region2_ela))
        r2_std = float(np.std(region2_ela))
        
        # Compare
        mean_diff = abs(r1_mean - r2_mean)
        std_diff = abs(r1_std - r2_std)
        
        # Significant difference may indicate manipulation
        significant_difference = mean_diff > 20 or std_diff > 10
        
        return {
            'region1': {'mean': r1_mean, 'std': r1_std},
            'region2': {'mean': r2_mean, 'std': r2_std},
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'significant_difference': significant_difference
        }
    
    def visualize_result(
        self,
        image: np.ndarray,
        result: ELAResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of ELA results.
        
        Args:
            image: Original image
            result: ELA result
            save_path: Optional path to save
            
        Returns:
            Visualization image
        """
        h, w = image.shape[:2]
        
        # Create side-by-side comparison
        vis_width = w * 2
        vis_image = np.ones((h, vis_width, 3), dtype=np.uint8) * 255
        
        # Original image
        vis_image[:h, :w] = image
        
        # ELA image
        ela_resized = cv2.resize(result.ela_image, (w, h))
        vis_image[:h, w:] = ela_resized
        
        # Add labels
        cv2.putText(
            vis_image,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        status = "TAMPERED" if result.is_tampered else "AUTHENTIC"
        status_color = (0, 0, 255) if result.is_tampered else (0, 255, 0)
        
        cv2.putText(
            vis_image,
            f"ELA - {status} ({result.confidence:.2f})",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )
        
        # Add statistics
        stats_text = [
            f"Mean Error: {result.mean_error:.1f}",
            f"Max Error: {result.max_error:.1f}",
            f"Error Ratio: {result.error_ratio:.3f}",
            f"Suspicious Regions: {len(result.suspicious_regions)}"
        ]
        
        y_offset = h - 100
        for text in stats_text:
            cv2.putText(
                vis_image,
                text,
                (w + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 25
        
        # Draw suspicious regions on original
        for region in result.suspicious_regions[:5]:  # Top 5
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test ELA analyzer
    print("ELA Analyzer Test")
    print("=" * 50)
    
    analyzer = ELAAnalyzer()
    print("Analyzer initialized!")
    
    # Create test image
    cv2 = __import__('cv2')
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Add some content
    cv2.rectangle(image, (100, 100), (200, 200), (150, 150, 150), -1)
    cv2.circle(image, (400, 200), 50, (100, 100, 100), -1)
    
    # Simulate tampering by adding inconsistent compression
    tampered_region = image[250:350, 300:500].copy()
    _, buf = cv2.imencode('.jpg', tampered_region, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    tampered_region = cv2.imdecode(buf, 1)
    image[250:350, 300:500] = tampered_region
    
    # Analyze
    result = analyzer.analyze(image)
    
    print(f"\nELA Results:")
    print(f"  Mean Error: {result.mean_error:.2f}")
    print(f"  Max Error: {result.max_error:.2f}")
    print(f"  Error Ratio: {result.error_ratio:.4f}")
    print(f"  Suspicious Regions: {len(result.suspicious_regions)}")
    print(f"  Is Tampered: {result.is_tampered}")
    print(f"  Confidence: {result.confidence:.2f}")
    
    print("\nTest completed!")
