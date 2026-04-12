"""Hologram detection for ID card security verification."""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HologramType(Enum):
    """Types of hologram patterns."""
    UNKNOWN = "unknown"
    DOT_MATRIX = "dot_matrix"
    TRUE_COLOR = "true_color"
    KINEGRAM = "kinegram"
    GUOCHES = "guoches"


@dataclass
class HologramRegion:
    """Detected hologram region."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    pattern_type: HologramType
    color_variance: float
    iridescence_score: float


class HologramDetector:
    """
    Detector for holographic security features on ID cards.
    
    Holograms on ID cards typically exhibit:
    - High color variance (rainbow effect)
    - Iridescence (color changes with angle)
    - Specific geometric patterns
    - High frequency texture
    
    This detector uses image analysis to identify potential hologram regions.
    """
    
    def __init__(
        self,
        min_region_size: int = 50,
        color_variance_threshold: float = 30.0,
        iridescence_threshold: float = 0.3,
    ):
        """
        Initialize hologram detector.
        
        Args:
            min_region_size: Minimum hologram region size in pixels
            color_variance_threshold: Minimum color variance for hologram
            iridescence_threshold: Minimum iridescence score
        """
        self.min_region_size = min_region_size
        self.color_variance_threshold = color_variance_threshold
        self.iridescence_threshold = iridescence_threshold
    
    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[HologramRegion]:
        """
        Detect hologram regions in image.
        
        Args:
            image: Input image
            roi: Region of interest (x1, y1, x2, y2)
            
        Returns:
            List of detected hologram regions
        """
        # Extract ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            img = image
            offset_x, offset_y = 0, 0
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Detect high saturation regions (holograms are colorful)
        saturation_mask = s > 80
        
        # Detect high value regions (holograms are reflective)
        value_mask = v > 100
        
        # Combine masks
        hologram_mask = saturation_mask & value_mask
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hologram_mask = cv2.morphologyEx(hologram_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        hologram_mask = cv2.morphologyEx(hologram_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(hologram_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < self.min_region_size * self.min_region_size:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region
            region = img[y:y+h, x:x+w]
            if region.size == 0:
                continue
            
            # Analyze region
            color_variance = self._calculate_color_variance(region)
            iridescence = self._calculate_iridescence(region)
            
            # Check thresholds
            if color_variance < self.color_variance_threshold:
                continue
            
            # Calculate confidence
            confidence = self._calculate_confidence(color_variance, iridescence, area)
            
            # Determine pattern type
            pattern_type = self._classify_pattern(region)
            
            regions.append(HologramRegion(
                bbox=(x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y),
                confidence=confidence,
                pattern_type=pattern_type,
                color_variance=color_variance,
                iridescence_score=iridescence
            ))
        
        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        return regions
    
    def _calculate_color_variance(self, region: np.ndarray) -> float:
        """
        Calculate color variance in region.
        
        Holograms typically have high color variance due to rainbow effect.
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Variance in hue indicates color changes
        hue_variance = np.var(h)
        sat_variance = np.var(s)
        
        # Combined variance score
        return float(hue_variance + sat_variance * 0.5)
    
    def _calculate_iridescence(self, region: np.ndarray) -> float:
        """
        Calculate iridescence score.
        
        Iridescence is the phenomenon of colors changing based on viewing angle.
        We approximate this by looking for smooth color gradients.
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        
        # Calculate gradient
        grad_x = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradient with smooth transitions indicates iridescence
        iridescence = np.mean(gradient_magnitude) / 255.0
        
        return float(min(1.0, iridescence * 2))
    
    def _calculate_confidence(
        self,
        color_variance: float,
        iridescence: float,
        area: float
    ) -> float:
        """Calculate overall confidence score."""
        # Normalize color variance (typical range 0-100)
        color_score = min(1.0, color_variance / 100.0)
        
        # Area score (larger regions more likely to be holograms)
        area_score = min(1.0, area / (100 * 100))
        
        # Weighted combination
        confidence = (
            color_score * 0.4 +
            iridescence * 0.4 +
            area_score * 0.2
        )
        
        return float(confidence)
    
    def _classify_pattern(self, region: np.ndarray) -> HologramType:
        """Classify hologram pattern type."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # High frequency content indicates dot matrix pattern
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Calculate high frequency ratio
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[center_y-10:center_y+10, center_x-10:center_x+10] = 1
        high_freq_mask = 1 - high_freq_mask
        
        high_freq_energy = np.sum(magnitude * high_freq_mask)
        total_energy = np.sum(magnitude)
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
        else:
            high_freq_ratio = 0
        
        # Classify based on frequency content
        if high_freq_ratio > 0.7:
            return HologramType.DOT_MATRIX
        elif high_freq_ratio > 0.5:
            return HologramType.KINEGRAM
        else:
            return HologramType.TRUE_COLOR
    
    def verify_authenticity(
        self,
        image: np.ndarray,
        expected_regions: int = 1
    ) -> Tuple[bool, Dict]:
        """
        Verify hologram authenticity.
        
        Args:
            image: Input image
            expected_regions: Expected number of hologram regions
            
        Returns:
            Tuple of (is_authentic, details)
        """
        regions = self.detect(image)
        
        details = {
            'detected_regions': len(regions),
            'regions': [],
            'overall_confidence': 0.0,
            'checks': {}
        }
        
        # Check 1: Number of regions
        region_check = len(regions) >= expected_regions
        details['checks']['region_count'] = {
            'passed': region_check,
            'expected': expected_regions,
            'found': len(regions)
        }
        
        # Check 2: Confidence scores
        if regions:
            avg_confidence = np.mean([r.confidence for r in regions])
            confidence_check = avg_confidence > 0.5
            details['overall_confidence'] = float(avg_confidence)
        else:
            confidence_check = False
            details['overall_confidence'] = 0.0
        
        details['checks']['confidence'] = {
            'passed': confidence_check,
            'threshold': 0.5,
            'actual': details['overall_confidence']
        }
        
        # Check 3: Color variance
        if regions:
            avg_variance = np.mean([r.color_variance for r in regions])
            variance_check = avg_variance > self.color_variance_threshold
        else:
            variance_check = False
            avg_variance = 0.0
        
        details['checks']['color_variance'] = {
            'passed': variance_check,
            'threshold': self.color_variance_threshold,
            'actual': float(avg_variance)
        }
        
        # Store region details
        for region in regions:
            details['regions'].append({
                'bbox': region.bbox,
                'confidence': region.confidence,
                'type': region.pattern_type.value,
                'color_variance': region.color_variance,
                'iridescence': region.iridescence_score
            })
        
        # Overall authenticity
        is_authentic = region_check and confidence_check and variance_check
        
        return is_authentic, details
    
    def visualize_detection(
        self,
        image: np.ndarray,
        regions: List[HologramRegion],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize hologram detection results.
        
        Args:
            image: Original image
            regions: Detected hologram regions
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            
            # Color based on confidence
            if region.confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif region.confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"Holo {i+1}: {region.confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Add type info
            type_label = f"Type: {region.pattern_type.value}"
            cv2.putText(
                vis_image,
                type_label,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test hologram detector
    print("Hologram Detector Test")
    print("=" * 50)
    
    detector = HologramDetector()
    print("Detector initialized!")
    
    # Create synthetic image with hologram-like region
    image = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Create hologram-like region (high saturation, colorful)
    y, x = np.ogrid[100:200, 100:200]
    
    # Rainbow gradient
    hue = ((x - 100) / 100 * 180).astype(np.uint8)
    sat = np.full((100, 100), 255, dtype=np.uint8)
    val = np.full((100, 100), 200, dtype=np.uint8)
    
    hsv_region = cv2.merge([hue, sat, val])
    bgr_region = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR)
    
    image[100:200, 100:200] = bgr_region
    
    # Detect
    regions = detector.detect(image)
    print(f"Detected {len(regions)} hologram region(s)")
    
    for i, region in enumerate(regions):
        print(f"  Region {i+1}:")
        print(f"    Bbox: {region.bbox}")
        print(f"    Confidence: {region.confidence:.2f}")
        print(f"    Type: {region.pattern_type.value}")
        print(f"    Color variance: {region.color_variance:.1f}")
    
    # Verify authenticity
    is_auth, details = detector.verify_authenticity(image)
    print(f"\nAuthenticity: {is_auth}")
    print(f"Overall confidence: {details['overall_confidence']:.2f}")
    
    print("\nTest completed!")
