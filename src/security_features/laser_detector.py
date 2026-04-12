"""Laser-engraved photo detection for ID card security verification."""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaserFeature(Enum):
    """Types of laser-engraved features."""
    PHOTO = "photo"
    TEXT = "text"
    GUILLOCHE = "guilloche"
    MICROTEXT = "microtext"
    UNKNOWN = "unknown"


@dataclass
class LaserRegion:
    """Detected laser-engraved region."""
    bbox: Tuple[int, int, int, int]
    feature_type: LaserFeature
    confidence: float
    texture_score: float
    edge_score: float
    depth_score: float


class LaserDetector:
    """
    Detector for laser-engraved features on ID cards.
    
    Laser-engraved features on ID cards exhibit:
    - Raised/relief texture (can be detected via shadows)
    - Sharp, precise edges
    - Specific grayscale patterns
    - Tactile characteristics (visible under side lighting)
    
    This detector uses image analysis to identify potential laser-engraved regions.
    """
    
    def __init__(
        self,
        edge_threshold: float = 50.0,
        texture_threshold: float = 0.3,
        depth_threshold: float = 0.2,
    ):
        """
        Initialize laser detector.
        
        Args:
            edge_threshold: Edge detection threshold
            texture_threshold: Texture analysis threshold
            depth_threshold: Depth/relief detection threshold
        """
        self.edge_threshold = edge_threshold
        self.texture_threshold = texture_threshold
        self.depth_threshold = depth_threshold
    
    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[LaserRegion]:
        """
        Detect laser-engraved regions in image.
        
        Args:
            image: Input image
            roi: Region of interest (x1, y1, x2, y2)
            
        Returns:
            List of detected laser regions
        """
        # Extract ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            img = image
            offset_x, offset_y = 0, 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect texture using local binary patterns
        texture = self._calculate_lbp(gray)
        
        # Detect relief/depth using gradient analysis
        relief = self._detect_relief(gray)
        
        # Combine features
        laser_mask = self._combine_features(edges, texture, relief)
        
        # Find contours
        contours, _ = cv2.findContours(laser_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum region size
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            region = gray[y:y+h, x:x+w]
            
            if region.size == 0:
                continue
            
            # Analyze region
            edge_score = self._analyze_edges(region)
            texture_score = self._analyze_texture(region)
            depth_score = self._analyze_depth(region)
            
            # Calculate confidence
            confidence = self._calculate_confidence(edge_score, texture_score, depth_score)
            
            # Classify feature type
            feature_type = self._classify_feature(edge_score, texture_score, region.shape)
            
            regions.append(LaserRegion(
                bbox=(x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y),
                feature_type=feature_type,
                confidence=confidence,
                texture_score=texture_score,
                edge_score=edge_score,
                depth_score=depth_score
            ))
        
        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        return regions
    
    def _calculate_lbp(self, gray: np.ndarray) -> np.ndarray:
        """
        Calculate Local Binary Pattern for texture analysis.
        
        Laser engraving creates specific texture patterns.
        """
        h, w = gray.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        # Simple LBP
        center = gray[1:-1, 1:-1]
        
        # 8 neighbors
        neighbors = [
            gray[0:-2, 0:-2],   # top-left
            gray[0:-2, 1:-1],   # top
            gray[0:-2, 2:],     # top-right
            gray[1:-1, 2:],     # right
            gray[2:, 2:],       # bottom-right
            gray[2:, 1:-1],     # bottom
            gray[2:, 0:-2],     # bottom-left
            gray[1:-1, 0:-2],   # left
        ]
        
        for i, neighbor in enumerate(neighbors):
            lbp += (neighbor > center).astype(np.uint8) << i
        
        return lbp
    
    def _detect_relief(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect relief/depth using gradient analysis.
        
        Laser engraving creates raised/lowered surfaces that cast subtle shadows.
        """
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return gradient_magnitude.astype(np.uint8)
    
    def _combine_features(
        self,
        edges: np.ndarray,
        texture: np.ndarray,
        relief: np.ndarray
    ) -> np.ndarray:
        """Combine feature maps into laser detection mask."""
        # Ensure same size
        h, w = edges.shape
        texture = cv2.resize(texture, (w, h))
        relief = cv2.resize(relief, (w, h))
        
        # Normalize
        edges_norm = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
        texture_norm = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        relief_norm = cv2.normalize(relief, None, 0, 1, cv2.NORM_MINMAX)
        
        # Weighted combination
        combined = (
            edges_norm * 0.4 +
            texture_norm * 0.3 +
            relief_norm * 0.3
        )
        
        # Threshold
        _, mask = cv2.threshold(combined, 0.3, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _analyze_edges(self, region: np.ndarray) -> float:
        """Analyze edge characteristics."""
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Laser engraving has sharp, dense edges
        return float(min(1.0, edge_density * 3))
    
    def _analyze_texture(self, region: np.ndarray) -> float:
        """Analyze texture characteristics."""
        lbp = self._calculate_lbp(region)
        
        # Calculate texture uniformity
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Entropy (higher = more textured)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Normalize to 0-1
        texture_score = min(1.0, entropy / 8.0)
        
        return float(texture_score)
    
    def _analyze_depth(self, region: np.ndarray) -> float:
        """Analyze depth/relief characteristics."""
        relief = self._detect_relief(region)
        
        # Measure gradient variance
        depth_score = np.var(relief) / 255.0
        
        return float(min(1.0, depth_score))
    
    def _calculate_confidence(
        self,
        edge_score: float,
        texture_score: float,
        depth_score: float
    ) -> float:
        """Calculate overall confidence."""
        # Weighted combination
        confidence = (
            edge_score * 0.4 +
            texture_score * 0.3 +
            depth_score * 0.3
        )
        
        return float(confidence)
    
    def _classify_feature(
        self,
        edge_score: float,
        texture_score: float,
        shape: Tuple[int, ...]
    ) -> LaserFeature:
        """Classify the type of laser feature."""
        h, w = shape[:2]
        aspect_ratio = w / h if h > 0 else 1
        
        # Photo: moderate aspect ratio, high texture
        if 0.7 < aspect_ratio < 1.3 and texture_score > 0.5:
            return LaserFeature.PHOTO
        
        # Text: high aspect ratio, high edges
        elif aspect_ratio > 3 and edge_score > 0.5:
            return LaserFeature.TEXT
        
        # Guilloche: square-ish, very high texture
        elif 0.9 < aspect_ratio < 1.1 and texture_score > 0.7:
            return LaserFeature.GUILLOCHE
        
        # Microtext: small region, high edge density
        elif max(h, w) < 50 and edge_score > 0.6:
            return LaserFeature.MICROTEXT
        
        else:
            return LaserFeature.UNKNOWN
    
    def verify_photo_authenticity(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[bool, Dict]:
        """
        Verify laser photo authenticity.
        
        Args:
            image: Input image
            face_bbox: Face region bbox (if known)
            
        Returns:
            Tuple of (is_authentic, details)
        """
        # Detect laser regions
        if face_bbox is not None:
            regions = self.detect(image, roi=face_bbox)
        else:
            regions = self.detect(image)
        
        # Filter for photo regions
        photo_regions = [r for r in regions if r.feature_type == LaserFeature.PHOTO]
        
        details = {
            'detected_regions': len(regions),
            'photo_regions': len(photo_regions),
            'overall_confidence': 0.0,
            'checks': {}
        }
        
        # Check 1: Photo region detected
        photo_check = len(photo_regions) > 0
        details['checks']['photo_detected'] = {
            'passed': photo_check,
            'found': len(photo_regions)
        }
        
        # Check 2: Confidence scores
        if photo_regions:
            avg_confidence = np.mean([r.confidence for r in photo_regions])
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
        
        # Check 3: Texture quality
        if photo_regions:
            avg_texture = np.mean([r.texture_score for r in photo_regions])
            texture_check = avg_texture > self.texture_threshold
        else:
            texture_check = False
            avg_texture = 0.0
        
        details['checks']['texture'] = {
            'passed': texture_check,
            'threshold': self.texture_threshold,
            'actual': float(avg_texture)
        }
        
        # Store region details
        details['regions'] = []
        for region in photo_regions:
            details['regions'].append({
                'bbox': region.bbox,
                'type': region.feature_type.value,
                'confidence': region.confidence,
                'texture': region.texture_score,
                'edges': region.edge_score,
                'depth': region.depth_score
            })
        
        # Overall authenticity
        is_authentic = photo_check and confidence_check and texture_check
        
        return is_authentic, details
    
    def visualize_detection(
        self,
        image: np.ndarray,
        regions: List[LaserRegion],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize laser detection results.
        
        Args:
            image: Original image
            regions: Detected laser regions
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Color map for feature types
        color_map = {
            LaserFeature.PHOTO: (0, 255, 0),      # Green
            LaserFeature.TEXT: (255, 0, 0),       # Blue
            LaserFeature.GUILLOCHE: (0, 255, 255), # Cyan
            LaserFeature.MICROTEXT: (255, 0, 255), # Magenta
            LaserFeature.UNKNOWN: (128, 128, 128) # Gray
        }
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            color = color_map.get(region.feature_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{region.feature_type.value}: {region.confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test laser detector
    print("Laser Detector Test")
    print("=" * 50)
    
    detector = LaserDetector()
    print("Detector initialized!")
    
    # Create synthetic image with laser-like region
    image = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Create laser-engraved-like region (high contrast, texture)
    laser_region = np.random.randint(100, 200, (150, 150), dtype=np.uint8)
    
    # Add sharp edges
    cv2 = __import__('cv2')
    cv2.rectangle(laser_region, (20, 20), (130, 130), 50, 2)
    cv2.circle(laser_region, (75, 75), 30, 200, 2)
    
    # Convert to 3-channel
    laser_color = cv2.cvtColor(laser_region, cv2.COLOR_GRAY2BGR)
    image[100:250, 100:250] = laser_color
    
    # Detect
    regions = detector.detect(image)
    print(f"Detected {len(regions)} laser region(s)")
    
    for i, region in enumerate(regions):
        print(f"  Region {i+1}:")
        print(f"    Bbox: {region.bbox}")
        print(f"    Type: {region.feature_type.value}")
        print(f"    Confidence: {region.confidence:.2f}")
        print(f"    Edge score: {region.edge_score:.2f}")
        print(f"    Texture score: {region.texture_score:.2f}")
    
    # Verify photo
    is_auth, details = detector.verify_photo_authenticity(image)
    print(f"\nPhoto authenticity: {is_auth}")
    print(f"Overall confidence: {details['overall_confidence']:.2f}")
    
    print("\nTest completed!")
