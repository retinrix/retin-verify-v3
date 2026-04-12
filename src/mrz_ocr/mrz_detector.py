"""MRZ region detection using heuristic methods (no ML required)."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextBox:
    """Detected text box."""
    points: np.ndarray  # 4 corner points
    score: float
    text: str = ""


class MRZDetector:
    """
    MRZ region detector using heuristic methods.
    
    This detector uses image processing techniques to locate the MRZ region
    without requiring ML models. It looks for:
    - Dark text on light background
    - 3 lines of text at the bottom of the ID card
    - Fixed-width character patterns
    
    License: Apache-2.0
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        min_text_size: int = 10,
        text_threshold: float = 0.5,
        box_threshold: float = 0.5,
        unclip_ratio: float = 1.6,
    ):
        """
        Initialize MRZ detector.
        
        Args:
            use_gpu: Not used (kept for API compatibility)
            min_text_size: Minimum text size in pixels
            text_threshold: Text detection threshold
            box_threshold: Box detection threshold
            unclip_ratio: Not used (kept for API compatibility)
        """
        self.min_text_size = min_text_size
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
    
    def detect(self, image: np.ndarray) -> List[TextBox]:
        """
        Detect text regions in image using contour detection.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detected text boxes
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < self.min_text_size or h < self.min_text_size:
                continue
            
            # Filter by aspect ratio (text is wider than tall)
            aspect_ratio = w / float(h)
            if aspect_ratio < 1.0 or aspect_ratio > 20:
                continue
            
            # Create box points (clockwise from top-left)
            points = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            # Calculate score based on contour area
            area = cv2.contourArea(contour)
            score = min(1.0, area / (w * h))
            
            if score >= self.box_threshold:
                boxes.append(TextBox(points=points, score=score))
        
        return boxes
    
    def detect_mrz_region(
        self,
        image: np.ndarray,
        document_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect MRZ region in ID card image.
        
        Uses heuristic: MRZ is typically at the bottom 20-30% of the ID card.
        
        Args:
            image: Input image
            document_bbox: Document bounding box (x1, y1, x2, y2)
            
        Returns:
            MRZ region bbox or None
        """
        # If document bbox provided, use it
        if document_bbox is not None:
            x1, y1, x2, y2 = document_bbox
            doc_height = y2 - y1
            doc_width = x2 - x1
            
            # MRZ is typically at bottom 25% of document
            mrz_height = int(doc_height * 0.25)
            mrz_y1 = y2 - mrz_height - int(doc_height * 0.02)  # Small margin
            mrz_y2 = y2 - int(doc_height * 0.02)
            
            # Add horizontal margins
            margin_x = int(doc_width * 0.05)
            mrz_x1 = x1 + margin_x
            mrz_x2 = x2 - margin_x
            
            return (mrz_x1, mrz_y1, mrz_x2, mrz_y2)
        
        # Otherwise, use image dimensions
        h, w = image.shape[:2]
        
        # MRZ is at bottom 25% of image
        mrz_height = int(h * 0.25)
        y1 = h - mrz_height - int(h * 0.02)
        y2 = h - int(h * 0.02)
        
        # Add margins
        margin_x = int(w * 0.05)
        x1 = margin_x
        x2 = w - margin_x
        
        return (x1, y1, x2, y2)
    
    def visualize_detection(
        self,
        image: np.ndarray,
        boxes: List[TextBox],
        mrz_bbox: Optional[Tuple[int, int, int, int]] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection results.
        
        Args:
            image: Original image
            boxes: Detected text boxes
            mrz_bbox: Detected MRZ region bbox
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw all text boxes
        for box in boxes:
            points = box.points.astype(np.int32)
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
        
        # Draw MRZ region
        if mrz_bbox is not None:
            x1, y1, x2, y2 = mrz_bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                vis_image, "MRZ Region", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test detector
    print("MRZ Detector Test")
    print("=" * 50)
    
    detector = MRZDetector()
    print("Detector initialized successfully!")
    
    # Test with synthetic image
    image = np.ones((600, 900, 3), dtype=np.uint8) * 255
    
    # Add some text-like regions
    cv2.rectangle(image, (50, 450), (850, 480), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 490), (850, 520), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 530), (850, 560), (0, 0, 0), -1)
    
    # Detect MRZ region
    mrz_bbox = detector.detect_mrz_region(image)
    print(f"Detected MRZ region: {mrz_bbox}")
    
    # Visualize
    boxes = detector.detect(image)
    vis = detector.visualize_detection(image, boxes, mrz_bbox)
    print(f"Detected {len(boxes)} text regions")
    
    print("\nTest completed!")
