"""Complete MRZ detection and recognition pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from .mrz_detector import MRZDetector
from .mrz_recognizer import MRZRecognizer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.mrz_utils import parse_mrz, validate_mrz, MRZData


@dataclass
class MRZResult:
    """Complete MRZ processing result."""
    success: bool
    mrz_data: Optional[MRZData]
    raw_lines: Optional[List[str]]
    confidences: Optional[List[float]]
    region_bbox: Optional[Tuple[int, int, int, int]]
    error_message: str = ""


class MRZPipeline:
    """
    Complete MRZ processing pipeline.
    
    Steps:
    1. Detect MRZ region in ID card
    2. Recognize text in MRZ region
    3. Parse and validate MRZ data
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        min_confidence: float = 0.7,
    ):
        """
        Initialize MRZ pipeline.
        
        Args:
            use_gpu: Use GPU for inference
            min_confidence: Minimum confidence for recognition
        """
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        
        # Initialize components
        self.detector = MRZDetector(use_gpu=use_gpu)
        self.recognizer = MRZRecognizer(use_gpu=use_gpu)
    
    def process(
        self,
        image: np.ndarray,
        document_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> MRZResult:
        """
        Process image to extract MRZ.
        
        Args:
            image: Input ID card image
            document_bbox: Optional document bounding box
            
        Returns:
            MRZResult with extracted data
        """
        try:
            # Step 1: Detect MRZ region
            mrz_bbox = self.detector.detect_mrz_region(image, document_bbox)
            
            if mrz_bbox is None:
                return MRZResult(
                    success=False,
                    mrz_data=None,
                    raw_lines=None,
                    confidences=None,
                    region_bbox=None,
                    error_message="MRZ region not detected"
                )
            
            # Step 2: Crop MRZ region
            x1, y1, x2, y2 = mrz_bbox
            mrz_image = image[y1:y2, x1:x2]
            
            # Step 3: Recognize MRZ text
            raw_lines, confidences = self.recognizer.recognize_mrz(mrz_image)
            
            if raw_lines is None or len(raw_lines) != 3:
                return MRZResult(
                    success=False,
                    mrz_data=None,
                    raw_lines=raw_lines,
                    confidences=confidences,
                    region_bbox=mrz_bbox,
                    error_message=f"Expected 3 MRZ lines, got {len(raw_lines) if raw_lines else 0}"
                )
            
            # Step 4: Validate MRZ format
            is_valid, error_msg = self.recognizer.validate_mrz_format(raw_lines)
            
            if not is_valid:
                return MRZResult(
                    success=False,
                    mrz_data=None,
                    raw_lines=raw_lines,
                    confidences=confidences,
                    region_bbox=mrz_bbox,
                    error_message=f"Invalid MRZ format: {error_msg}"
                )
            
            # Step 5: Parse MRZ data
            mrz_data = parse_mrz(raw_lines)
            
            if mrz_data is None:
                return MRZResult(
                    success=False,
                    mrz_data=None,
                    raw_lines=raw_lines,
                    confidences=confidences,
                    region_bbox=mrz_bbox,
                    error_message="Failed to parse MRZ data"
                )
            
            # Step 6: Validate check digits
            is_valid, validation = validate_mrz(raw_lines)
            mrz_data.valid = is_valid
            
            if not is_valid:
                return MRZResult(
                    success=False,
                    mrz_data=mrz_data,
                    raw_lines=raw_lines,
                    confidences=confidences,
                    region_bbox=mrz_bbox,
                    error_message=f"MRZ check digit validation failed: {validation}"
                )
            
            # Check minimum confidence
            if confidences and min(confidences) < self.min_confidence:
                return MRZResult(
                    success=False,
                    mrz_data=mrz_data,
                    raw_lines=raw_lines,
                    confidences=confidences,
                    region_bbox=mrz_bbox,
                    error_message=f"Low confidence: {min(confidences):.2f} < {self.min_confidence}"
                )
            
            return MRZResult(
                success=True,
                mrz_data=mrz_data,
                raw_lines=raw_lines,
                confidences=confidences,
                region_bbox=mrz_bbox,
                error_message=""
            )
            
        except Exception as e:
            return MRZResult(
                success=False,
                mrz_data=None,
                raw_lines=None,
                confidences=None,
                region_bbox=None,
                error_message=f"Processing error: {str(e)}"
            )
    
    def process_file(self, image_path: str) -> MRZResult:
        """
        Process image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            MRZResult
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return MRZResult(
                success=False,
                mrz_data=None,
                raw_lines=None,
                confidences=None,
                region_bbox=None,
                error_message=f"Cannot load image: {image_path}"
            )
        
        return self.process(image)
    
    def visualize_result(
        self,
        image: np.ndarray,
        result: MRZResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize MRZ detection result.
        
        Args:
            image: Original image
            result: MRZ processing result
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw MRZ region
        if result.region_bbox is not None:
            x1, y1, x2, y2 = result.region_bbox
            color = (0, 255, 0) if result.success else (0, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = "MRZ" if result.success else "MRZ (Failed)"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        # Add MRZ text
        if result.raw_lines is not None:
            y_offset = 30
            for i, line in enumerate(result.raw_lines):
                text = f"Line {i+1}: {line}"
                cv2.putText(
                    vis_image,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_offset += 30
        
        # Add parsed data
        if result.mrz_data is not None:
            y_offset += 20
            info_lines = [
                f"Doc: {result.mrz_data.document_number}",
                f"DOB: {result.mrz_data.date_of_birth}",
                f"Exp: {result.mrz_data.date_of_expiry}",
                f"Name: {result.mrz_data.surname} {result.mrz_data.given_names}",
                f"Valid: {result.mrz_data.valid}",
            ]
            
            for line in info_lines:
                cv2.putText(
                    vis_image,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                y_offset += 25
        
        # Add error message if failed
        if not result.success and result.error_message:
            y_offset += 20
            cv2.putText(
                vis_image,
                f"Error: {result.error_message[:50]}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        # Save if path provided
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test pipeline
    print("MRZ Pipeline Test")
    print("=" * 50)
    
    try:
        pipeline = MRZPipeline()
        print("Pipeline initialized successfully!")
        print("\nTo test with an image:")
        print("  result = pipeline.process_file('path/to/id_card.jpg')")
        print("  print(result.mrz_data)")
    except ImportError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
