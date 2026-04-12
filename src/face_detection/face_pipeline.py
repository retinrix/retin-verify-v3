"""Complete face extraction pipeline for ID card verification."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

from .face_detector import FaceDetector, FaceDetectionResult
from .face_aligner import FaceAligner, AlignedFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceExtractionResult:
    """Complete face extraction result."""
    success: bool
    face_image: Optional[np.ndarray]
    detection: Optional[FaceDetectionResult]
    alignment: Optional[AlignedFace]
    error_message: str = ""
    
    @property
    def face_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get face bounding box."""
        if self.detection:
            return self.detection.bbox
        return None
    
    @property
    def confidence(self) -> float:
        """Get detection confidence."""
        if self.detection:
            return self.detection.confidence
        return 0.0


class FacePipeline:
    """
    Complete face extraction pipeline.
    
    Steps:
    1. Detect face in ID card image
    2. Align face using landmarks
    3. Extract normalized face chip
    4. Apply quality checks
    
    For Algerian ID cards, the face is typically in the upper portion
    of the card, with specific positioning requirements.
    """
    
    def __init__(
        self,
        detector_model_path: Optional[str] = None,
        target_size: Tuple[int, int] = (112, 112),
        min_confidence: float = 0.7,
        min_face_size: int = 80,
    ):
        """
        Initialize face pipeline.
        
        Args:
            detector_model_path: Path to face detection model
            target_size: Target face size for alignment
            min_confidence: Minimum detection confidence
            min_face_size: Minimum face size in pixels
        """
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        
        # Initialize components
        self.detector = FaceDetector(model_path=detector_model_path)
        self.aligner = FaceAligner(target_size=target_size)
    
    def extract_face(
        self,
        image: np.ndarray,
        document_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_region: Optional[Tuple[int, int, int, int]] = None
    ) -> FaceExtractionResult:
        """
        Extract face from ID card image.
        
        Args:
            image: Input ID card image
            document_bbox: Document bounding box (if already detected)
            face_region: Specific region to search for face
            
        Returns:
            Face extraction result
        """
        try:
            # Determine search region
            search_region = self._get_search_region(
                image, document_bbox, face_region
            )
            
            if search_region is None:
                return FaceExtractionResult(
                    success=False,
                    face_image=None,
                    detection=None,
                    alignment=None,
                    error_message="Could not determine face search region"
                )
            
            # Detect faces in region
            faces = self.detector.detect_in_region(
                image, search_region, return_landmarks=True
            )
            
            if not faces:
                return FaceExtractionResult(
                    success=False,
                    face_image=None,
                    detection=None,
                    alignment=None,
                    error_message="No face detected"
                )
            
            # Filter by confidence and size
            valid_faces = self._filter_faces(faces)
            
            if not valid_faces:
                return FaceExtractionResult(
                    success=False,
                    face_image=None,
                    detection=faces[0],
                    alignment=None,
                    error_message=f"Face detected but below thresholds"
                )
            
            # Select best face (largest valid face)
            best_face = max(valid_faces, key=lambda f: f.area)
            
            # Align face
            alignment = self.aligner.align_from_detection(image, best_face)
            
            if alignment is None:
                # Fallback to simple extraction
                face_image = self.aligner.extract_face_chip(
                    image, best_face.bbox
                )
                alignment = None
            else:
                face_image = alignment.image
            
            # Apply quality check
            quality_score = self._check_quality(face_image)
            
            if quality_score < 0.3:
                return FaceExtractionResult(
                    success=False,
                    face_image=face_image,
                    detection=best_face,
                    alignment=alignment,
                    error_message=f"Low quality face image: {quality_score:.2f}"
                )
            
            return FaceExtractionResult(
                success=True,
                face_image=face_image,
                detection=best_face,
                alignment=alignment,
                error_message=""
            )
            
        except Exception as e:
            logger.error(f"Face extraction error: {e}")
            return FaceExtractionResult(
                success=False,
                face_image=None,
                detection=None,
                alignment=None,
                error_message=f"Extraction error: {str(e)}"
            )
    
    def extract_face_from_file(
        self,
        image_path: str,
        document_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> FaceExtractionResult:
        """
        Extract face from image file.
        
        Args:
            image_path: Path to image file
            document_bbox: Document bounding box
            
        Returns:
            Face extraction result
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return FaceExtractionResult(
                success=False,
                face_image=None,
                detection=None,
                alignment=None,
                error_message=f"Cannot load image: {image_path}"
            )
        
        return self.extract_face(image, document_bbox)
    
    def _get_search_region(
        self,
        image: np.ndarray,
        document_bbox: Optional[Tuple[int, int, int, int]],
        face_region: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Determine face search region.
        
        For Algerian ID cards, face is typically in upper portion.
        """
        # If face region specified, use it
        if face_region is not None:
            return face_region
        
        # If document bbox specified, use upper portion
        if document_bbox is not None:
            x1, y1, x2, y2 = document_bbox
            doc_w = x2 - x1
            doc_h = y2 - y1
            
            # Face is typically in upper 60% of ID card
            # and centered horizontally
            face_w = int(doc_w * 0.5)
            face_h = int(doc_h * 0.5)
            
            face_x1 = x1 + (doc_w - face_w) // 2
            face_y1 = y1 + int(doc_h * 0.1)  # Top margin
            face_x2 = face_x1 + face_w
            face_y2 = face_y1 + face_h
            
            return (face_x1, face_y1, face_x2, face_y2)
        
        # Use full image upper portion
        h, w = image.shape[:2]
        return (int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.6))
    
    def _filter_faces(
        self,
        faces: List[FaceDetectionResult]
    ) -> List[FaceDetectionResult]:
        """
        Filter faces by quality criteria.
        
        Args:
            faces: Detected faces
            
        Returns:
            Filtered faces
        """
        valid_faces = []
        
        for face in faces:
            # Check confidence
            if face.confidence < self.min_confidence:
                continue
            
            # Check size
            if face.width < self.min_face_size or face.height < self.min_face_size:
                continue
            
            # Check aspect ratio (face should be roughly oval)
            aspect_ratio = face.width / face.height
            if aspect_ratio < 0.6 or aspect_ratio > 1.0:
                continue
            
            valid_faces.append(face)
        
        return valid_faces
    
    def _check_quality(self, face_image: np.ndarray) -> float:
        """
        Check face image quality.
        
        Args:
            face_image: Face image
            
        Returns:
            Quality score (0-1)
        """
        if face_image is None or face_image.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        scores = []
        
        # Check brightness
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        scores.append(max(0, brightness_score))
        
        # Check contrast
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 50)
        scores.append(contrast_score)
        
        # Check sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 500)
        scores.append(sharpness_score)
        
        # Average scores
        return float(np.mean(scores))
    
    def visualize_result(
        self,
        image: np.ndarray,
        result: FaceExtractionResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize face extraction result.
        
        Args:
            image: Original image
            result: Face extraction result
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw detection
        if result.detection is not None:
            x1, y1, x2, y2 = result.detection.bbox
            color = (0, 255, 0) if result.success else (0, 0, 255)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"Face: {result.detection.confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            # Draw landmarks
            if result.detection.landmarks is not None:
                for (lx, ly) in result.detection.landmarks:
                    cv2.circle(vis_image, (int(lx), int(ly)), 3, (255, 0, 0), -1)
        
        # Add status
        status = "Success" if result.success else f"Failed: {result.error_message[:30]}"
        cv2.putText(
            vis_image,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if result.success else (0, 0, 255),
            2
        )
        
        # Add extracted face inset
        if result.face_image is not None:
            face_h, face_w = result.face_image.shape[:2]
            inset_size = 150
            face_resized = cv2.resize(result.face_image, (inset_size, inset_size))
            
            # Place in top-right corner
            vis_h, vis_w = vis_image.shape[:2]
            x_offset = vis_w - inset_size - 10
            y_offset = 10
            
            vis_image[y_offset:y_offset+inset_size, x_offset:x_offset+inset_size] = face_resized
            
            # Draw border
            cv2.rectangle(
                vis_image,
                (x_offset, y_offset),
                (x_offset + inset_size, y_offset + inset_size),
                (0, 255, 0),
                2
            )
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def compare_faces(
        self,
        face1: np.ndarray,
        face2: np.ndarray,
        method: str = "histogram"
    ) -> float:
        """
        Compare two face images.
        
        Args:
            face1: First face image
            face2: Second face image
            method: Comparison method ('histogram', 'template', 'orb')
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Resize to same size
        size = (112, 112)
        f1 = cv2.resize(face1, size)
        f2 = cv2.resize(face2, size)
        
        if method == "histogram":
            # Histogram comparison
            f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            
            hist1 = cv2.calcHist([f1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([f2_gray], [0], None, [256], [0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0, similarity)
        
        elif method == "template":
            # Template matching
            f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            
            result = cv2.matchTemplate(f1_gray, f2_gray, cv2.TM_CCOEFF_NORMED)
            return float(np.max(result))
        
        elif method == "orb":
            # ORB feature matching
            orb = cv2.ORB_create(nfeatures=100)
            
            f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = orb.detectAndCompute(f1_gray, None)
            kp2, des2 = orb.detectAndCompute(f2_gray, None)
            
            if des1 is None or des2 is None:
                return 0.0
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if not matches:
                return 0.0
            
            # Average distance (lower is better)
            avg_distance = np.mean([m.distance for m in matches])
            similarity = max(0, 1 - avg_distance / 100)
            
            return similarity
        
        else:
            raise ValueError(f"Unknown comparison method: {method}")


if __name__ == "__main__":
    # Test face pipeline
    print("Face Pipeline Test")
    print("=" * 50)
    
    try:
        pipeline = FacePipeline()
        print("Pipeline initialized!")
        
        # Create synthetic ID card with face
        image = np.ones((600, 900, 3), dtype=np.uint8) * 240
        
        # Draw face in upper portion (typical ID layout)
        face_center_x, face_center_y = 450, 200
        cv2.ellipse(
            image,
            (face_center_x, face_center_y),
            (80, 100),
            0, 0, 360,
            (180, 150, 120),
            -1
        )
        
        # Draw eyes
        cv2.circle(image, (face_center_x - 30, face_center_y - 20), 10, (50, 50, 50), -1)
        cv2.circle(image, (face_center_x + 30, face_center_y - 20), 10, (50, 50, 50), -1)
        
        # Draw nose
        cv2.ellipse(image, (face_center_x, face_center_y + 10), (10, 15), 0, 0, 360, (100, 80, 60), -1)
        
        # Draw mouth
        cv2.ellipse(image, (face_center_x, face_center_y + 50), (30, 10), 0, 0, 180, (100, 50, 50), 2)
        
        # Extract face
        result = pipeline.extract_face(image)
        
        print(f"Extraction success: {result.success}")
        if result.success:
            print(f"Face shape: {result.face_image.shape}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Alignment score: {result.alignment.alignment_score:.3f}")
        else:
            print(f"Error: {result.error_message}")
        
        # Visualize
        vis = pipeline.visualize_result(image, result)
        print("Visualization created!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")
