"""Face alignment and preprocessing for ID card photos."""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AlignedFace:
    """Aligned face result."""
    image: np.ndarray
    transformation_matrix: np.ndarray
    original_bbox: Tuple[int, int, int, int]
    alignment_score: float


class FaceAligner:
    """
    Face alignment for ID card photos.
    
    Aligns faces to a standard position using facial landmarks:
    - Eyes horizontally aligned
    - Consistent face size
    - Standardized position
    
    This is important for:
    - Face comparison/verification
    - Consistent face extraction
    - Biometric matching
    """
    
    # Standard face template (5 landmarks)
    # Right eye, Left eye, Nose, Right mouth, Left mouth
    TEMPLATE_LANDMARKS = np.array([
        [38.2946, 51.6963],   # Right eye
        [73.5318, 51.5014],   # Left eye
        [56.0252, 71.7366],   # Nose
        [41.5493, 92.3655],   # Right mouth
        [70.7299, 92.2041],   # Left mouth
    ], dtype=np.float32)
    
    # Target face size
    TARGET_SIZE = (112, 112)
    
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        """
        Initialize face aligner.
        
        Args:
            target_size: Target output size (width, height)
        """
        self.target_size = target_size
        
        # Scale template to target size
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        
        self.template = self.TEMPLATE_LANDMARKS.copy()
        self.template[:, 0] *= scale_x
        self.template[:, 1] *= scale_y
    
    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> AlignedFace:
        """
        Align face using landmarks.
        
        Args:
            image: Input image
            landmarks: 5 facial landmarks (right eye, left eye, nose, right mouth, left mouth)
            bbox: Optional face bounding box
            
        Returns:
            Aligned face result
        """
        # Convert landmarks to float32
        src_points = landmarks.astype(np.float32)
        dst_points = self.template
        
        # Estimate affine transformation
        transformation_matrix = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.LMEDS
        )[0]
        
        if transformation_matrix is None:
            # Fallback to simple crop
            return self._fallback_align(image, bbox)
        
        # Apply transformation
        aligned_image = cv2.warpAffine(
            image,
            transformation_matrix,
            self.target_size,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128)
        )
        
        # Calculate alignment score (how well landmarks match template)
        aligned_landmarks = cv2.transform(
            src_points.reshape(1, -1, 2),
            transformation_matrix
        ).reshape(-1, 2)
        
        alignment_score = self._calculate_alignment_score(aligned_landmarks, dst_points)
        
        return AlignedFace(
            image=aligned_image,
            transformation_matrix=transformation_matrix,
            original_bbox=bbox or (0, 0, image.shape[1], image.shape[0]),
            alignment_score=alignment_score
        )
    
    def align_from_detection(
        self,
        image: np.ndarray,
        face_detection
    ) -> Optional[AlignedFace]:
        """
        Align face from detection result.
        
        Args:
            image: Input image
            face_detection: FaceDetectionResult object
            
        Returns:
            Aligned face or None if no landmarks
        """
        if face_detection.landmarks is None:
            # No landmarks, use fallback
            return self._fallback_align(image, face_detection.bbox)
        
        return self.align(image, face_detection.landmarks, face_detection.bbox)
    
    def _fallback_align(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]]
    ) -> AlignedFace:
        """
        Fallback alignment using simple crop and resize.
        
        Args:
            image: Input image
            bbox: Face bounding box
            
        Returns:
            Aligned face
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_crop = image[y1:y2, x1:x2]
        else:
            face_crop = image
        
        # Resize to target size
        aligned_image = cv2.resize(face_crop, self.target_size)
        
        # Identity transformation
        transformation_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        return AlignedFace(
            image=aligned_image,
            transformation_matrix=transformation_matrix,
            original_bbox=bbox or (0, 0, image.shape[1], image.shape[0]),
            alignment_score=0.5  # Lower score for fallback
        )
    
    def _calculate_alignment_score(
        self,
        aligned_landmarks: np.ndarray,
        target_landmarks: np.ndarray
    ) -> float:
        """
        Calculate alignment quality score.
        
        Args:
            aligned_landmarks: Landmarks after alignment
            target_landmarks: Target template landmarks
            
        Returns:
            Alignment score (0-1, higher is better)
        """
        # Calculate mean squared error
        mse = np.mean(np.square(aligned_landmarks - target_landmarks))
        
        # Convert to score (inverse of error, normalized)
        max_error = 100.0  # pixels
        score = max(0, 1 - mse / max_error)
        
        return float(score)
    
    def extract_face_chip(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Extract face chip with padding.
        
        Args:
            image: Input image
            bbox: Face bounding box (x1, y1, x2, y2)
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            Face chip
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        face_w = x2 - x1
        face_h = y2 - y1
        
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)
        
        # Apply padding with boundary check
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Extract chip
        chip = image[y1:y2, x1:x2]
        
        # Resize to target size
        chip = cv2.resize(chip, self.target_size)
        
        return chip
    
    def normalize_face(
        self,
        face_image: np.ndarray,
        method: str = "standard"
    ) -> np.ndarray:
        """
        Normalize face image.
        
        Args:
            face_image: Face image
            method: Normalization method ('standard', 'histeq', 'clahe')
            
        Returns:
            Normalized face image
        """
        if method == "standard":
            # Simple normalization to [0, 255]
            normalized = cv2.normalize(face_image, None, 0, 255, cv2.NORM_MINMAX)
            
        elif method == "histeq":
            # Histogram equalization
            if len(face_image.shape) == 3:
                # Convert to YUV, equalize Y channel
                yuv = cv2.cvtColor(face_image, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                normalized = cv2.equalizeHist(face_image)
                
        elif method == "clahe":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(face_image.shape) == 3:
                lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                normalized = clahe.apply(face_image)
        else:
            normalized = face_image
        
        return normalized


if __name__ == "__main__":
    # Test face aligner
    print("Face Aligner Test")
    print("=" * 50)
    
    try:
        aligner = FaceAligner(target_size=(112, 112))
        print("Face aligner initialized!")
        
        # Create synthetic face with landmarks
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Draw face ellipse
        cv2.ellipse(image, (320, 240), (100, 120), 0, 0, 360, (150, 150, 150), -1)
        
        # Define landmarks (right eye, left eye, nose, right mouth, left mouth)
        landmarks = np.array([
            [280, 220],  # Right eye
            [360, 220],  # Left eye
            [320, 270],  # Nose
            [290, 320],  # Right mouth
            [350, 320],  # Left mouth
        ])
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)][i]
            cv2.circle(image, (x, y), 5, color, -1)
        
        # Align face
        aligned = aligner.align(image, landmarks)
        print(f"Aligned face shape: {aligned.image.shape}")
        print(f"Alignment score: {aligned.alignment_score:.3f}")
        
        # Test chip extraction
        chip = aligner.extract_face_chip(image, (220, 120, 420, 360))
        print(f"Face chip shape: {chip.shape}")
        
        # Test normalization
        normalized = aligner.normalize_face(chip, method="clahe")
        print("Normalization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")
