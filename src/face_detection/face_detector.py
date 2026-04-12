"""Face detection using OpenCV YuNet (DNN-based)."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionResult:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    face_id: Optional[int] = None
    
    @property
    def width(self) -> int:
        """Get face width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get face height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get face center."""
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )
    
    @property
    def area(self) -> int:
        """Get face area."""
        return self.width * self.height


class FaceDetector:
    """
    Face detector using OpenCV YuNet DNN model.
    
    YuNet is a lightweight face detection model that:
    - Runs efficiently on CPU
    - Detects faces with bounding boxes
    - Provides 5 facial landmarks (eyes, nose, mouth corners)
    - Apache-2.0 licensed
    
    Model download: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
    """
    
    # Default model URL
    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        input_size: Tuple[int, int] = (320, 320),
    ):
        """
        Initialize face detector.
        
        Args:
            model_path: Path to YuNet ONNX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            top_k: Maximum number of detections
            input_size: Input size for model (width, height)
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.input_size = input_size
        
        self.model = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None) -> None:
        """Load YuNet model."""
        if model_path is None:
            # Try to find model in default locations
            possible_paths = [
                "models/face_detection_yunet.onnx",
                "models/yunet/face_detection_yunet.onnx",
                "/usr/share/opencv/models/face_detection_yunet.onnx",
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
        
        if model_path is None or not Path(model_path).exists():
            logger.warning(
                f"YuNet model not found at {model_path}. "
                f"Please download from: {self.MODEL_URL}"
            )
            logger.info("Using Haar Cascade fallback")
            self._load_haar_fallback()
            return
        
        try:
            # Load YuNet model using OpenCV DNN
            self.model = cv2.FaceDetectorYN_create(
                model=model_path,
                config="",
                input_size=self.input_size,
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k
            )
            logger.info(f"YuNet model loaded from {model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load YuNet model: {e}")
            self._load_haar_fallback()
    
    def _load_haar_fallback(self) -> None:
        """Load Haar Cascade as fallback."""
        try:
            self.model = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_haar = True
            logger.info("Haar Cascade fallback loaded")
        except Exception as e:
            logger.error(f"Failed to load Haar Cascade: {e}")
            self.model = None
            self.use_haar = False
    
    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True
    ) -> List[FaceDetectionResult]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of face detection results
        """
        if self.model is None:
            logger.error("No face detection model loaded")
            return []
        
        h, w = image.shape[:2]
        
        # Check if using YuNet or Haar
        if hasattr(self, 'use_haar') and self.use_haar:
            return self._detect_haar(image)
        
        # YuNet detection
        return self._detect_yunet(image, w, h, return_landmarks)
    
    def _detect_yunet(
        self,
        image: np.ndarray,
        width: int,
        height: int,
        return_landmarks: bool
    ) -> List[FaceDetectionResult]:
        """Detect faces using YuNet."""
        # Set input size
        self.model.setInputSize((width, height))
        
        # Detect faces
        _, faces = self.model.detect(image)
        
        results = []
        if faces is not None:
            for i, face in enumerate(faces):
                # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence]
                x, y, w, h = face[:4].astype(int)
                confidence = float(face[14])
                
                # Convert to (x1, y1, x2, y2)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                
                # Extract landmarks if requested
                landmarks = None
                if return_landmarks:
                    # 5 landmarks: right eye, left eye, nose, right mouth corner, left mouth corner
                    landmarks = face[4:14].reshape(5, 2).astype(int)
                
                results.append(FaceDetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    landmarks=landmarks,
                    face_id=i
                ))
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def _detect_haar(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using Haar Cascade fallback."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detections = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        if detections is not None:
            for i, (x, y, w, h) in enumerate(detections):
                confidence = 0.7  # Haar doesn't provide confidence
                
                results.append(FaceDetectionResult(
                    bbox=(x, y, x + w, y + h),
                    confidence=confidence,
                    landmarks=None,
                    face_id=i
                ))
        
        return results
    
    def detect_largest_face(
        self,
        image: np.ndarray,
        return_landmarks: bool = True
    ) -> Optional[FaceDetectionResult]:
        """
        Detect the largest face in image.
        
        Args:
            image: Input image
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            Largest face detection result or None
        """
        faces = self.detect(image, return_landmarks)
        
        if not faces:
            return None
        
        # Return largest face by area
        return max(faces, key=lambda f: f.area)
    
    def detect_in_region(
        self,
        image: np.ndarray,
        region_bbox: Tuple[int, int, int, int],
        return_landmarks: bool = True
    ) -> List[FaceDetectionResult]:
        """
        Detect faces in specific region.
        
        Args:
            image: Input image
            region_bbox: Region to search (x1, y1, x2, y2)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of face detection results (coordinates relative to original image)
        """
        x1, y1, x2, y2 = region_bbox
        region = image[y1:y2, x1:x2]
        
        faces = self.detect(region, return_landmarks)
        
        # Adjust coordinates to original image
        adjusted_faces = []
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox
            adjusted_bbox = (fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1)
            
            adjusted_landmarks = None
            if face.landmarks is not None:
                adjusted_landmarks = face.landmarks + np.array([x1, y1])
            
            adjusted_faces.append(FaceDetectionResult(
                bbox=adjusted_bbox,
                confidence=face.confidence,
                landmarks=adjusted_landmarks,
                face_id=face.face_id
            ))
        
        return adjusted_faces
    
    def visualize_detection(
        self,
        image: np.ndarray,
        faces: List[FaceDetectionResult],
        save_path: Optional[str] = None,
        show_confidence: bool = True,
        show_landmarks: bool = True
    ) -> np.ndarray:
        """
        Visualize face detection results.
        
        Args:
            image: Original image
            faces: Face detection results
            save_path: Optional path to save visualization
            show_confidence: Whether to show confidence scores
            show_landmarks: Whether to show facial landmarks
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
        ]
        
        for i, face in enumerate(faces):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = face.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            if show_confidence:
                label = f"Face {i+1}: {face.confidence:.2f}"
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Draw landmarks
            if show_landmarks and face.landmarks is not None:
                landmark_colors = [
                    (0, 0, 255),    # Right eye - red
                    (0, 255, 0),    # Left eye - green
                    (255, 0, 0),    # Nose - blue
                    (0, 255, 255),  # Right mouth - yellow
                    (255, 0, 255),  # Left mouth - magenta
                ]
                
                for j, (lx, ly) in enumerate(face.landmarks):
                    cv2.circle(vis_image, (int(lx), int(ly)), 3, landmark_colors[j], -1)
        
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Test face detector
    print("Face Detector Test")
    print("=" * 50)
    
    try:
        detector = FaceDetector()
        print("Face detector initialized successfully!")
        
        # Test with synthetic image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Draw a face-like region
        cv2.ellipse(image, (320, 240), (100, 120), 0, 0, 360, (150, 150, 150), -1)
        cv2.circle(image, (280, 220), 15, (100, 100, 100), -1)  # Left eye
        cv2.circle(image, (360, 220), 15, (100, 100, 100), -1)  # Right eye
        cv2.ellipse(image, (320, 280), (40, 20), 0, 0, 180, (100, 100, 100), 2)  # Mouth
        
        # Detect faces
        faces = detector.detect(image)
        print(f"Detected {len(faces)} face(s)")
        
        for i, face in enumerate(faces):
            print(f"  Face {i+1}: bbox={face.bbox}, conf={face.confidence:.2f}")
        
        # Visualize
        vis = detector.visualize_detection(image, faces)
        print("Visualization created!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest completed!")
