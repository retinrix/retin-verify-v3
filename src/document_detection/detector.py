"""Document detection using YOLOX (Apache-2.0 license)."""

import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.image_utils import resize_image


@dataclass
class DetectionResult:
    """Document detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class DocumentDetector:
    """
    Document detector using YOLOX ONNX model.
    
    License: Apache-2.0 (free for commercial use)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ):
        """
        Initialize document detector.
        
        Args:
            model_path: Path to YOLOX ONNX model
            input_size: Model input size (width, height)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS IoU threshold
        """
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load model
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "yolox_document.onnx"
        
        self.model_path = str(model_path)
        self.session = None
        self.input_name = None
        self.output_name = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model."""
        # Check if model exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Please download or train a YOLOX model first."
            )
        
        # Create ONNX Runtime session
        providers = ort.get_available_providers()
        preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_providers = [p for p in preferred_providers if p in providers]
        
        self.session = ort.InferenceSession(
            self.model_path,
            providers=session_providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model loaded: {self.model_path}")
        print(f"Providers: {session_providers}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (preprocessed image, scale factors)
        """
        # Store original size
        orig_h, orig_w = image.shape[:2]
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize (YOLOX uses ImageNet normalization)
        resized = resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        resized = resized[:, :, ::-1]
        
        # Transpose to NCHW format
        input_tensor = np.transpose(resized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Calculate scale factors
        scale_w = orig_w / self.input_size[0]
        scale_h = orig_h / self.input_size[1]
        
        return input_tensor, (scale_w, scale_h)
    
    def postprocess(
        self,
        outputs: np.ndarray,
        scale_factors: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[DetectionResult]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Model raw outputs
            scale_factors: (scale_w, scale_h)
            orig_shape: Original image shape (h, w)
            
        Returns:
            List of detection results
        """
        scale_w, scale_h = scale_factors
        orig_h, orig_w = orig_shape
        
        # Parse outputs (YOLOX format: [num_boxes, 85] where 85 = x, y, w, h, obj_conf, 80 class_probs)
        predictions = outputs[0]  # Remove batch dimension
        
        # Filter by confidence
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            # Get objectness score
            obj_conf = pred[4]
            if obj_conf < self.confidence_threshold:
                continue
            
            # Get class with highest probability
            class_probs = pred[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            
            # Final confidence
            confidence = obj_conf * class_score
            if confidence < self.confidence_threshold:
                continue
            
            # Get box coordinates (center_x, center_y, width, height)
            cx, cy, w, h = pred[0:4]
            
            # Convert to (x1, y1, x2, y2)
            x1 = int((cx - w/2) * scale_w)
            y1 = int((cy - h/2) * scale_h)
            x2 = int((cx + w/2) * scale_w)
            y2 = int((cy + h/2) * scale_h)
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)
        
        if not boxes:
            return []
        
        # Apply NMS
        boxes_np = np.array(boxes)
        scores_np = np.array(scores)
        
        indices = cv2.dnn.NMSBoxes(
            boxes_np.tolist(),
            scores_np.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Build results
        results = []
        for idx in indices.flatten() if isinstance(indices, np.ndarray) else indices:
            results.append(DetectionResult(
                bbox=tuple(boxes[idx]),
                confidence=float(scores[idx]),
                class_id=int(class_ids[idx]),
                class_name="id_card"  # Single class for now
            ))
        
        return results
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect documents in image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detection results
        """
        orig_shape = image.shape[:2]
        
        # Preprocess
        input_tensor, scale_factors = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        
        # Postprocess
        results = self.postprocess(outputs, scale_factors, orig_shape)
        
        return results
    
    def detect_best(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Detect the best (highest confidence) document.
        
        Args:
            image: Input image
            
        Returns:
            Best detection result or None
        """
        results = self.detect(image)
        if not results:
            return None
        return max(results, key=lambda r: r.confidence)
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Visualize detection results.
        
        Args:
            image: Input image
            detections: Detection results
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with visualizations
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(
                result,
                (x1, y1_label - label_size[1] - 10),
                (x1 + label_size[0], y1_label),
                color,
                -1
            )
            cv2.putText(
                result,
                label,
                (x1, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return result


def download_pretrained_model(output_dir: str = "models") -> str:
    """
    Download pretrained YOLOX model.
    
    Note: In production, use a proper model. This is a placeholder.
    
    Args:
        output_dir: Output directory for model
        
    Returns:
        Path to downloaded model
    """
    # For now, this is a placeholder
    # In production, download from model repository
    model_path = Path(output_dir) / "yolox_document.onnx"
    
    if not model_path.exists():
        print(f"Pretrained model not found at: {model_path}")
        print("Please train a model or download a pretrained one.")
        print("\nTo train on Colab:")
        print("  1. Open colab/yolox_document_detection.ipynb")
        print("  2. Follow training instructions")
        print("  3. Export model to ONNX format")
        print("  4. Place model in models/ directory")
    
    return str(model_path)


if __name__ == "__main__":
    # Test detector
    print("Document Detector Test")
    print("=" * 50)
    
    try:
        detector = DocumentDetector()
        print("Detector initialized successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train or download a model first.")
