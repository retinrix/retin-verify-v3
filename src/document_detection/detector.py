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
        min_aspect_ratio: float = 1.3,
        max_aspect_ratio: float = 1.9,
        min_box_area: float = 0.05,
    ):
        """
        Initialize document detector.
        
        Args:
            model_path: Path to YOLOX ONNX model
            input_size: Model input size (width, height)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS IoU threshold
            min_aspect_ratio: Minimum width/height aspect ratio for a valid ID card
            max_aspect_ratio: Maximum width/height aspect ratio for a valid ID card
            min_box_area: Minimum detection area as fraction of image area
        """
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_box_area = min_box_area
        
        self.class_names = {0: "id-front", 1: "id-back"}
        
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
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """
        Preprocess image for inference using letterbox resize (same as YOLOX training).
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (preprocessed image, scale factors, padding offsets)
        """
        # Store original size
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # Letterbox resize: scale to fit within target size, pad with gray
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (gray = 114, as YOLOX uses)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Center the resized image
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert BGR to RGB
        padded = padded[:, :, ::-1].copy()
        
        # Transpose to NCHW format
        input_tensor = np.transpose(padded, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        return input_tensor, (scale, scale), (pad_x, pad_y)
    
    def postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        orig_shape: Tuple[int, int],
    ) -> List[DetectionResult]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Model raw outputs
            scale: Resize scale factor
            pad_x: Horizontal padding
            pad_y: Vertical padding
            orig_shape: Original image shape (h, w)
            
        Returns:
            List of detection results
        """
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
            
            # Get box coordinates (center_x, center_y, width, height) in padded image space
            cx, cy, w, h = pred[0:4]
            
            # Remove padding offset
            cx -= pad_x
            cy -= pad_y
            
            # Scale back to original image
            cx /= scale
            cy /= scale
            w /= scale
            h /= scale
            
            # Convert to (x1, y1, x2, y2)
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            box_w = x2 - x1
            box_h = y2 - y1
            if box_h <= 0 or box_w <= 0:
                continue
            
            aspect = box_w / box_h
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue
            
            area_ratio = (box_w * box_h) / (orig_w * orig_h)
            if area_ratio < self.min_box_area:
                continue
            
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
                class_name=self.class_names.get(int(class_ids[idx]), "unknown")
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
        input_tensor, (scale, _), (pad_x, pad_y) = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        
        # Postprocess
        results = self.postprocess(outputs, scale, pad_x, pad_y, orig_shape)
        
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
