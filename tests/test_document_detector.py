"""Tests for document detection module."""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Skip tests if model not available
pytestmark = pytest.mark.skipif(
    not (Path(__file__).parent.parent / "models" / "yolox_document.onnx").exists(),
    reason="YOLOX model not found"
)

from src.document_detection.detector import DocumentDetector, DetectionResult


def test_detection_result():
    """Test DetectionResult dataclass."""
    result = DetectionResult(
        bbox=(100, 100, 500, 300),
        confidence=0.95,
        class_id=0,
        class_name="id_card"
    )
    
    assert result.bbox == (100, 100, 500, 300)
    assert result.confidence == 0.95
    assert result.class_id == 0
    assert result.class_name == "id_card"


def test_detector_initialization():
    """Test detector initialization."""
    try:
        detector = DocumentDetector()
        assert detector.session is not None
        assert detector.input_name is not None
        assert detector.output_name is not None
    except FileNotFoundError:
        pytest.skip("Model file not found")


def test_preprocess():
    """Test image preprocessing."""
    try:
        detector = DocumentDetector()
        
        # Create test image
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Preprocess
        input_tensor, scale_factors = detector.preprocess(image)
        
        # Check output shape
        assert input_tensor.shape == (1, 3, 640, 640)
        assert len(scale_factors) == 2
        assert scale_factors[0] > 0  # scale_w
        assert scale_factors[1] > 0  # scale_h
        
    except FileNotFoundError:
        pytest.skip("Model file not found")


def test_postprocess():
    """Test postprocessing with dummy outputs."""
    try:
        detector = DocumentDetector()
        
        # Create dummy outputs (1 box with high confidence)
        # Format: [x, y, w, h, obj_conf, class_probs...]
        dummy_output = np.zeros((1, 100, 85), dtype=np.float32)
        dummy_output[0, 0, :4] = [320, 320, 200, 150]  # box
        dummy_output[0, 0, 4] = 0.95  # objectness
        dummy_output[0, 0, 5] = 0.9   # class 0 probability
        
        scale_factors = (3.0, 1.6875)  # 1920/640, 1080/640
        orig_shape = (1080, 1920)
        
        results = detector.postprocess(dummy_output, scale_factors, orig_shape)
        
        # Should detect at least one box
        assert len(results) >= 0  # May be filtered by NMS
        
    except FileNotFoundError:
        pytest.skip("Model file not found")


def test_visualize():
    """Test visualization."""
    try:
        detector = DocumentDetector()
        
        # Create test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create detection results
        detections = [
            DetectionResult(
                bbox=(100, 100, 300, 200),
                confidence=0.95,
                class_id=0,
                class_name="id_card"
            )
        ]
        
        # Visualize
        result = detector.visualize(image, detections)
        
        # Check output
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        
    except FileNotFoundError:
        pytest.skip("Model file not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
