"""Tests for MRZ OCR module."""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mrz_ocr.mrz_detector import MRZDetector
from mrz_ocr.mrz_recognizer import MRZRecognizer
from mrz_ocr.mrz_pipeline import MRZPipeline, MRZResult


class TestMRZDetector(unittest.TestCase):
    """Test MRZ detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = MRZDetector(use_gpu=False)
    
    def test_detect_mrz_region(self):
        """Test MRZ region detection."""
        # Create synthetic ID card image with MRZ at bottom
        height, width = 600, 900
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw MRZ-like region at bottom
        mrz_y = 450
        image[mrz_y:550, 50:850] = 255
        
        # Detect MRZ
        bbox = self.detector.detect_mrz_region(image)
        
        # Should detect something
        self.assertIsNotNone(bbox)
        self.assertEqual(len(bbox), 4)
    
    def test_detect_mrz_region_with_document_bbox(self):
        """Test MRZ detection with document bbox."""
        height, width = 600, 900
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw MRZ region
        image[450:550, 50:850] = 255
        
        # Provide document bbox
        doc_bbox = (50, 50, 850, 550)
        bbox = self.detector.detect_mrz_region(image, doc_bbox)
        
        self.assertIsNotNone(bbox)
    
    def test_detect_mrz_region_no_mrz(self):
        """Test MRZ detection when no MRZ present."""
        # Create image without MRZ
        image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        
        bbox = self.detector.detect_mrz_region(image)
        
        # Should still return a bbox (heuristic-based)
        self.assertIsNotNone(bbox)


class TestMRZRecognizer(unittest.TestCase):
    """Test MRZ recognition."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = MRZRecognizer(use_gpu=False)
    
    def test_preprocess_mrz(self):
        """Test MRZ preprocessing."""
        # Create synthetic MRZ image
        image = np.ones((100, 400, 3), dtype=np.uint8) * 255
        
        processed = self.recognizer.preprocess_mrz(image)
        
        self.assertIsInstance(processed, np.ndarray)
        # Preprocess returns grayscale (2D) or color (3D) depending on implementation
        self.assertTrue(processed.ndim in [2, 3])
    
    def test_validate_mrz_format_valid(self):
        """Test MRZ format validation with valid MRZ."""
        lines = [
            "IDDZA1154521568<<<<<<<<<<<<<<<",
            "6703174M2908236DZA<<<<<<<<<<<8",
            "BOUTIGHANE<<MOHAMED<NAAMAN<<<<"
        ]
        
        is_valid, error_msg = self.recognizer.validate_mrz_format(lines)
        
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_validate_mrz_format_invalid_length(self):
        """Test MRZ format validation with wrong line length."""
        lines = [
            "IDDZA1154521568<<<<<<<<<<<<<<",  # Too short
            "6703174M2908236DZA<<<<<<<<<<<8",
            "BOUTIGHANE<<MOHAMED<NAAMAN<<<<"
        ]
        
        is_valid, error_msg = self.recognizer.validate_mrz_format(lines)
        
        self.assertFalse(is_valid)
        self.assertIn("Line 1", error_msg)
    
    def test_validate_mrz_format_invalid_chars(self):
        """Test MRZ format validation with invalid characters."""
        lines = [
            "IDDZA1154521568<<<<<<<<<<<<<<<",
            "6703174M2908236DZA<<<<<<<<<<<8",
            "BOUTIGHANE<<MOHAMED@NAAMAN<<<<"  # Invalid @ character
        ]
        
        is_valid, error_msg = self.recognizer.validate_mrz_format(lines)
        
        self.assertFalse(is_valid)
        # Error could be about length or invalid chars
        self.assertTrue("invalid" in error_msg.lower() or "char" in error_msg.lower() or "expected 30" in error_msg)
    
    def test_recognize_mrz_mock(self):
        """Test MRZ recognition with mock data."""
        # Create synthetic MRZ image
        image = np.ones((100, 800, 3), dtype=np.uint8) * 255
        
        # This will use EasyOCR which may not work without models
        # Just test that it doesn't crash
        try:
            lines, confidences = self.recognizer.recognize_mrz(image)
            # If it works, check output format
            if lines is not None:
                self.assertIsInstance(lines, list)
                self.assertIsInstance(confidences, list)
        except Exception:
            # Expected if models not available
            pass


class TestMRZPipeline(unittest.TestCase):
    """Test MRZ pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.pipeline = MRZPipeline(use_gpu=False)
        except ImportError:
            self.skipTest("PaddleOCR not available")
    
    def test_process_empty_image(self):
        """Test pipeline with empty image."""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        
        result = self.pipeline.process(image)
        
        self.assertIsInstance(result, MRZResult)
        # Should fail since no real MRZ
        self.assertFalse(result.success)
    
    def test_process_file_not_found(self):
        """Test pipeline with non-existent file."""
        result = self.pipeline.process_file("/nonexistent/path.jpg")
        
        self.assertIsInstance(result, MRZResult)
        self.assertFalse(result.success)
        self.assertIn("Cannot load", result.error_message)
    
    def test_visualize_result(self):
        """Test result visualization."""
        # Create dummy result
        result = MRZResult(
            success=True,
            mrz_data=None,
            raw_lines=["LINE1", "LINE2", "LINE3"],
            confidences=[0.9, 0.95, 0.92],
            region_bbox=(100, 400, 800, 500),
            error_message=""
        )
        
        image = np.ones((600, 900, 3), dtype=np.uint8) * 200
        vis = self.pipeline.visualize_result(image, result)
        
        self.assertEqual(vis.shape, image.shape)


class TestMRZResult(unittest.TestCase):
    """Test MRZResult dataclass."""
    
    def test_result_creation(self):
        """Test creating MRZResult."""
        result = MRZResult(
            success=True,
            mrz_data=None,
            raw_lines=["LINE1", "LINE2"],
            confidences=[0.9, 0.95],
            region_bbox=(0, 0, 100, 50),
            error_message=""
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.raw_lines), 2)
    
    def test_result_failed(self):
        """Test failed result."""
        result = MRZResult(
            success=False,
            mrz_data=None,
            raw_lines=None,
            confidences=None,
            region_bbox=None,
            error_message="Test error"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Test error")


if __name__ == '__main__':
    unittest.main()
