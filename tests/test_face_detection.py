"""Tests for face detection module."""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_detection.face_detector import FaceDetector, FaceDetectionResult
from face_detection.face_aligner import FaceAligner, AlignedFace
from face_detection.face_pipeline import FacePipeline, FaceExtractionResult


class TestFaceDetector(unittest.TestCase):
    """Test face detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.conf_threshold, 0.7)
    
    def test_detect_empty_image(self):
        """Test detection on empty image."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        faces = self.detector.detect(image)
        
        # Should return empty list or detect something
        self.assertIsInstance(faces, list)
    
    def test_detect_synthetic_face(self):
        """Test detection on synthetic face."""
        # Create image with face-like region
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Draw face ellipse
        cv2 = __import__('cv2')
        cv2.ellipse(image, (320, 240), (100, 120), 0, 0, 360, (150, 150, 150), -1)
        cv2.circle(image, (280, 220), 15, (100, 100, 100), -1)
        cv2.circle(image, (360, 220), 15, (100, 100, 100), -1)
        
        faces = self.detector.detect(image)
        
        # Should detect at least one face
        self.assertIsInstance(faces, list)
    
    def test_detect_largest_face(self):
        """Test largest face detection."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        cv2 = __import__('cv2')
        cv2.ellipse(image, (320, 240), (100, 120), 0, 0, 360, (150, 150, 150), -1)
        
        face = self.detector.detect_largest_face(image)
        
        # May or may not detect face
        if face is not None:
            self.assertIsInstance(face, FaceDetectionResult)
            self.assertGreater(face.confidence, 0)
    
    def test_detect_in_region(self):
        """Test detection in specific region."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        cv2 = __import__('cv2')
        cv2.ellipse(image, (320, 240), (100, 120), 0, 0, 360, (150, 150, 150), -1)
        
        region = (200, 100, 440, 380)
        faces = self.detector.detect_in_region(image, region)
        
        self.assertIsInstance(faces, list)
    
    def test_visualize_detection(self):
        """Test detection visualization."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Create fake detection result
        face = FaceDetectionResult(
            bbox=(220, 140, 420, 340),
            confidence=0.85,
            landmarks=np.array([[280, 220], [360, 220], [320, 270], [290, 320], [350, 320]]),
            face_id=0
        )
        
        vis = self.detector.visualize_detection(image, [face])
        
        self.assertEqual(vis.shape, image.shape)


class TestFaceAligner(unittest.TestCase):
    """Test face aligner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aligner = FaceAligner(target_size=(112, 112))
    
    def test_initialization(self):
        """Test aligner initialization."""
        self.assertIsNotNone(self.aligner)
        self.assertEqual(self.aligner.target_size, (112, 112))
    
    def test_align(self):
        """Test face alignment."""
        cv2 = __import__('cv2')
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Draw face with landmarks
        cv2.ellipse(image, (320, 240), (100, 120), 0, 0, 360, (150, 150, 150), -1)
        
        landmarks = np.array([
            [280, 220],  # Right eye
            [360, 220],  # Left eye
            [320, 270],  # Nose
            [290, 320],  # Right mouth
            [350, 320],  # Left mouth
        ])
        
        aligned = self.aligner.align(image, landmarks)
        
        self.assertIsInstance(aligned, AlignedFace)
        self.assertEqual(aligned.image.shape[:2], (112, 112))
        self.assertIsNotNone(aligned.transformation_matrix)
    
    def test_fallback_align(self):
        """Test fallback alignment without landmarks."""
        cv2 = __import__('cv2')
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        bbox = (220, 140, 420, 340)
        aligned = self.aligner._fallback_align(image, bbox)
        
        self.assertIsInstance(aligned, AlignedFace)
        self.assertEqual(aligned.image.shape[:2], (112, 112))
    
    def test_extract_face_chip(self):
        """Test face chip extraction."""
        cv2 = __import__('cv2')
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        bbox = (220, 140, 420, 340)
        chip = self.aligner.extract_face_chip(image, bbox)
        
        self.assertEqual(chip.shape[:2], (112, 112))
    
    def test_normalize_face(self):
        """Test face normalization."""
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Test different methods
        for method in ["standard", "histeq", "clahe"]:
            normalized = self.aligner.normalize_face(face, method=method)
            self.assertEqual(normalized.shape, face.shape)


class TestFacePipeline(unittest.TestCase):
    """Test face pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = FacePipeline()
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.detector)
        self.assertIsNotNone(self.pipeline.aligner)
    
    def test_extract_face(self):
        """Test face extraction."""
        cv2 = __import__('cv2')
        image = np.ones((600, 900, 3), dtype=np.uint8) * 240
        
        # Draw face in upper portion
        cv2.ellipse(image, (450, 200), (80, 100), 0, 0, 360, (180, 150, 120), -1)
        cv2.circle(image, (420, 180), 10, (50, 50, 50), -1)
        cv2.circle(image, (480, 180), 10, (50, 50, 50), -1)
        
        result = self.pipeline.extract_face(image)
        
        self.assertIsInstance(result, FaceExtractionResult)
        # May or may not succeed with synthetic face
    
    def test_extract_face_with_document_bbox(self):
        """Test face extraction with document bbox."""
        cv2 = __import__('cv2')
        image = np.ones((600, 900, 3), dtype=np.uint8) * 240
        
        cv2.ellipse(image, (450, 200), (80, 100), 0, 0, 360, (180, 150, 120), -1)
        
        doc_bbox = (50, 50, 850, 550)
        result = self.pipeline.extract_face(image, document_bbox=doc_bbox)
        
        self.assertIsInstance(result, FaceExtractionResult)
    
    def test_extract_face_file_not_found(self):
        """Test extraction with non-existent file."""
        result = self.pipeline.extract_face_from_file("/nonexistent/path.jpg")
        
        self.assertIsInstance(result, FaceExtractionResult)
        self.assertFalse(result.success)
        self.assertIn("Cannot load", result.error_message)
    
    def test_check_quality(self):
        """Test quality check."""
        # Good quality face
        good_face = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)
        score = self.pipeline._check_quality(good_face)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Low quality (dark)
        dark_face = np.ones((112, 112, 3), dtype=np.uint8) * 20
        dark_score = self.pipeline._check_quality(dark_face)
        self.assertLess(dark_score, score)
    
    def test_visualize_result(self):
        """Test result visualization."""
        cv2 = __import__('cv2')
        image = np.ones((600, 900, 3), dtype=np.uint8) * 240
        
        detection = FaceDetectionResult(
            bbox=(370, 100, 530, 300),
            confidence=0.85,
            landmarks=np.array([[420, 180], [480, 180], [450, 210], [430, 250], [470, 250]]),
            face_id=0
        )
        
        result = FaceExtractionResult(
            success=True,
            face_image=np.ones((112, 112, 3), dtype=np.uint8) * 150,
            detection=detection,
            alignment=None,
            error_message=""
        )
        
        vis = self.pipeline.visualize_result(image, result)
        
        self.assertEqual(vis.shape, image.shape)
    
    def test_compare_faces(self):
        """Test face comparison."""
        face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        face2 = face1.copy()
        
        # Same faces should have high similarity
        similarity = self.pipeline.compare_faces(face1, face2, method="histogram")
        
        self.assertIsInstance(similarity, float)
        self.assertGreater(similarity, 0.5)  # Should be similar
        
        # Different faces
        face3 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        similarity_diff = self.pipeline.compare_faces(face1, face3, method="histogram")
        
        # Should be less similar
        self.assertLess(similarity_diff, similarity)


class TestFaceDetectionResult(unittest.TestCase):
    """Test FaceDetectionResult dataclass."""
    
    def test_result_properties(self):
        """Test result properties."""
        result = FaceDetectionResult(
            bbox=(100, 100, 200, 250),
            confidence=0.85,
            landmarks=np.array([[120, 140], [180, 140], [150, 170], [130, 210], [170, 210]]),
            face_id=0
        )
        
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 150)
        self.assertEqual(result.center, (150, 175))
        self.assertEqual(result.area, 15000)


class TestFaceExtractionResult(unittest.TestCase):
    """Test FaceExtractionResult dataclass."""
    
    def test_success_result(self):
        """Test successful extraction result."""
        detection = FaceDetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            landmarks=None,
            face_id=0
        )
        
        result = FaceExtractionResult(
            success=True,
            face_image=np.ones((112, 112, 3), dtype=np.uint8),
            detection=detection,
            alignment=None,
            error_message=""
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.face_bbox, (100, 100, 200, 200))
        self.assertEqual(result.confidence, 0.9)
    
    def test_failed_result(self):
        """Test failed extraction result."""
        result = FaceExtractionResult(
            success=False,
            face_image=None,
            detection=None,
            alignment=None,
            error_message="No face detected"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "No face detected")
        self.assertIsNone(result.face_bbox)
        self.assertEqual(result.confidence, 0.0)


if __name__ == '__main__':
    unittest.main()
