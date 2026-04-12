"""Tests for security features and tampering detection modules."""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from security_features.hologram_detector import HologramDetector, HologramRegion, HologramType
from security_features.laser_detector import LaserDetector, LaserRegion, LaserFeature
from security_features.security_analyzer import SecurityAnalyzer, SecurityResult
from tampering_detection.ela_analyzer import ELAAnalyzer, ELAResult
from tampering_detection.copy_move_detector import CopyMoveDetector
from tampering_detection.tampering_pipeline import TamperingPipeline, TamperingResult


class TestHologramDetector(unittest.TestCase):
    """Test hologram detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HologramDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.min_region_size, 50)
    
    def test_detect(self):
        """Test hologram detection."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # Create hologram-like region (high saturation)
        hsv_region = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_region[:, :, 0] = np.linspace(0, 180, 100).reshape(1, -1).astype(np.uint8)  # Hue gradient
        hsv_region[:, :, 1] = 255  # Full saturation
        hsv_region[:, :, 2] = 200  # Value
        bgr = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR)
        image[100:200, 100:200] = bgr
        
        regions = self.detector.detect(image)
        
        self.assertIsInstance(regions, list)
    
    def test_verify_authenticity(self):
        """Test authenticity verification."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        is_auth, details = self.detector.verify_authenticity(image)
        
        self.assertIsInstance(is_auth, bool)
        self.assertIsInstance(details, dict)
        self.assertIn('detected_regions', details)
        self.assertIn('overall_confidence', details)
    
    def test_calculate_color_variance(self):
        """Test color variance calculation."""
        cv2 = __import__('cv2')
        region = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        variance = self.detector._calculate_color_variance(region)
        
        self.assertIsInstance(variance, float)
        self.assertGreaterEqual(variance, 0)


class TestLaserDetector(unittest.TestCase):
    """Test laser detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LaserDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.edge_threshold, 50.0)
    
    def test_detect(self):
        """Test laser detection."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 240
        
        # Create laser-like region
        laser_region = np.random.randint(100, 200, (100, 100), dtype=np.uint8)
        laser_color = cv2.cvtColor(laser_region, cv2.COLOR_GRAY2BGR)
        image[100:200, 100:200] = laser_color
        
        regions = self.detector.detect(image)
        
        self.assertIsInstance(regions, list)
    
    def test_verify_photo_authenticity(self):
        """Test photo authenticity verification."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 240
        
        is_auth, details = self.detector.verify_photo_authenticity(image)
        
        self.assertIsInstance(is_auth, bool)
        self.assertIsInstance(details, dict)
    
    def test_classify_feature(self):
        """Test feature classification."""
        feature_type = self.detector._classify_feature(0.6, 0.4, (100, 100))
        
        self.assertIsInstance(feature_type, LaserFeature)


class TestSecurityAnalyzer(unittest.TestCase):
    """Test security analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SecurityAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.hologram_detector)
        self.assertIsNotNone(self.analyzer.laser_detector)
    
    def test_analyze(self):
        """Test complete analysis."""
        cv2 = __import__('cv2')
        image = np.ones((600, 900, 3), dtype=np.uint8) * 240
        
        # Add face
        cv2.ellipse(image, (450, 200), (80, 100), 0, 0, 360, (180, 160, 140), -1)
        
        result = self.analyzer.analyze(image)
        
        self.assertIsInstance(result, SecurityResult)
        self.assertIsInstance(result.is_authentic, bool)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)
    
    def test_analyze_print_quality(self):
        """Test print quality analysis."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        quality = self.analyzer._analyze_print_quality(image)
        
        self.assertIsInstance(quality, dict)
        self.assertIn('resolution', quality)
        self.assertIn('sharpness', quality)
    
    def test_error_level_analysis(self):
        """Test error level analysis."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        ela = self.analyzer._error_level_analysis(image)
        
        self.assertIsInstance(ela, dict)
        self.assertIn('mean_error', ela)
        self.assertIn('max_error', ela)
    
    def test_detect_copy_move(self):
        """Test copy-move detection."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        result = self.analyzer.detect_copy_move(image)
        
        self.assertIsInstance(result, dict)
        self.assertIn('forgery_detected', result)


class TestELAAnalyzer(unittest.TestCase):
    """Test ELA analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ELAAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.quality, 95)
    
    def test_analyze(self):
        """Test ELA analysis."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        cv2.rectangle(image, (100, 100), (200, 200), (150, 150, 150), -1)
        
        result = self.analyzer.analyze(image)
        
        self.assertIsInstance(result, ELAResult)
        self.assertIsInstance(result.ela_image, np.ndarray)
        self.assertIsInstance(result.is_tampered, bool)
        self.assertGreaterEqual(result.confidence, 0)
    
    def test_detect_suspicious_regions(self):
        """Test suspicious region detection."""
        diff_gray = np.zeros((100, 100), dtype=np.uint8)
        diff_gray[40:60, 40:60] = 100  # High error region
        
        regions = self.analyzer._detect_suspicious_regions(diff_gray)
        
        self.assertIsInstance(regions, list)


class TestCopyMoveDetector(unittest.TestCase):
    """Test copy-move detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = CopyMoveDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.block_size, 8)
    
    def test_detect(self):
        """Test copy-move detection."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # Add copy-move forgery
        pattern = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        image[50:100, 50:100] = pattern
        image[200:250, 300:350] = pattern
        
        result = self.detector.detect(image)
        
        self.assertIsInstance(result, dict)
        self.assertIn('forgery_detected', result)
        self.assertIn('confidence', result)
    
    def test_extract_blocks(self):
        """Test block extraction."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        blocks, positions = self.detector._extract_blocks(gray)
        
        self.assertIsInstance(blocks, np.ndarray)
        self.assertIsInstance(positions, list)
        self.assertEqual(len(blocks), len(positions))
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        from tampering_detection.copy_move_detector import CopyMoveMatch
        
        matches = [
            CopyMoveMatch((0, 0, 10, 10), (50, 50, 60, 60), 0.95, 0),
            CopyMoveMatch((20, 20, 30, 30), (70, 70, 80, 80), 0.92, 1),
        ]
        clusters = [{'size': 2}]
        
        confidence = self.detector._calculate_confidence(matches, clusters)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)


class TestTamperingPipeline(unittest.TestCase):
    """Test tampering pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = TamperingPipeline()
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.ela_analyzer)
        self.assertIsNotNone(self.pipeline.copy_move_detector)
    
    def test_analyze(self):
        """Test complete analysis."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # Add content
        cv2.rectangle(image, (50, 50), (150, 150), (150, 150, 150), -1)
        cv2.circle(image, (300, 200), 50, (100, 100, 100), -1)
        
        result = self.pipeline.analyze(image)
        
        self.assertIsInstance(result, TamperingResult)
        self.assertIsInstance(result.is_tampered, bool)
        self.assertIsInstance(result.inconsistencies, list)
        self.assertIsInstance(result.recommendations, list)
    
    def test_analyze_file_not_found(self):
        """Test analysis with non-existent file."""
        result = self.pipeline.analyze_file("/nonexistent/path.jpg")
        
        self.assertIsInstance(result, TamperingResult)
        self.assertFalse(result.is_tampered)
        self.assertIn("Failed to load image", result.inconsistencies)
    
    def test_perform_consistency_checks(self):
        """Test consistency checks."""
        cv2 = __import__('cv2')
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        checks = self.pipeline._perform_consistency_checks(image)
        
        self.assertIsInstance(checks, list)


class TestHologramRegion(unittest.TestCase):
    """Test HologramRegion dataclass."""
    
    def test_creation(self):
        """Test region creation."""
        region = HologramRegion(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            pattern_type=HologramType.DOT_MATRIX,
            color_variance=50.0,
            iridescence_score=0.7
        )
        
        self.assertEqual(region.bbox, (100, 100, 200, 200))
        self.assertEqual(region.confidence, 0.85)
        self.assertEqual(region.pattern_type, HologramType.DOT_MATRIX)


class TestLaserRegion(unittest.TestCase):
    """Test LaserRegion dataclass."""
    
    def test_creation(self):
        """Test region creation."""
        region = LaserRegion(
            bbox=(100, 100, 200, 200),
            feature_type=LaserFeature.PHOTO,
            confidence=0.8,
            texture_score=0.7,
            edge_score=0.6,
            depth_score=0.5
        )
        
        self.assertEqual(region.bbox, (100, 100, 200, 200))
        self.assertEqual(region.feature_type, LaserFeature.PHOTO)


class TestSecurityResult(unittest.TestCase):
    """Test SecurityResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SecurityResult(
            is_authentic=True,
            overall_score=0.85,
            hologram_result={'authentic': True},
            laser_result={'authentic': True},
            print_quality={'resolution_check': True},
            error_checks={'suspicious': False},
            warnings=[]
        )
        
        d = result.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d['is_authentic'], True)
        self.assertEqual(d['overall_score'], 0.85)


class TestTamperingResult(unittest.TestCase):
    """Test TamperingResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TamperingResult(
            is_tampered=False,
            overall_confidence=0.2,
            ela_result=None,
            copy_move_result=None,
            metadata_analysis={},
            inconsistencies=[],
            recommendations=[]
        )
        
        d = result.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d['is_tampered'], False)


if __name__ == '__main__':
    unittest.main()
