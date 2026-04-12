"""MRZ text recognition using PaddleOCR or EasyOCR fallback."""

import cv2
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PaddleOCR, fall back to EasyOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logger.info("PaddleOCR available")
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available, will use EasyOCR fallback")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("EasyOCR available")
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")


class MRZRecognizer:
    """
    MRZ text recognizer using PaddleOCR or EasyOCR fallback.
    
    Supports:
    - Latin characters for MRZ (ICAO 9303)
    - Arabic characters for ID front text
    
    License: Apache-2.0
    """
    
    # Valid MRZ characters
    MRZ_CHARS = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<')
    
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = 'en',
        use_angle_cls: bool = True,
    ):
        """
        Initialize MRZ recognizer.
        
        Args:
            use_gpu: Use GPU for inference
            lang: Language code ('en' for Latin, 'ar' for Arabic)
            use_angle_cls: Use angle classification
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        
        self.ocr_engine = None
        self.easyocr_reader = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load OCR model."""
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
                logger.info("PaddleOCR engine loaded")
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR: {e}")
                self.ocr_engine = None
        
        # Fallback to EasyOCR
        if self.ocr_engine is None and EASYOCR_AVAILABLE:
            try:
                lang_list = ['en']
                if self.lang == 'ar':
                    lang_list = ['ar', 'en']
                
                self.easyocr_reader = easyocr.Reader(
                    lang_list,
                    gpu=self.use_gpu
                )
                logger.info("EasyOCR engine loaded")
            except Exception as e:
                logger.warning(f"Failed to load EasyOCR: {e}")
                self.easyocr_reader = None
    
    def preprocess_mrz(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess MRZ image for better recognition.
        
        Args:
            image: Input MRZ image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small (MRZ needs good resolution)
        h, w = gray.shape
        if h < 100:
            scale = 100 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply contrast enhancement
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        return gray
    
    def recognize_text(
        self,
        image: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Recognize text in image.
        
        Args:
            image: Input image
            
        Returns:
            List of (text, confidence) tuples
        """
        results = []
        
        if self.ocr_engine is not None:
            # Use PaddleOCR
            try:
                result = self.ocr_engine.ocr(image, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        if line:
                            text = line[1][0]
                            confidence = line[1][1]
                            results.append((text, confidence))
            except Exception as e:
                logger.warning(f"PaddleOCR recognition failed: {e}")
        
        elif self.easyocr_reader is not None:
            # Use EasyOCR
            try:
                result = self.easyocr_reader.readtext(image)
                for detection in result:
                    bbox, text, confidence = detection
                    results.append((text, confidence))
            except Exception as e:
                logger.warning(f"EasyOCR recognition failed: {e}")
        
        return results
    
    def recognize_mrz(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[List[str]], Optional[List[float]]]:
        """
        Recognize MRZ text from image.
        
        Args:
            image: MRZ region image
            
        Returns:
            Tuple of (lines, confidences) or (None, None)
        """
        # Preprocess
        preprocessed = self.preprocess_mrz(image)
        
        # Recognize text
        results = self.recognize_text(preprocessed)
        
        if not results:
            return None, None
        
        # Extract lines and confidences
        lines = []
        confidences = []
        
        for text, conf in results:
            # Clean text (keep only MRZ characters)
            cleaned = self._clean_mrz_text(text.upper())
            if cleaned:
                lines.append(cleaned)
                confidences.append(conf)
        
        # Sort by vertical position if we have bbox info
        # For now, assume top-to-bottom order
        
        # We expect 3 lines for TD-1 format
        if len(lines) < 3:
            logger.warning(f"Expected 3 MRZ lines, found {len(lines)}")
            # Try to split long lines
            lines = self._split_lines(lines)
        
        # Pad or truncate to 3 lines
        while len(lines) < 3:
            lines.append("" * 30)
            confidences.append(0.0)
        
        lines = lines[:3]
        confidences = confidences[:3]
        
        return lines, confidences
    
    def _clean_mrz_text(self, text: str) -> str:
        """
        Clean text to keep only valid MRZ characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Replace common OCR errors
        replacements = {
            '0': 'O',  # Zero to letter O (MRZ uses O)
            '1': 'I',  # One to letter I
            '5': 'S',  # Five to letter S
            '8': 'B',  # Eight to letter B (sometimes)
        }
        
        cleaned = ""
        for char in text.upper():
            if char in self.MRZ_CHARS:
                cleaned += char
            elif char in replacements:
                cleaned += replacements[char]
            # Skip other characters
        
        return cleaned
    
    def _split_lines(self, lines: List[str]) -> List[str]:
        """
        Try to split long lines into 3 MRZ lines.
        
        Args:
            lines: Input lines
            
        Returns:
            Split lines
        """
        # If we have one long line, try to split it
        if len(lines) == 1 and len(lines[0]) > 60:
            text = lines[0]
            # Try to find natural split points
            if len(text) >= 90:
                return [text[:30], text[30:60], text[60:90]]
            elif len(text) >= 60:
                return [text[:30], text[30:60], text[60:]]
        
        return lines
    
    def validate_mrz_format(self, lines: List[str]) -> Tuple[bool, str]:
        """
        Validate MRZ format.
        
        Args:
            lines: MRZ lines
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(lines) != 3:
            return False, f"Expected 3 lines, got {len(lines)}"
        
        # Check line lengths (TD-1 format: 30 characters per line)
        for i, line in enumerate(lines):
            if len(line) != 30:
                return False, f"Line {i+1} has {len(line)} chars, expected 30"
        
        # Check first character (document type)
        if lines[0][0] not in ['I', 'P', 'C', 'V']:
            return False, f"Invalid document type: {lines[0][0]}"
        
        # Check for valid characters
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char not in self.MRZ_CHARS:
                    return False, f"Line {i+1} position {j+1} has invalid character: {char}"
        
        return True, ""


if __name__ == "__main__":
    # Test recognizer
    print("MRZ Recognizer Test")
    print("=" * 50)
    
    try:
        recognizer = MRZRecognizer(use_gpu=False)
        print("Recognizer initialized successfully!")
        
        # Test with synthetic image
        image = np.ones((150, 800, 3), dtype=np.uint8) * 255
        
        # Add text-like patterns
        cv2.putText(image, "IDDZA1154521568<<<<<<<<<<<<<<<", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "6703174M2908236DZA<<<<<<<<<<<8", (50, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "BOUTIGHANE<<MOHAMED<NAAMAN<<<<", (50, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Recognize
        lines, confidences = recognizer.recognize_mrz(image)
        
        if lines:
            print("\nRecognized MRZ lines:")
            for i, (line, conf) in enumerate(zip(lines, confidences)):
                print(f"  Line {i+1}: {line} (conf: {conf:.2f})")
        else:
            print("No text recognized")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest completed!")
