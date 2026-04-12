"""Tests for image utilities."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.utils.image_utils import (
    resize_image,
    get_image_info,
    denoise_image,
    enhance_contrast,
)


def test_resize_image():
    """Test image resizing."""
    # Create test image
    image = np.random.randint(0, 255, (1000, 2000, 3), dtype=np.uint8)
    
    # Test max_size
    resized = resize_image(image, max_size=500)
    h, w = resized.shape[:2]
    assert max(h, w) <= 500
    
    # Test target_size
    resized = resize_image(image, target_size=(640, 480))
    assert resized.shape == (480, 640, 3)


def test_get_image_info():
    """Test image info extraction."""
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    info = get_image_info(image)
    
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["channels"] == 3
    assert info["aspect_ratio"] == 1920 / 1080


def test_denoise_image():
    """Test image denoising."""
    # Create noisy image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Denoise
    denoised = denoise_image(image)
    
    assert denoised.shape == image.shape
    assert denoised.dtype == image.dtype


def test_enhance_contrast():
    """Test contrast enhancement."""
    # Create low contrast image
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    # Enhance
    enhanced = enhance_contrast(image)
    
    assert enhanced.shape == image.shape
    assert enhanced.dtype == image.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
