"""Validation utilities for image quality and processing."""

import cv2
import numpy as np
from typing import Dict, Tuple


def validate_image_quality(image: np.ndarray) -> Tuple[bool, Dict]:
    """
    Validate image quality for ID card processing.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (is_valid, quality_metrics)
    """
    metrics = {
        "resolution_ok": False,
        "not_blurry": False,
        "good_contrast": False,
        "good_lighting": False,
        "overall_score": 0.0,
    }
    
    h, w = image.shape[:2]
    
    # Check resolution
    min_resolution = 640
    metrics["resolution_ok"] = (w >= min_resolution and h >= min_resolution)
    metrics["resolution"] = {"width": w, "height": h}
    
    # Check blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur_score = detect_blur(gray)
    metrics["not_blurry"] = blur_score > 100
    metrics["blur_score"] = blur_score
    
    # Check contrast
    contrast_score = compute_contrast(gray)
    metrics["good_contrast"] = contrast_score > 30
    metrics["contrast_score"] = contrast_score
    
    # Check lighting
    lighting_score = compute_lighting_uniformity(gray)
    metrics["good_lighting"] = lighting_score > 0.5
    metrics["lighting_score"] = lighting_score
    
    # Overall score (0-100)
    score = 0
    if metrics["resolution_ok"]:
        score += 25
    if metrics["not_blurry"]:
        score += 25
    if metrics["good_contrast"]:
        score += 25
    if metrics["good_lighting"]:
        score += 25
    
    metrics["overall_score"] = score
    
    is_valid = score >= 75
    
    return is_valid, metrics


def detect_blur(gray_image: np.ndarray) -> float:
    """
    Detect image blur using Laplacian variance.
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        Blur score (higher = less blurry)
    """
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var


def compute_contrast(gray_image: np.ndarray) -> float:
    """
    Compute image contrast.
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        Contrast score (standard deviation of pixel values)
    """
    return float(np.std(gray_image))


def compute_lighting_uniformity(gray_image: np.ndarray) -> float:
    """
    Compute lighting uniformity.
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        Uniformity score (1.0 = perfect uniformity, 0.0 = very uneven)
    """
    # Divide image into regions
    h, w = gray_image.shape
    regions = []
    
    # Split into 3x3 grid
    for i in range(3):
        for j in range(3):
            y1 = int(h * i / 3)
            y2 = int(h * (i + 1) / 3)
            x1 = int(w * j / 3)
            x2 = int(w * (j + 1) / 3)
            
            region = gray_image[y1:y2, x1:x2]
            regions.append(np.mean(region))
    
    # Compute coefficient of variation
    mean_brightness = np.mean(regions)
    std_brightness = np.std(regions)
    
    if mean_brightness == 0:
        return 0.0
    
    cv = std_brightness / mean_brightness
    uniformity = max(0.0, 1.0 - cv)
    
    return uniformity


def check_glare(image: np.ndarray) -> Tuple[bool, float]:
    """
    Check for glare in image.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (has_glare, glare_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Threshold for bright areas
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Calculate glare area
    glare_area = np.sum(bright_mask > 0) / bright_mask.size
    
    has_glare = glare_area > 0.05  # More than 5% bright area
    
    return has_glare, float(glare_area)


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate image noise level.
    
    Args:
        image: Input image
        
    Returns:
        Noise estimate (lower = less noisy)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Use median absolute deviation
    median = np.median(gray)
    mad = np.median(np.abs(gray - median))
    
    return float(mad)
