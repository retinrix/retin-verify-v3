"""Image processing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format)
        
    Raises:
        FileNotFoundError: If image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image array to save
        output_path: Output file path
    """
    cv2.imwrite(str(output_path), image)


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    min_size: Optional[int] = None,
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Exact target size (width, height)
        max_size: Maximum dimension
        min_size: Minimum dimension
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if target_size is not None:
        return cv2.resize(image, target_size)
    
    scale = 1.0
    
    if max_size is not None:
        max_dim = max(h, w)
        if max_dim > max_size:
            scale = max_size / max_dim
    
    if min_size is not None:
        min_dim = min(h, w)
        if min_dim < min_size:
            scale = max(scale, min_size / min_dim)
    
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    
    return image


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply denoising to improve image quality."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle in degrees."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image to bounding box.
    
    Args:
        image: Input image
        bbox: (x1, y1, x2, y2) coordinates
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_image_info(image: np.ndarray) -> dict:
    """Get image information."""
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "aspect_ratio": w / h,
        "size_bytes": image.nbytes,
    }
