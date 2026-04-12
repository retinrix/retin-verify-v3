#!/usr/bin/env python3
"""
Generate synthetic Algerian ID card images for document detection training.

This creates synthetic ID card images with various backgrounds, rotations,
and lighting conditions to train the YOLOX document detection model.
"""

import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np


# Algerian ID card dimensions (aspect ratio ~1.58:1)
ID_CARD_WIDTH = 856
ID_CARD_HEIGHT = 540


def generate_background(width, height):
    """Generate a random background image."""
    bg_type = random.choice(['solid', 'gradient', 'noise', 'texture'])
    
    if bg_type == 'solid':
        color = (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240))
        image = Image.new('RGB', (width, height), color=color)
    
    elif bg_type == 'gradient':
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        color1 = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        color2 = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    elif bg_type == 'noise':
        noise = np.random.randint(150, 240, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(noise)
    
    else:  # texture
        image = Image.new('RGB', (width, height), (220, 220, 220))
        draw = ImageDraw.Draw(image)
        for _ in range(1000):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            color = (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240))
            draw.ellipse([x, y, x+size, y+size], fill=color)
    
    return image


def generate_id_card_base():
    """Generate a synthetic Algerian ID card base image."""
    # Create base card with Algerian ID colors (green/white theme)
    card = Image.new('RGB', (ID_CARD_WIDTH, ID_CARD_HEIGHT), (245, 245, 240))
    draw = ImageDraw.Draw(card)
    
    # Add green header (Algerian flag color)
    header_height = 80
    draw.rectangle([0, 0, ID_CARD_WIDTH, header_height], fill=(0, 120, 60))
    
    # Add emblem placeholder (circle)
    emblem_x = 80
    emblem_y = 120
    emblem_size = 100
    draw.ellipse([emblem_x, emblem_y, emblem_x + emblem_size, emblem_y + emblem_size], 
                 fill=(220, 220, 210), outline=(0, 100, 50), width=2)
    
    # Add photo placeholder
    photo_x = ID_CARD_WIDTH - 200
    photo_y = 100
    photo_size = 140
    draw.rectangle([photo_x, photo_y, photo_x + photo_size, photo_y + photo_size * 1.3], 
                   fill=(230, 230, 220), outline=(100, 100, 100), width=2)
    
    # Add text fields
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add field labels
    fields = [
        ("NOM/SURNAME:", 220, 130),
        ("PRENOM/GIVEN NAMES:", 220, 170),
        ("DATE DE NAISSANCE/DOB:", 220, 210),
        ("LIEU DE NAISSANCE/POB:", 220, 250),
        ("NUMERO/NUMBER:", 220, 290),
    ]
    
    for label, x, y in fields:
        draw.text((x, y), label, fill=(80, 80, 80), font=font)
    
    # Add MRZ zone at bottom
    mrz_y = ID_CARD_HEIGHT - 100
    draw.rectangle([50, mrz_y, ID_CARD_WIDTH - 50, ID_CARD_HEIGHT - 20], 
                   fill=(240, 240, 235), outline=(150, 150, 150))
    
    # Add MRZ-like text
    try:
        mrz_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    except:
        mrz_font = font
    
    draw.text((60, mrz_y + 15), "IDDZA" + "X" * 25, fill=(50, 50, 50), font=mrz_font)
    draw.text((60, mrz_y + 45), "X" * 30, fill=(50, 50, 50), font=mrz_font)
    
    # Add some noise/texture
    noise = np.random.normal(0, 3, (ID_CARD_HEIGHT, ID_CARD_WIDTH, 3)).astype(np.uint8)
    noise_img = Image.fromarray(noise)
    card = Image.blend(card, noise_img, 0.05)
    
    return card


def apply_transformations(image, max_rotation=15, max_scale=0.15):
    """Apply random transformations to the ID card."""
    # Rotation
    angle = random.uniform(-max_rotation, max_rotation)
    image = image.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    
    # Scale
    scale = 1 + random.uniform(-max_scale, max_scale)
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Perspective distortion (simulate camera angle)
    if random.random() > 0.5:
        width, height = image.size
        
        # Random perspective points
        margin = int(min(width, height) * 0.05)
        
        src_points = [
            (0, 0), (width, 0), (width, height), (0, height)
        ]
        
        dst_points = [
            (random.randint(0, margin), random.randint(0, margin)),
            (width - random.randint(0, margin), random.randint(0, margin)),
            (width - random.randint(0, margin), height - random.randint(0, margin)),
            (random.randint(0, margin), height - random.randint(0, margin))
        ]
        
        coeffs = find_coeffs(dst_points, src_points)
        image = image.transform((width, height), Image.Transform.PERSPECTIVE,
                                coeffs, Image.Resampling.BICUBIC)
    
    return image


def find_coeffs(pa, pb):
    """Find perspective transform coefficients."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def composite_onto_background(card, background):
    """Composite the ID card onto a background."""
    bg_width, bg_height = background.size
    card_width, card_height = card.size
    
    # Random position (ensure card is mostly visible)
    max_x = bg_width - card_width
    max_y = bg_height - card_height
    
    if max_x > 0:
        x = random.randint(0, max_x)
    else:
        x = max_x // 2
    if max_y > 0:
        y = random.randint(0, max_y)
    else:
        y = max_y // 2
    
    # Paste card onto background
    if card.mode == 'RGBA':
        background.paste(card, (x, y), card)
    else:
        background.paste(card, (x, y))
    
    # Calculate bounding box (YOLO format: x_center, y_center, width, height)
    x_center = (x + card_width / 2) / bg_width
    y_center = (y + card_height / 2) / bg_height
    width_norm = card_width / bg_width
    height_norm = card_height / bg_height
    
    return background, (x_center, y_center, width_norm, height_norm)


def apply_image_effects(image):
    """Apply random image effects."""
    # Blur
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    
    # Brightness
    if random.random() > 0.6:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Contrast
    if random.random() > 0.6:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.3))
    
    # Color balance
    if random.random() > 0.7:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Add noise
    if random.random() > 0.6:
        noise = np.random.normal(0, random.randint(5, 15), 
                                  (image.height, image.width, 3)).astype(np.uint8)
        noise_img = Image.fromarray(noise)
        image = Image.blend(image, noise_img, random.uniform(0.03, 0.1))
    
    return image


def generate_document_dataset(output_dir, num_samples, img_size=640):
    """Generate a complete document detection dataset."""
    output_path = Path(output_dir)
    
    # Create directories
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic document images...")
    
    for i in range(num_samples):
        # Generate background
        background = generate_background(img_size, img_size)
        
        # Generate ID card
        card = generate_id_card_base()
        
        # Apply transformations
        card = apply_transformations(card)
        
        # Composite onto background
        final_image, bbox = composite_onto_background(card, background)
        
        # Apply image effects
        final_image = apply_image_effects(final_image)
        
        # Save image
        img_path = images_dir / f"doc_{i:06d}.jpg"
        final_image.save(img_path, quality=random.randint(85, 95))
        
        # Save label (YOLO format: class x_center y_center width height)
        label_path = labels_dir / f"doc_{i:06d}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} images")
    
    print(f"\nDataset generation complete!")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Total: {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic document detection training data')
    parser.add_argument('--output-dir', type=str, default='data/document_detection',
                        help='Output directory for dataset')
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Number of samples to generate')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size (square)')
    
    args = parser.parse_args()
    
    generate_document_dataset(args.output_dir, args.num_samples, args.img_size)


if __name__ == '__main__':
    main()
