#!/usr/bin/env python3
"""
Generate synthetic MRZ (Machine Readable Zone) images for training.

This script generates realistic MRZ images with various augmentations
to train the PaddleOCR model.
"""

import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np


def generate_mrz_data():
    """Generate random MRZ data following ICAO 9303 standard."""
    # Document number (9 digits + 1 check digit)
    doc_number = ''.join([str(random.randint(0, 9)) for _ in range(9)])
    doc_check = str(random.randint(0, 9))
    
    # Date of birth (YYMMDD + check digit)
    year = random.randint(50, 99)  # 1950-1999 for older people
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    dob = f"{year:02d}{month:02d}{day:02d}"
    dob_check = str(random.randint(0, 9))
    
    # Sex
    sex = random.choice(['M', 'F'])
    
    # Expiry date (YYMMDD + check digit)
    exp_year = random.randint(25, 35)  # 2025-2035
    exp_month = random.randint(1, 12)
    exp_day = random.randint(1, 28)
    expiry = f"{exp_year:02d}{exp_month:02d}{exp_day:02d}"
    exp_check = str(random.randint(0, 9))
    
    # Nationality
    nationality = "DZA"
    
    # Names (Algerian names)
    surnames = [
        "BOUTIGHANE", "BENALI", "BOUCHIKHI", "BENMANSOUR", "BELHADJ",
        "BOULAHBAL", "BOUMAAZA", "BENYAHIA", "BENSALAH", "BENAMOR"
    ]
    given_names = [
        "MOHAMED", "AHMED", "ALI", "OMAR", "KHALED",
        "FATIMA", "AICHA", "KARIMA", "SAMIRA", "NADIA"
    ]
    
    surname = random.choice(surnames)
    given_name = random.choice(given_names)
    
    # Build MRZ lines
    # Line 1: Document type + issuing country + document number + check digit
    line1 = f"IDDZA{doc_number}{doc_check}<<<<<<<<<<<<<<<"
    
    # Line 2: DOB + check + sex + expiry + check + nationality + optional data + composite check
    line2 = f"{dob}{dob_check}{sex}{expiry}{exp_check}{nationality}<<<<<<<<<<<{exp_check}"
    
    # Line 3: Name (surname << given names)
    name_part = f"{surname}<<{given_name}"
    line3 = name_part + "<" * (30 - len(name_part))
    
    return line1, line2, line3


def create_mrz_image(line1, line2, line3, output_path, augment=True):
    """Create an MRZ image with optional augmentations."""
    # Image dimensions
    img_width = 1000
    img_height = 200
    
    # Background color (slightly off-white for realism)
    bg_color = (
        random.randint(240, 255),
        random.randint(240, 255),
        random.randint(240, 255)
    )
    
    image = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to load a monospace font, fallback to default
    font_size = random.randint(26, 32)
    try:
        # Try system monospace fonts
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
            '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
            '/System/Library/Fonts/Menlo.ttc',  # macOS
            'C:\\Windows\\Fonts\\consola.ttf',  # Windows
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Add noise
    if augment and random.random() > 0.3:
        noise = np.random.normal(0, random.randint(3, 8), (img_height, img_width, 3)).astype(np.uint8)
        noise_img = Image.fromarray(noise)
        image = Image.blend(image, noise_img, random.uniform(0.05, 0.15))
        draw = ImageDraw.Draw(image)
    
    # Draw MRZ lines with slight variations
    y_start = 30
    line_spacing = 50
    
    for i, line in enumerate([line1, line2, line3]):
        y_offset = y_start + i * line_spacing
        
        # Add slight rotation to each line
        if augment and random.random() > 0.5:
            angle = random.uniform(-1.5, 1.5)
            text_img = Image.new('RGBA', (img_width, 60), (*bg_color, 0))
            text_draw = ImageDraw.Draw(text_img)
            text_draw.text((50, 10), line, fill=(0, 0, 0), font=font)
            text_img = text_img.rotate(angle, expand=0)
            image.paste(text_img, (0, y_offset), text_img)
        else:
            # Add slight position variation
            x_offset = random.randint(45, 55)
            y_offset += random.randint(-2, 2)
            draw.text((x_offset, y_offset), line, fill=(0, 0, 0), font=font)
    
    # Apply augmentations
    if augment:
        # Gaussian blur
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))
        
        # Contrast adjustment
        if random.random() > 0.7:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Brightness adjustment
        if random.random() > 0.7:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # Save image
    image.save(output_path, quality=random.randint(85, 95))
    
    return image


def generate_dataset(output_dir, num_samples, split_ratio=0.9):
    """Generate a complete MRZ dataset."""
    output_path = Path(output_dir)
    
    # Create directories
    train_dir = output_path / 'train' / 'images'
    val_dir = output_path / 'val' / 'images'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    train_labels = []
    val_labels = []
    
    print(f"Generating {num_samples} synthetic MRZ images...")
    
    for i in range(num_samples):
        # Generate MRZ data
        line1, line2, line3 = generate_mrz_data()
        label = f"{line1}\n{line2}\n{line3}"
        
        # Determine split
        is_train = i < int(num_samples * split_ratio)
        
        if is_train:
            img_path = train_dir / f"mrz_{i:06d}.jpg"
            label_path = f"images/mrz_{i:06d}.jpg\t{label}"
            train_labels.append(label_path)
        else:
            img_path = val_dir / f"mrz_{i:06d}.jpg"
            label_path = f"images/mrz_{i:06d}.jpg\t{label}"
            val_labels.append(label_path)
        
        # Create image
        create_mrz_image(line1, line2, line3, img_path, augment=is_train)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} images")
    
    # Save label files
    with open(output_path / 'train' / 'label.txt', 'w') as f:
        f.write('\n'.join(train_labels))
    
    with open(output_path / 'val' / 'label.txt', 'w') as f:
        f.write('\n'.join(val_labels))
    
    print(f"\nDataset generation complete!")
    print(f"  Train: {len(train_labels)} images")
    print(f"  Val: {len(val_labels)} images")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic MRZ training data')
    parser.add_argument('--output-dir', type=str, default='data/mrz_ocr',
                        help='Output directory for dataset')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--split-ratio', type=float, default=0.9,
                        help='Train/val split ratio (default: 0.9)')
    
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.num_samples, args.split_ratio)


if __name__ == '__main__':
    main()
