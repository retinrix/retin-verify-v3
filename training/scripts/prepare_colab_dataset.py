#!/usr/bin/env python3
"""
Prepare dataset for Google Colab training.

This script helps you:
1. Organize your images into train/val splits
2. Create COCO format annotations
3. Generate a zip file ready for Google Drive upload

Usage:
    python prepare_colab_dataset.py --input-dir /path/to/images --output-dir ./dataset
"""

import os
import sys
import json
import random
import argparse
import shutil
from pathlib import Path
from PIL import Image
import zipfile


def resolve_path(path):
    """Resolve a path, following symlinks if present."""
    p = Path(path)
    # Resolve to absolute path and follow symlinks
    return p.resolve()


def create_coco_annotation(image_files, output_path):
    """Create COCO format annotation file."""
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "id_card"}]
    }
    
    annotation_id = 0
    
    for img_id, img_path in enumerate(image_files, 1):
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        # Create a default annotation (full image - you'll need to adjust this)
        # For proper training, you should use a labeling tool like LabelImg
        margin = int(min(width, height) * 0.1)
        bbox = [margin, margin, width - 2*margin, height - 2*margin]
        
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": 0,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        annotation_id += 1
    
    # Save annotation
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return len(coco_data["images"]), len(coco_data["annotations"])


def prepare_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """Prepare dataset for Colab training."""
    
    # Resolve paths (follow symlinks)
    input_path = resolve_path(input_dir)
    output_path = Path(output_dir).resolve()
    
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Check if input path exists
    if not input_path.exists():
        print(f"❌ Input directory does not exist: {input_path}")
        return False
    
    # Check if it's a symlink and report
    if Path(input_dir).is_symlink():
        print(f"  (Resolved from symlink: {input_dir} -> {input_path})")
    
    # Find all images (including in subdirectories)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = []
    
    print(f"\nScanning for images in {input_path}...")
    
    # Walk through directory tree
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    # Filter to only files (not symlinks to directories)
    image_files = [f for f in image_files if f.is_file()]
    
    # Remove duplicates (in case of multiple extensions matching)
    image_files = list(dict.fromkeys(image_files))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        print(f"   Supported formats: {image_extensions}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val")
    
    # Create directories
    train_dir = output_path / 'train' / 'images'
    val_dir = output_path / 'val' / 'images'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train images
    print("\nCopying train images...")
    for img_file in train_files:
        try:
            shutil.copy2(img_file, train_dir / img_file.name)
        except Exception as e:
            print(f"  Warning: Failed to copy {img_file}: {e}")
    
    # Copy val images
    print("Copying val images...")
    for img_file in val_files:
        try:
            shutil.copy2(img_file, val_dir / img_file.name)
        except Exception as e:
            print(f"  Warning: Failed to copy {img_file}: {e}")
    
    # Create annotations
    print("\nCreating annotations...")
    
    train_img_count, train_ann_count = create_coco_annotation(
        train_files, output_path / 'train' / 'annotations.json'
    )
    val_img_count, val_ann_count = create_coco_annotation(
        val_files, output_path / 'val' / 'annotations.json'
    )
    
    print(f"  Train: {train_img_count} images, {train_ann_count} annotations")
    print(f"  Val: {val_img_count} images, {val_ann_count} annotations")
    
    # Create README
    readme_content = f"""# Dataset Information

Generated: {os.popen('date').read().strip()}
Total Images: {len(image_files)}
Train: {len(train_files)} ({train_ratio*100:.0f}%)
Val: {len(val_files)} ({(1-train_ratio)*100:.0f}%)

## Structure
```
dataset/
├── train/
│   ├── images/          # Training images
│   └── annotations.json # COCO format annotations
└── val/
    ├── images/          # Validation images
    └── annotations.json # COCO format annotations
```

## Upload to Google Drive

1. Zip this folder:
   ```bash
   zip -r algerian_id_cards.zip dataset/
   ```

2. Upload to Google Drive:
   - Go to https://drive.google.com
   - Create folder: `MyDrive/datasets/`
   - Upload `algerian_id_cards.zip`
   - Extract in Drive

3. Verify structure:
   ```
   MyDrive/datasets/algerian_id_cards/
   ├── train/
   │   ├── images/
   │   └── annotations.json
   └── val/
       ├── images/
       └── annotations.json
   ```

## Next Steps

1. Open `colab/yolox_document_detection.ipynb`
2. Upload to Google Colab
3. Run training cells

## Important Note

⚠️ The annotations in this dataset are AUTO-GENERATED with default bounding boxes.
For best results, you should use a proper annotation tool like LabelImg or CVAT
to create accurate bounding boxes around the ID cards.

"""
    
    with open(output_path / 'README.txt', 'w') as f:
        f.write(readme_content)
    
    print(f"\n✅ Dataset prepared at: {output_path.absolute()}")
    print(f"   Read the README.txt for upload instructions")
    
    return True


def create_zip(dataset_dir, output_zip):
    """Create a zip file of the dataset."""
    print(f"\nCreating zip file: {output_zip}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=dataset_dir)
                zipf.write(file_path, arcname)
    
    # Get file size
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"✅ Zip created: {output_zip} ({size_mb:.1f} MB)")
    
    return output_zip


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for Google Colab training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_colab_dataset.py --input-dir ~/Pictures/id_cards --output-dir ./dataset

  # With symlink (e.g., mounted drive)
  python prepare_colab_dataset.py -i /mnt/d/dataset_280326 -o ./dataset

  # Custom train/val split
  python prepare_colab_dataset.py -i ~/Pictures/id_cards -o ./dataset --train-ratio 0.85

  # Create zip for easy upload
  python prepare_colab_dataset.py -i ~/Pictures/id_cards -o ./dataset --zip
        """
    )
    
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory containing ID card images (symlinks will be followed)')
    parser.add_argument('-o', '--output-dir', default='./dataset',
                        help='Output directory (default: ./dataset)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--zip', action='store_true',
                        help='Create zip file for easy upload')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Preparation for Google Colab")
    print("="*60)
    
    # Prepare dataset
    if prepare_dataset(args.input_dir, args.output_dir, args.train_ratio, args.seed):
        # Create zip if requested
        if args.zip:
            zip_path = str(Path(args.output_dir)) + '.zip'
            create_zip(args.output_dir, zip_path)
            
            print(f"\n{'='*60}")
            print("Next Steps:")
            print(f"{'='*60}")
            print(f"1. Upload {zip_path} to Google Drive")
            print(f"2. Extract in Drive: MyDrive/datasets/algerian_id_cards/")
            print(f"3. Open colab/yolox_document_detection.ipynb")
            print(f"4. Run training cells")
    else:
        print("\n❌ Dataset preparation failed")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
