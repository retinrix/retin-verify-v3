#!/usr/bin/env python3
"""
Prepare dataset for Google Colab training from LabelMe JSON annotations.

Supports:
- front/  -> images with id-front JSON annotations
- back/   -> images with id-back JSON annotations
- no-card/-> negative samples (no JSON = no annotations)

Converts LabelMe polygon annotations to tight bounding boxes for YOLOX training.

Usage:
    python prepare_colab_dataset.py -i data/collected -o ./algerian_id_cards --zip
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
    return Path(path).resolve()


CLASS_MAP = {
    "id-front": 0,
    "id-back": 1,
}


def parse_labelme_json(json_path):
    """
    Parse a LabelMe JSON file and return list of (xmin, ymin, xmax, ymax, class_id).
    Converts polygon points to tight bounding boxes.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    objects = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        if label not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[label]

        points = shape.get("points", [])
        if not points:
            continue

        # Calculate tight bounding box from polygon points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin = int(min(xs))
        ymin = int(min(ys))
        xmax = int(max(xs))
        ymax = int(max(ys))

        objects.append((xmin, ymin, xmax, ymax, class_id))

    return objects


def create_coco_annotation(image_files, json_lookup, output_path):
    """Create COCO format annotation file from images and JSON lookup."""

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "id-front"},
            {"id": 1, "name": "id-back"},
        ],
    }

    annotation_id = 0

    for img_id, img_path in enumerate(image_files, 1):
        with Image.open(img_path) as img:
            width, height = img.size

        coco_data["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        # Look up JSON annotation for this image
        ann_path = json_lookup.get(img_path.name)
        if ann_path and ann_path.exists():
            objs = parse_labelme_json(ann_path)
            for xmin, ymin, xmax, ymax, class_id in objs:
                # Ensure valid box
                xmin = max(0, min(xmin, width - 1))
                ymin = max(0, min(ymin, height - 1))
                xmax = max(0, min(xmax, width - 1))
                ymax = max(0, min(ymax, height - 1))
                bw = xmax - xmin
                bh = ymax - ymin
                if bw <= 0 or bh <= 0:
                    continue
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                annotation_id += 1
        # If no JSON exists, image is included as a negative sample (no annotations)

    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    return len(coco_data["images"]), len(coco_data["annotations"])


def prepare_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """Prepare dataset for Colab training."""

    input_path = resolve_path(input_dir)
    output_path = Path(output_dir).resolve()

    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")

    if not input_path.exists():
        print(f"❌ Input directory does not exist: {input_path}")
        return False

    # Expected subfolders
    front_dir = input_path / "front"
    back_dir = input_path / "back"
    no_card_dir = input_path / "no-card"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_images = []
    json_lookup = {}

    for subdir in [front_dir, back_dir, no_card_dir]:
        if not subdir.exists():
            continue
        for ext in image_extensions:
            for img_path in subdir.rglob(f"*{ext}"):
                if not img_path.is_file():
                    continue
                all_images.append(img_path)
                json_path = img_path.with_suffix(".json")
                if json_path.exists():
                    json_lookup[img_path.name] = json_path

    # Remove duplicates
    all_images = list(dict.fromkeys(all_images))

    if not all_images:
        print("❌ No images found")
        return False

    print(f"\nFound {len(all_images)} images")
    print(f"  - With JSON annotations: {len(json_lookup)}")
    print(f"  - Negative samples (no JSON): {len(all_images) - len(json_lookup)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    print(f"Split: {len(train_files)} train, {len(val_files)} val")

    # Create directories
    train_img_dir = output_path / "train" / "images"
    val_img_dir = output_path / "val" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    print("\nCopying train images...")
    for img_file in train_files:
        try:
            shutil.copy2(img_file, train_img_dir / img_file.name)
        except Exception as e:
            print(f"  Warning: Failed to copy {img_file}: {e}")

    print("Copying val images...")
    for img_file in val_files:
        try:
            shutil.copy2(img_file, val_img_dir / img_file.name)
        except Exception as e:
            print(f"  Warning: Failed to copy {img_file}: {e}")

    # Create annotations
    print("\nCreating annotations from LabelMe JSON files...")
    train_img_count, train_ann_count = create_coco_annotation(
        train_files, json_lookup, output_path / "train" / "annotations.json"
    )
    val_img_count, val_ann_count = create_coco_annotation(
        val_files, json_lookup, output_path / "val" / "annotations.json"
    )

    print(f"  Train: {train_img_count} images, {train_ann_count} annotations")
    print(f"  Val: {val_img_count} images, {val_ann_count} annotations")

    # Create README
    readme_content = f"""# Dataset Information

Generated: {os.popen('date').read().strip()}
Total Images: {len(all_images)}
Train: {len(train_files)} ({train_ratio*100:.0f}%)
Val: {len(val_files)} ({(1-train_ratio)*100:.0f}%)
Classes: id-front (0), id-back (1)

## Structure
```
dataset/
├── train/
│   ├── images/          # Training images
│   └── annotations.json # COCO format annotations from LabelMe JSON
└── val/
    ├── images/          # Validation images
    └── annotations.json # COCO format annotations from LabelMe JSON
```

## Upload to Google Drive

1. Zip this folder:
   ```bash
   zip -r algerian_id_cards.zip dataset/
   ```

2. Upload to Google Drive and extract to:
   ```
   MyDrive/retin-verify/v3/algerian_id_cards/
   ```

3. Open `colab/yolox_document_detection.ipynb` and verify `num_classes = 2`
"""

    with open(output_path / "README.txt", "w") as f:
        f.write(readme_content)

    print(f"\n✅ Dataset prepared at: {output_path.absolute()}")
    return True


def create_zip(dataset_dir, output_zip):
    """Create a zip file of the dataset."""
    print(f"\nCreating zip file: {output_zip}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=dataset_dir)
                zipf.write(file_path, arcname)
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"✅ Zip created: {output_zip} ({size_mb:.1f} MB)")
    return output_zip


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Google Colab training from LabelMe JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_colab_dataset.py -i data/collected -o ./algerian_id_cards --zip
        """,
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing front/, back/, no-card/ subfolders")
    parser.add_argument("-o", "--output-dir", default="./algerian_id_cards", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--zip", action="store_true", help="Create zip file")
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Preparation for Google Colab")
    print("=" * 60)

    if prepare_dataset(args.input_dir, args.output_dir, args.train_ratio, args.seed):
        if args.zip:
            zip_path = str(Path(args.output_dir)) + ".zip"
            create_zip(args.output_dir, zip_path)
            print(f"\n{'='*60}")
            print("Next Steps:")
            print(f"{'='*60}")
            print(f"1. Upload {zip_path} to Google Drive")
            print(f"2. Extract in Drive: MyDrive/retin-verify/v3/algerian_id_cards/")
            print(f"3. Open colab/yolox_document_detection.ipynb")
            print(f"4. Verify num_classes = 2 and run training")
    else:
        print("\n❌ Dataset preparation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
