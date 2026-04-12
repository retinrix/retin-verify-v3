#!/usr/bin/env python3
"""
Master training script for all Retin-Verify V3 models.

This script orchestrates the training of all models:
1. Document Detection (YOLOX)
2. MRZ OCR (PaddleOCR)
3. Face Detection (YuNet - download only)
4. Security Features (Custom)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def generate_data(args):
    """Generate synthetic training data."""
    print("\n" + "="*60)
    print("STEP 1: Generating Synthetic Training Data")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    
    # Generate document detection data
    if not args.skip_document_data:
        print("\n1.1 Generating document detection data...")
        cmd = f"{sys.executable} {scripts_dir}/generate_synthetic_documents.py --output-dir data/document_detection --num-samples {args.num_document_samples}"
        if not run_command(cmd, "Document data generation"):
            return False
    
    # Generate MRZ OCR data
    if not args.skip_mrz_data:
        print("\n1.2 Generating MRZ OCR data...")
        cmd = f"{sys.executable} {scripts_dir}/generate_synthetic_mrz.py --output-dir data/mrz_ocr --num-samples {args.num_mrz_samples}"
        if not run_command(cmd, "MRZ data generation"):
            return False
    
    return True


def train_document_detection(args):
    """Train document detection model."""
    print("\n" + "="*60)
    print("STEP 2: Training Document Detection Model (YOLOX)")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    
    cmd = f"{sys.executable} {scripts_dir}/train_yolox.py --data-dir data/document_detection --output-dir models --batch-size {args.batch_size_yolox} --epochs {args.epochs_yolox}"
    
    if args.skip_install:
        cmd += " --skip-install"
    
    return run_command(cmd, "YOLOX training")


def train_mrz_ocr(args):
    """Train MRZ OCR model."""
    print("\n" + "="*60)
    print("STEP 3: Training MRZ OCR Model (PaddleOCR)")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    
    cmd = f"{sys.executable} {scripts_dir}/train_paddleocr.py --data-dir data/mrz_ocr --output-dir models"
    
    if args.skip_install:
        cmd += " --skip-install"
    
    return run_command(cmd, "PaddleOCR training")


def download_face_detection(args):
    """Download face detection model."""
    print("\n" + "="*60)
    print("STEP 4: Downloading Face Detection Model (YuNet)")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    
    cmd = f"{sys.executable} {scripts_dir}/download_yunet.py --output-dir models"
    
    return run_command(cmd, "YuNet download")


def copy_to_production():
    """Copy trained models to production."""
    print("\n" + "="*60)
    print("STEP 5: Copying Models to Production")
    print("="*60)
    
    src_dir = Path("models")
    dst_dir = Path("../models")
    
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_copy = [
        ("yolox_idcard.onnx", "Document detection"),
        ("paddleocr_mrz", "MRZ OCR"),
        ("face_detection_yunet.onnx", "Face detection"),
    ]
    
    for model_name, description in models_to_copy:
        src = src_dir / model_name
        dst = dst_dir / model_name
        
        if src.exists():
            if src.is_dir():
                import shutil
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                import shutil
                shutil.copy2(src, dst)
            print(f"✅ Copied {description} model")
        else:
            print(f"⚠️ {description} model not found at {src}")
    
    print(f"\nModels copied to {dst_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Train all Retin-Verify V3 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models from scratch
  python train_all.py

  # Skip data generation (use existing data)
  python train_all.py --skip-document-data --skip-mrz-data

  # Quick test with fewer samples
  python train_all.py --num-document-samples 100 --num-mrz-samples 500 --epochs-yolox 10

  # Only download pre-trained models
  python train_all.py --skip-document-data --skip-mrz-data --skip-train-yolox --skip-train-mrz
        """
    )
    
    # Data generation options
    parser.add_argument('--num-document-samples', type=int, default=2000,
                        help='Number of document detection samples (default: 2000)')
    parser.add_argument('--num-mrz-samples', type=int, default=5000,
                        help='Number of MRZ OCR samples (default: 5000)')
    parser.add_argument('--skip-document-data', action='store_true',
                        help='Skip document detection data generation')
    parser.add_argument('--skip-mrz-data', action='store_true',
                        help='Skip MRZ OCR data generation')
    
    # Training options
    parser.add_argument('--batch-size-yolox', type=int, default=16,
                        help='YOLOX batch size (default: 16)')
    parser.add_argument('--epochs-yolox', type=int, default=100,
                        help='YOLOX training epochs (default: 100)')
    parser.add_argument('--skip-train-yolox', action='store_true',
                        help='Skip YOLOX training')
    parser.add_argument('--skip-train-mrz', action='store_true',
                        help='Skip PaddleOCR training')
    
    # Other options
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip framework installation (use existing)')
    parser.add_argument('--skip-copy', action='store_true',
                        help='Skip copying models to production')
    parser.add_argument('--skip-face', action='store_true',
                        help='Skip face detection model download')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Retin-Verify V3 - Model Training Pipeline          ║
╚══════════════════════════════════════════════════════════════╝

This script will train all models for the V3 system:
  1. Document Detection (YOLOX) - Detect ID cards in images
  2. MRZ OCR (PaddleOCR) - Read MRZ text from ID cards
  3. Face Detection (YuNet) - Detect and extract face photos
  4. Security Features - Detect holograms and security elements

Requirements:
  - NVIDIA GPU with 8GB+ VRAM
  - CUDA and cuDNN installed
  - Python 3.8+
  - ~10GB free disk space

Press Ctrl+C to cancel at any time.
""")
    
    # Change to training directory
    os.chdir(Path(__file__).parent.parent)
    
    success = True
    
    # Step 1: Generate data
    if not (args.skip_document_data and args.skip_mrz_data):
        if not generate_data(args):
            print("\n❌ Data generation failed!")
            success = False
    
    # Step 2: Train document detection
    if success and not args.skip_train_yolox:
        if not train_document_detection(args):
            print("\n⚠️ YOLOX training failed, continuing...")
            success = False
    
    # Step 3: Train MRZ OCR
    if not args.skip_train_mrz:
        if not train_mrz_ocr(args):
            print("\n⚠️ PaddleOCR training failed, continuing...")
    
    # Step 4: Download face detection
    if not args.skip_face:
        if not download_face_detection(args):
            print("\n⚠️ Face detection download failed, continuing...")
    
    # Step 5: Copy to production
    if not args.skip_copy:
        copy_to_production()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nTrained models are available in:")
    print(f"  - Training: {Path('models').absolute()}")
    print(f"  - Production: {Path('../models').absolute()}")
    print("\nNext steps:")
    print("  1. Test the models: python scripts/validate_models.py")
    print("  2. Start the API: cd .. && uvicorn api.main:app --reload")
    print("  3. Run full pipeline tests")


if __name__ == '__main__':
    main()
