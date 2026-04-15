#!/usr/bin/env python3
"""
Train YOLOX model for Algerian ID card document detection.

This script handles the complete training pipeline:
1. Install YOLOX dependencies
2. Prepare dataset in COCO format
3. Train the model
4. Export to ONNX
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import shutil


def install_yolox():
    """Install YOLOX framework."""
    print("Installing YOLOX...")
    
    yolox_dir = Path("yolox")
    if not yolox_dir.exists():
        subprocess.run([
            "git", "clone", "https://github.com/Megvii-BaseDetection/YOLOX.git", "yolox"
        ], check=True)
    
    # Install dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "yolox"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
    
    print("YOLOX installed successfully!")


def convert_to_coco(data_dir, output_dir):
    """Convert YOLO format labels to COCO format."""
    print("Converting labels to COCO format...")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
        
        # Create COCO structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "id_card"}]
        }
        
        annotation_id = 0
        
        # Process each image
        for img_file in sorted(images_dir.glob("*.jpg")):
            img_id = int(img_file.stem.split('_')[1])
            
            # Get image dimensions
            from PIL import Image
            with Image.open(img_file) as img:
                width, height = img.size
            
            # Add image info
            coco_data["images"].append({
                "id": img_id,
                "file_name": img_file.name,
                "width": width,
                "height": height
            })
            
            # Read label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height
                        
                        # Convert to COCO bbox format (x, y, w, h)
                        x = x_center - w / 2
                        y = y_center - h / 2
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
        
        # Save COCO annotation
        with open(output_path / f"{split}_annotations.json", 'w') as f:
            json.dump(coco_data, f)
        
        print(f"  {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")


def create_yolox_config(exp_name, data_dir, num_classes=1):
    """Create YOLOX experiment configuration."""
    config_content = f'''#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "{exp_name}"
        self.output_dir = "./YOLOX_outputs"
        
        # Dataset config
        self.num_classes = {num_classes}
        self.data_dir = "{data_dir}"
        
        # Training config
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        
        # Input config
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.translate = 0.0  # FIX: prevent bbox out-of-bounds crash during no-aug phase
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import COCODataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection
        import torch
        
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file="train_annotations.json",
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache_img,
        )
        
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        
        if is_distributed:
            batch_size = batch_size // torch.cuda.device_count()
        
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        
        dataloader_kwargs = {{"num_workers": self.data_num_workers, "pin_memory": True}}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        
        return DataLoader(dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import COCODataset, ValTransform
        import torch
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file="val_annotations.json",
            name="",
            img_size=self.test_size,
            preproc=ValTransform(legacy=False),
        )
        
        if is_distributed:
            batch_size = batch_size // torch.cuda.device_count()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {{"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler}}
        dataloader_kwargs["batch_size"] = batch_size
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator
        
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
'''
    
    # Save config
    config_dir = Path("yolox/exps/custom")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{exp_name}.py"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config saved to {config_path}")
    return config_path


def train_model(config_path, batch_size, epochs, pretrained=None):
    """Train the YOLOX model."""
    print(f"\nStarting training...")
    print(f"  Config: {config_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    cmd = [
        sys.executable, "yolox/tools/train.py",
        "-f", str(config_path),
        "-d", "1",  # Use 1 GPU
        "-b", str(batch_size),
        "--fp16",
        "-o",
    ]
    
    if pretrained:
        cmd.extend(["-c", pretrained])
    
    subprocess.run(cmd, check=True)


def export_onnx(config_path, checkpoint_path, output_path):
    """Export trained model to ONNX format."""
    print(f"\nExporting to ONNX...")
    
    cmd = [
        sys.executable, "yolox/tools/export_onnx.py",
        "--output-name", str(output_path),
        "-f", str(config_path),
        "-c", str(checkpoint_path)
    ]
    
    subprocess.run(cmd, check=True)
    print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOX for document detection')
    parser.add_argument('--data-dir', type=str, default='data/document_detection',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--exp-name', type=str, default='yolox_idcard',
                        help='Experiment name')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip YOLOX installation')
    
    args = parser.parse_args()
    
    # Install YOLOX
    if not args.skip_install:
        install_yolox()
    
    # Convert dataset to COCO format
    coco_dir = Path(args.data_dir) / "coco"
    convert_to_coco(args.data_dir, coco_dir)
    
    # Create config
    config_path = create_yolox_config(args.exp_name, str(coco_dir.absolute()))
    
    # Train
    train_model(config_path, args.batch_size, args.epochs, args.pretrained)
    
    # Export to ONNX
    checkpoint_path = f"yolox/YOLOX_outputs/{args.exp_name}/best_ckpt.pth"
    output_path = Path(args.output_dir) / "yolox_idcard.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if Path(checkpoint_path).exists():
        export_onnx(config_path, checkpoint_path, output_path)
        print(f"\n✅ Training complete! Model saved to {output_path}")
    else:
        print(f"\n⚠️ Checkpoint not found at {checkpoint_path}")
        print("Training may have failed or checkpoint name is different.")


if __name__ == '__main__':
    main()
