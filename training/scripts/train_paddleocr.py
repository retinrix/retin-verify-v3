#!/usr/bin/env python3
"""
Train PaddleOCR model for MRZ text recognition.

This script handles the complete training pipeline:
1. Install PaddleOCR dependencies
2. Prepare dataset
3. Train the model
4. Export to ONNX
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def install_paddleocr():
    """Install PaddleOCR framework."""
    print("Installing PaddleOCR...")
    
    paddleocr_dir = Path("paddleocr")
    if not paddleocr_dir.exists():
        subprocess.run([
            "git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git", "paddleocr"
        ], check=True)
    
    # Install dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "paddleocr/requirements.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imgaug", "lmdb", "paddle2onnx"], check=True)
    
    print("PaddleOCR installed successfully!")


def create_mrz_dict():
    """Create character dictionary for MRZ."""
    mrz_dict = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
    
    dict_path = Path("paddleocr/ppocr/utils/mrz_dict.txt")
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dict_path, 'w') as f:
        for char in mrz_dict:
            f.write(char + '\n')
    
    print(f"MRZ dictionary created with {len(mrz_dict)} characters")
    return dict_path


def create_config(data_dir, output_dir):
    """Create PaddleOCR training configuration."""
    config_content = f'''Global:
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {output_dir}/mrz_rec
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model:
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img:
  character_dict_path: paddleocr/ppocr/utils/mrz_dict.txt
  max_text_length: 30
  infer_mode: false
  use_space_char: false
  distributed: false
  save_res_path: ./checkpoints/rec/predicts_ppocrv3.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    out_channels_list: 
      CTCLabelDecode: 40
      SARLabelDecode: 41

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: false

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {data_dir}/train
    label_file_list: [{data_dir}/train/label.txt]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape: [48, 320, 3]
          max_text_length: 30
      - RecAug:
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: [image, label, length]
  loader:
    shuffle: true
    batch_size_per_card: 128
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {data_dir}/val
    label_file_list: [{data_dir}/val/label.txt]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: [image, label, length]
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 128
    num_workers: 4
'''
    
    config_path = Path("configs/rec_mrz.yml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config saved to {config_path}")
    return config_path


def train_model(config_path):
    """Train the PaddleOCR model."""
    print(f"\nStarting training...")
    print(f"  Config: {config_path}")
    
    cmd = [
        sys.executable, "paddleocr/tools/train.py",
        "-c", str(config_path)
    ]
    
    subprocess.run(cmd, check=True)


def export_model(config_path, checkpoint_dir, output_dir):
    """Export trained model to inference format."""
    print(f"\nExporting model...")
    
    # Export to Paddle inference format
    cmd = [
        sys.executable, "paddleocr/tools/export_model.py",
        "-c", str(config_path),
        "-o", f"Global.checkpoints={checkpoint_dir}/best_accuracy",
        "-o", f"Global.save_inference_dir={output_dir}"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Model exported to {output_dir}")


def export_onnx(paddle_model_dir, onnx_path):
    """Export Paddle model to ONNX format."""
    print(f"\nConverting to ONNX...")
    
    cmd = [
        "paddle2onnx",
        "--model_dir", str(paddle_model_dir),
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(onnx_path),
        "--opset_version", "11",
        "--enable_onnx_checker", "True"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"ONNX model saved to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description='Train PaddleOCR for MRZ recognition')
    parser.add_argument('--data-dir', type=str, default='data/mrz_ocr',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip PaddleOCR installation')
    
    args = parser.parse_args()
    
    # Install PaddleOCR
    if not args.skip_install:
        install_paddleocr()
    
    # Create MRZ dictionary
    create_mrz_dict()
    
    # Create config
    config_path = create_config(args.data_dir, args.output_dir)
    
    # Train
    train_model(config_path)
    
    # Export models
    checkpoint_dir = f"{args.output_dir}/mrz_rec"
    inference_dir = f"{args.output_dir}/paddleocr_mrz"
    
    if Path(checkpoint_dir).exists():
        export_model(config_path, checkpoint_dir, inference_dir)
        
        # Export to ONNX
        onnx_path = Path(args.output_dir) / "paddleocr_mrz.onnx"
        export_onnx(inference_dir, onnx_path)
        
        print(f"\n✅ Training complete!")
        print(f"   Paddle model: {inference_dir}")
        print(f"   ONNX model: {onnx_path}")
    else:
        print(f"\n⚠️ Checkpoint not found at {checkpoint_dir}")
        print("Training may have failed or checkpoint name is different.")


if __name__ == '__main__':
    main()
