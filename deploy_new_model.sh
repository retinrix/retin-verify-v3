#!/bin/bash
# deploy_new_model.sh - Helper to deploy the newly trained model

set -e

echo "=========================================="
echo "Deploy New YOLOX Model"
echo "=========================================="

# Check if model was provided as argument
MODEL_PATH="${1:-$HOME/Downloads/yolox_idcard.onnx}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo ""
    echo "Please download the ONNX model from Google Drive:"
    echo "  MyDrive/models/yolox_idcard.onnx"
    echo ""
    echo "Usage: ./deploy_new_model.sh [path/to/yolox_idcard.onnx]"
    exit 1
fi

# Backup old model
if [ -f "models/yolox_idcard.onnx" ]; then
    BACKUP="models/yolox_idcard_$(date +%Y%m%d_%H%M%S).onnx"
    cp "models/yolox_idcard.onnx" "$BACKUP"
    echo "✅ Backed up old model to: $BACKUP"
fi

# Copy new model
cp "$MODEL_PATH" "models/yolox_idcard.onnx"
echo "✅ Deployed new model to: models/yolox_idcard.onnx"

# Quick test
source venv/bin/activate
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from document_detection.detector import DocumentDetector
d = DocumentDetector()
print(f'Model loaded successfully: {d.model_path}')
print(f'Input size: {d.input_size}')
print(f'Classes: {d.class_names}')
"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo "Test with:"
echo "  python3 test_document_detection.py path/to/image.jpg"
echo "  python3 test_document_detection.py --camera"
