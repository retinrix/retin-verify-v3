#!/bin/bash
# Setup script for Retin-Verify V3

set -e

echo "============================================"
echo "Retin-Verify V3 - Environment Setup"
echo "============================================"
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment
echo
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo
echo "Creating directories..."
mkdir -p models data/collected data/processed data/synthetic

# Run tests
echo
echo "Running tests..."
pytest tests/ -v --tb=short

echo
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo
echo "To run tests:"
echo "  pytest tests/ -v"
echo
echo "To start API server:"
echo "  uvicorn api.main:app --reload"
echo
