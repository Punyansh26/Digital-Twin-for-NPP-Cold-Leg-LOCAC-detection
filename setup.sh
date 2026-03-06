#!/bin/bash

# AP1000 Digital Twin - Setup Script (Linux/Mac)

echo "======================================================================"
echo "AP1000 Digital Twin - Automated Setup"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
python_version=$(python3 --version 2>&1)
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ $python_version"

# Create virtual environment (optional)
echo ""
echo "Creating virtual environment (optional)..."
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed"

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA support..."
read -p "Install PyTorch with CUDA? (Recommended for GPU) (y/n): " install_cuda
if [ "$install_cuda" = "y" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    echo "✓ PyTorch with CUDA installed"
else
    echo "Skipping CUDA installation. CPU-only mode will be used."
fi

# Create directory structure
echo ""
echo "Setting up directory structure..."
python3 scripts/create_gitkeep.py
echo "✓ Directory structure created"

# Test imports
echo ""
echo "Testing installation..."
python3 -c "import torch; import numpy; import pandas; import yaml; print('✓ All imports successful')"
if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed"
    exit 1
fi

# Check CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Setup complete
echo ""
echo "======================================================================"
echo "✓ Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Read QUICKSTART.md for a 10-minute tutorial"
echo "  2. Run the pipeline: python3 run_pipeline.py --use-mock-data"
echo "  3. Review results in results/plots/"
echo ""
echo "Quick test:"
echo "  python3 run_pipeline.py --use-mock-data"
echo ""
echo "For help:"
echo "  python3 run_pipeline.py --help"
echo ""
