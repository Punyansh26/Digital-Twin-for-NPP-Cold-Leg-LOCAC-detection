# AP1000 Digital Twin - Setup Script
# Run this after cloning/downloading the project

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "AP1000 Digital Twin - Automated Setup" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion" -ForegroundColor Green

# Create virtual environment (optional)
Write-Host "`nCreating virtual environment (optional)..." -ForegroundColor Yellow
$createVenv = Read-Host "Create virtual environment? (y/n)"
if ($createVenv -eq 'y') {
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment created and activated" -ForegroundColor Green
}

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray

pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Install PyTorch with CUDA
Write-Host "`nInstalling PyTorch with CUDA support..." -ForegroundColor Yellow
$installCuda = Read-Host "Install PyTorch with CUDA? (Recommended for GPU) (y/n)"
if ($installCuda -eq 'y') {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    Write-Host "✓ PyTorch with CUDA installed" -ForegroundColor Green
} else {
    Write-Host "Skipping CUDA installation. CPU-only mode will be used." -ForegroundColor Gray
}

# Create directory structure
Write-Host "`nSetting up directory structure..." -ForegroundColor Yellow
python scripts\create_gitkeep.py
Write-Host "✓ Directory structure created" -ForegroundColor Green

# Test imports
Write-Host "`nTesting installation..." -ForegroundColor Yellow
python -c "import torch; import numpy; import pandas; import yaml; print('✓ All imports successful')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Import test failed" -ForegroundColor Red
    exit 1
}

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Setup complete
Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "✓ Setup Complete!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Read QUICKSTART.md for a 10-minute tutorial" -ForegroundColor White
Write-Host "  2. Run the pipeline: python run_pipeline.py --use-mock-data" -ForegroundColor White
Write-Host "  3. Review results in results/plots/" -ForegroundColor White
Write-Host ""
Write-Host "Quick test:" -ForegroundColor Yellow
Write-Host "  python run_pipeline.py --use-mock-data" -ForegroundColor Cyan
Write-Host ""
Write-Host "For help:" -ForegroundColor Yellow
Write-Host "  python run_pipeline.py --help" -ForegroundColor White
Write-Host ""
