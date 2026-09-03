# INSTALLATION TROUBLESHOOTING

## Issue: Build Errors on Windows with Python 3.13

If you encounter errors like:
- `error: Microsoft Visual C++ 14.0 or greater is required`
- `Unable to load dependency HDF5, make sure HDF5 is installed properly`
- `Failed building wheel for h5py/kiwisolver/greenlet`

### Solution 1: Use Python 3.11 or 3.12 (RECOMMENDED)

Python 3.13 (especially free-threaded) has limited pre-built wheel support.

```bash
# Uninstall Python 3.13
# Install Python 3.11 or 3.12 from python.org
# Then:
pip install -r requirements.txt
```

### Solution 2: Install Pre-built Wheels

```bash
# For h5py on Windows with Python 3.13
pip install --only-binary=h5py h5py

# Or install from conda-forge
conda install h5py -c conda-forge

# For all problematic packages
pip install --only-binary=:all: -r requirements.txt
```

### Solution 3: Install Visual C++ Build Tools

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart terminal
4. Run: `pip install -r requirements.txt`

### Solution 4: Use Simplified Requirements (Quick Fix)

```bash
# Install only essential packages without compilation
pip install torch torchvision numpy pandas matplotlib seaborn PyYAML
pip install scikit-learn xgboost tqdm

# Skip h5py for now - we'll add alternative data format
```

### Solution 5: Use Conda (EASIEST)

```bash
# Install Miniconda from conda.io
# Then:
conda create -n digital_twin python=3.11
conda activate digital_twin
conda install pytorch torchvision -c pytorch -c nvidia
conda install numpy pandas matplotlib seaborn scikit-learn h5py -c conda-forge
pip install xgboost lightgbm optuna tqdm plotly PyYAML
```

## Current Issue Analysis

Your system is running **Python 3.13t** (free-threaded build). This version:
- ❌ Limited pre-built wheels
- ❌ Requires compilation for many packages
- ❌ Needs Visual C++ Build Tools
- ❌ Needs HDF5 C library for h5py

**Recommendation**: Switch to **Python 3.11** for best compatibility.

## Quick Test

After fixing, test your installation:

```bash
python -c "import torch; import numpy; import pandas; import yaml; print('✓ Core imports work')"
```

If h5py still fails, you can modify the code to use pickle instead of HDF5.
