# Quick Start Guide

## 🚀 Get Running in 10 Minutes

This guide will get you up and running with the AP1000 Digital Twin using mock data (no ANSYS Fluent required).

### Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd "f:\Study\Semester 4\Minor\minorProj"

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Run the Full Pipeline (5-8 minutes)

**Single command to run everything:**

```bash
python run_pipeline.py --use-mock-data
```

This will:
1. ✅ Generate 2000 mock CFD simulations (~30 seconds)
2. ✅ Preprocess data for DeepONet (~1 minute)
3. ✅ Train DeepONet model (~3-5 minutes on GPU)
4. ✅ Generate visualizations (~30 seconds)
5. ✅ Train LOCAC detector (~20 seconds)
6. ✅ Run inference tests (~10 seconds)

### Step 3: View Results

**Check the results:**

```bash
# View generated plots
start results/plots/

# Check models
dir results\models\
```

### Step 4: Run Custom Inference

**Test with your own parameters:**

```bash
# Single case
python scripts/run_inference.py --mode single

# Time series simulation
python scripts/run_inference.py --mode time_series
```

## 📊 Expected Output

### Training Progress
```
Epoch 50/2000
Train Loss: 0.000234
Val Loss: 0.000189

Field Metrics:
  pressure:
    R²: 0.9567
    Rel L2: 0.0312
    MAE: 0.000145
  velocity_magnitude:
    R²: 0.9623
    Rel L2: 0.0287
    MAE: 0.000132
  ...

✓ Saved best model
```

### Inference Output
```
Input Parameters:
  Velocity: 5.0 m/s
  Break size: 5.0% of diameter
  Temperature: 305.0°C

✓ DeepONet prediction: 15.23 ms

✓ LOCAC Detection:
    Probability: 0.8734
    Decision: LOCAC DETECTED

Speedup: 237000x ✓
```

## 🎯 What You Get

After running the pipeline, you'll have:

1. **Trained Models**
   - `results/models/best_model.pth` - DeepONet
   - `results/models/locac_detector.pkl` - LOCAC detector

2. **Visualizations**
   - Field comparison plots (CFD vs DeepONet)
   - Error heatmaps
   - Training curves
   - ROC curves
   - Time series predictions

3. **Performance Metrics**
   - DeepONet accuracy >90%
   - LOCAC detection accuracy >90%
   - 1000x+ speedup vs CFD

## 🔧 Customization

### Change Simulation Parameters

Edit `configs/config.yaml`:

```yaml
parameter_sweep:
  velocity:
    min: 4.0
    max: 6.0
    samples: 20  # Increase for more data
  break_size:
    min: 0.0
    max: 10.0
    samples: 10
```

### Adjust Training

```yaml
training:
  batch_size: 16  # Decrease if GPU memory limited
  epochs: 2000    # Increase for better accuracy
  learning_rate: 1e-3
```

### Change Network Architecture

```yaml
deeponet:
  branch_net:
    hidden_dims: [256, 512, 512]  # Modify layers
  trunk_net:
    hidden_dims: [256, 512, 256]
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

Reduce batch size in `configs/config.yaml`:

```yaml
training:
  batch_size: 8  # Down from 16
```

### Training Too Slow

Enable mixed precision (should be default):

```yaml
training:
  mixed_precision: true
```

## 📚 Next Steps

1. **Analyze Results**: Open plots in `results/plots/`

2. **Try Different Scenarios**:
   ```bash
   # Edit parameters in the script and run
   python scripts/run_inference.py
   ```

3. **Use Real CFD Data**: 
   - Install ANSYS Fluent
   - Run: `python scripts/generate_dataset.py --run-fluent`
   - Wait several hours for 2000 simulations

4. **Add Your Own NPPAD Data**:
   - Place CSV files in `data/nppad/operation_csv_data/`
   - Rerun: `python scripts/train_locac_model.py`

## ⏱️ Time Breakdown

On a system with RTX 4060:

| Step | Time | GPU Used |
|------|------|----------|
| Mock Data Generation | 30s | No |
| Data Preprocessing | 1min | No |
| DeepONet Training | 3-5min | Yes |
| Visualization | 30s | No |
| LOCAC Training | 20s | No |
| Inference | <1s | Yes |
| **Total** | **~6-8 min** | |

## 🎓 Understanding the Output

### DeepONet Metrics

- **R² Score**: How well predictions match CFD (0-1, higher better)
- **Rel L2 Error**: Relative error norm (<0.1 is good)
- **MAE**: Mean absolute error (normalized)

### LOCAC Detection

- **Accuracy**: Overall correct predictions
- **Precision**: Of detected LOCACs, how many are real
- **Recall**: Of real LOCACs, how many detected
- **F1**: Harmonic mean of precision/recall
- **ROC-AUC**: Area under ROC curve (>0.9 is excellent)

## 💡 Tips

1. **First Run**: Use mock data to verify everything works
2. **GPU Acceleration**: Massive speedup for training and inference
3. **Checkpoints**: Training saves best model automatically
4. **Visualization**: Always check plots to verify predictions
5. **Parameters**: Start with defaults, tune if needed

---

**Ready? Run this:**

```bash
python run_pipeline.py --use-mock-data
```

**Questions?** Check `README.md` for detailed documentation.
