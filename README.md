# AP1000 Digital Twin for LOCAC Detection

A research prototype implementing a digital twin system for detecting Cold-Leg Loss of Coolant Accidents (LOCAC) in AP1000 nuclear reactors using DeepONet neural operators and machine learning.

## 🎯 Project Overview

This system combines:
- **CFD Simulation**: ANSYS Fluent simulations of AP1000 cold-leg pipe segments
- **DeepONet**: Neural operator for fast prediction of flow fields
- **LOCAC Detection**: Machine learning model for accident detection
- **Real-time Inference**: 1000x+ faster than traditional CFD

## 📁 Project Structure

```
ap1000_digital_twin/
│
├── data/                          # Data storage
│   ├── fluent_raw/               # Raw CFD outputs (CSV files)
│   ├── fluent_processed/         # Processed simulation parameters
│   ├── deeponet_dataset/         # Training data for DeepONet
│   └── nppad/                    # NPPAD nuclear plant dataset
│
├── fluent/                        # ANSYS Fluent automation
│   ├── geometry/                 # Geometry files
│   ├── mesh/                     # Mesh files
│   ├── journals/                 # Journal scripts for Fluent
│   └── automation/               # Python automation scripts
│
├── src/                          # Source code
│   ├── preprocessing/            # Data preprocessing
│   ├── deeponet/                 # DeepONet model
│   ├── feature_translation/      # CFD to system-level features
│   ├── accident_model/           # LOCAC detection model
│   └── inference/                # Inference pipeline
│
├── notebooks/                     # Jupyter notebooks (analysis)
│
├── results/                       # Outputs
│   ├── models/                   # Trained models
│   ├── plots/                    # Visualizations
│   ├── metrics/                  # Performance metrics
│   └── predictions/              # Prediction results
│
├── configs/                       # Configuration files
│   └── config.yaml               # Main configuration
│
├── scripts/                       # Execution scripts
│   ├── generate_dataset.py       # Generate CFD data
│   ├── generate_mock_data.py     # Generate synthetic data
│   ├── train_deeponet.py         # Train DeepONet
│   ├── train_locac_model.py      # Train LOCAC detector
│   └── run_inference.py          # Run inference
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (RTX 4060 or better recommended)
- ANSYS Fluent (for CFD generation) - optional, can use mock data

### Installation

1. **Clone the repository**
```bash
cd f:\Study\Semester 4\Minor\minorProj
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install PyTorch with CUDA** (for RTX 4060)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Usage

### Option A: Using Mock Data (No Fluent Required)

Perfect for testing the pipeline without ANSYS Fluent:

```bash
# Step 1: Generate mock CFD data
python scripts/generate_mock_data.py

# Step 2: Preprocess data for DeepONet
python src/preprocessing/prepare_deeponet_data.py

# Step 3: Train DeepONet
python scripts/train_deeponet.py

# Step 4: Visualize predictions
python src/deeponet/visualize.py

# Step 5: Train LOCAC detection model
python scripts/train_locac_model.py

# Step 6: Run inference
python scripts/run_inference.py --mode single
python scripts/run_inference.py --mode time_series
```

### Option B: Using ANSYS Fluent

For actual CFD simulations:

```bash
# Step 1: Generate CFD simulations (requires Fluent)
python scripts/generate_dataset.py --run-fluent

# (This will take several hours for 2000 simulations)

# Continue with steps 2-6 as above
```

## 🏗️ Selecting Model Architectures

The pipeline supports four neural operator architectures across three tiers.
Use the `--model-version` flag on any script, or set `model_version` in
`configs/model_config.yaml`.

| Flag value | Tier | Description |
|---|---|---|
| `deeponet_fourier` | Tier 1 | Optimized DeepONet (Fourier + Sobolev + Divergence) — **default** |
| `transolver` | Tier 2 | Transformer Neural Operator (Transolver++) |
| `clifford` | Tier 2 | Clifford Neural Operator (Geometric Algebra) |
| `deeponet` | Legacy | Baseline DeepONet (no Fourier features) |

**Via the full pipeline:**
```bash
python run_pipeline.py --model-version deeponet_fourier   # Tier 1 (default)
python run_pipeline.py --model-version transolver          # Tier 2 Transformer
python run_pipeline.py --model-version clifford            # Tier 2 Clifford
```

**Via individual scripts:**
```bash
python scripts/train_deeponet.py  --model-version deeponet_fourier
python scripts/train_operator.py  --model-version transolver
python scripts/train_operator.py  --model-version clifford
python scripts/run_inference.py   --model-version transolver
```

**Via config file** (`configs/model_config.yaml`):
```yaml
model_version: transolver   # applied when --model-version is not passed
```

> The CLI flag always takes precedence over the config file value.

## � Complete CLI Flags Reference

### `run_pipeline.py` — Full Pipeline

| Flag | Type | Default | Description |
|---|---|---|---|
| `--use-mock-data` | flag | `False` | Use synthetic mock data instead of Fluent CFD simulations |
| `--skip-training` | flag | `False` | Skip training and use existing saved models |
| `--model-version` | string | `None` | Select neural operator architecture. Choices: `deeponet`, `deeponet_fourier`, `transolver`, `clifford`. Overrides `configs/model_config.yaml` |

```bash
python run_pipeline.py --use-mock-data --model-version deeponet_fourier
python run_pipeline.py --skip-training --model-version transolver
```

### `scripts/train_deeponet.py` — Train DeepONet Models

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-version` | string | `None` | Architecture to train. Choices: `deeponet`, `deeponet_fourier` |
| `--operator` | string | `None` | Legacy alias for `--model-version`. Choices: `deeponet`, `deeponet_fourier` |
| `--epochs` | int | config value | Number of training epochs |
| `--lr` | float | config value | Learning rate |
| `--sobolev-weight` | float | `0.1` | Weight for Sobolev (gradient) loss term |
| `--divergence-weight` | float | `0.01` | Weight for divergence-free loss term |
| `--no-sobolev` | flag | `False` | Disable Sobolev loss entirely |
| `--no-divergence` | flag | `False` | Disable divergence loss entirely |
| `--benchmark` | flag | `False` | Run a speed/accuracy benchmark after training |

```bash
python scripts/train_deeponet.py --model-version deeponet_fourier --epochs 300 --lr 5e-4
python scripts/train_deeponet.py --model-version deeponet_fourier --no-sobolev --no-divergence
python scripts/train_deeponet.py --model-version deeponet --benchmark
```

### `scripts/train_operator.py` — Train Tier-2 Operators (Transolver / Clifford)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-version` | string | `None` | Architecture to train. Choices: `transolver`, `clifford` |
| `--operator` | string | `None` | Legacy alias for `--model-version`. Choices: `transolver`, `clifford` |
| `--epochs` | int | `500` | Number of training epochs |
| `--lr` | float | `1e-3` | Learning rate |
| `--batch-size` | int | `16` | Training batch size |
| `--config` | path | `configs/config.yaml` | Path to main config file |
| `--model-config` | path | `configs/model_config.yaml` | Path to model architecture config file |

```bash
python scripts/train_operator.py --model-version transolver --epochs 300 --lr 5e-4
python scripts/train_operator.py --model-version clifford --batch-size 8
```

### `scripts/train_diffusion.py` — Train Diffusion / Uncertainty Model

| Flag | Type | Default | Description |
|---|---|---|---|
| `--operator-ckpt` | path | `results/models/best_model.pth` | Path to a pre-trained operator checkpoint |
| `--config` | path | `configs/config.yaml` | Path to main config file |
| `--model-config` | path | `configs/model_config.yaml` | Path to model architecture config file |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `1e-4` | Learning rate |
| `--batch-size` | int | `8` | Training batch size |
| `--demo` | flag | `False` | Generate sample realisations after training |
| `--n-samples` | int | `5` | Number of demo samples to generate (used with `--demo`) |

```bash
python scripts/train_diffusion.py --epochs 200 --lr 2e-4 --batch-size 4
python scripts/train_diffusion.py --demo --n-samples 10
python scripts/train_diffusion.py --operator-ckpt results/models/transolver_best.pth
```

### `scripts/run_inference.py` — Run Inference / Predictions

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model-version` | string | `None` | Select model architecture. Choices: `deeponet`, `deeponet_fourier`, `transolver`, `clifford`, `mamba`, `diffusion` |
| `--operator` | string | `deeponet_fourier` | Legacy alias for `--model-version`. Choices: `deeponet`, `deeponet_fourier`, `transolver`, `clifford` |
| `--mode` | string | `single` | Inference mode. Choices: `single` (one case), `time_series` (temporal sequence), `benchmark` (speed test) |
| `--velocity` | float | `5.0` | Inlet velocity in m/s |
| `--break-size` | float | `2.0` | Pipe break size as % of diameter |
| `--temperature` | float | `305.0` | Coolant temperature in °C |
| `--diffusion` | int | `0` | Number of turbulence realisations via diffusion model (0 = disabled) |
| `--wss` | flag | `False` | Compute wall shear stress in predictions |

```bash
python scripts/run_inference.py --model-version deeponet_fourier --mode single
python scripts/run_inference.py --model-version transolver --mode benchmark
python scripts/run_inference.py --velocity 8.0 --break-size 5.0 --temperature 310.0
python scripts/run_inference.py --mode time_series --diffusion 10 --wss
```

### `scripts/generate_dataset.py` — Generate CFD Dataset

| Flag | Type | Default | Description |
|---|---|---|---|
| `--run-fluent` | flag | `False` | Actually run ANSYS Fluent simulations (requires Fluent installed) |
| `--config` | string | `configs/config.yaml` | Path to configuration file |

```bash
python scripts/generate_dataset.py --run-fluent
python scripts/generate_dataset.py --config configs/custom_config.yaml
```

### `fluent/automation/generate_simulations.py` — Fluent Simulation Automation

| Flag | Type | Default | Description |
|---|---|---|---|
| `--run` | flag | `False` | Run Fluent simulations (requires ANSYS Fluent) |
| `--config` | string | `configs/config.yaml` | Path to configuration file |

```bash
python fluent/automation/generate_simulations.py --run
python fluent/automation/generate_simulations.py --run --config configs/custom_config.yaml
```

## �🔧 Configuration

Edit `configs/config.yaml` to customize:

- **Geometry parameters**: Pipe diameter, length, elbow radius
- **Simulation parameters**: Velocity range, break sizes, temperatures
- **DeepONet architecture**: Network dimensions, activation functions
- **Training parameters**: Batch size, learning rate, epochs
- **LOCAC detection**: Model type, thresholds

## 📈 Performance Targets

- **DeepONet Accuracy**: >90% R² score for all fields
- **LOCAC Detection Accuracy**: >90%
- **Inference Speed**: >1000x faster than CFD
- **Fields Predicted**:
  - Pressure
  - Velocity magnitude
  - Turbulence kinetic energy
  - Temperature

## 🧪 Running Tests

Test individual components:

```bash
# Test DeepONet architecture
python src/deeponet/model.py

# Test dataset loading
python src/deeponet/dataset.py

# Test feature translation
python src/feature_translation/translator.py
```

## 📊 Results

After training, results are saved in `results/`:

- **Models**: `results/models/`
  - `best_model.pth` - Best DeepONet model
  - `locac_detector.pkl` - Trained LOCAC detector

- **Visualizations**: `results/plots/`
  - Training curves
  - Field comparisons (CFD vs DeepONet)
  - Error heatmaps
  - ROC curves
  - Time series predictions

- **Metrics**: Saved in model checkpoints and console output

## 🔬 Example Output

### Single Case Inference
```
Running Digital Twin Inference
============================================================
Input Parameters:
  Velocity: 5.0 m/s
  Break size: 5.0% of diameter
  Temperature: 305.0°C

✓ DeepONet prediction: 15.23 ms

✓ Extracted features:
    average_pressure: 15420000.0
    pressure_drop: 75000.0
    mass_flow_rate: 14250.5
    ...

✓ LOCAC Detection:
    Probability: 0.8734
    Decision: LOCAC DETECTED
```

### Performance Benchmark
```
CFD simulation time (estimated): 3600.0 seconds
DeepONet inference time: 0.0152 seconds
Speedup: 237000x

✓ Achieved >1000x speedup target!
```

## 🎓 Technical Details

### DeepONet Architecture

- **Branch Network**: Processes simulation parameters
  - Input: [velocity, break_size, temperature]
  - Architecture: 3 → 256 → 512 → 512 → 256
  
- **Trunk Network**: Processes spatial coordinates
  - Input: [x, y, z]
  - Architecture: 3 → 256 → 512 → 256

- **Operator**: Dot product of branch and trunk outputs
- **Total Parameters**: ~2-3 million

### Training Details

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Mean Squared Error
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience=50 epochs
- **Mixed Precision**: Enabled for RTX 4060
- **Batch Size**: 16 (optimized for 8GB VRAM)

### LOCAC Detection

- **Model**: Gradient Boosting Classifier (XGBoost)
- **Features**: 7 system-level indicators
- **Alternative**: Neural network classifier available

## 📝 Notes

### Mock Data vs Real CFD

- **Mock data** is generated synthetically for demonstration
- **Real CFD** requires ANSYS Fluent and takes hours to run
- Mock data captures key physics but is simplified
- For research/production, use actual CFD simulations

### GPU Requirements

- **Minimum**: NVIDIA GPU with 6GB VRAM
- **Recommended**: RTX 4060 (8GB) or better
- **Training time**: ~2-4 hours on RTX 4060
- **CPU mode**: Available but much slower

### NPPAD Dataset

- Place NPPAD data in `data/nppad/`
- Format: CSV files with time-series accident data
- If not available, synthetic NPPAD data is generated

## 🤝 Contributing

This is a research prototype. Suggestions for improvements:

1. Implement geometry creation in SpaceClaim/DesignModeler
2. Add more sophisticated mesh refinement
3. Incorporate transient simulations
4. Add more physics models (heat transfer, phase change)
5. Expand to full reactor loop
6. Integrate with actual plant DCS

## 📚 References

### DeepONet
- Lu et al. (2021). "Learning nonlinear operators via DeepONet"
- Wang et al. (2021). "Learning the solution operator of parametric PDEs"

### Nuclear Safety
- AP1000 Design Control Document
- NPPAD Database Documentation
- NRC LOCA Analysis Guidelines

## ⚠️ Disclaimer

This is a research prototype for educational and research purposes only. It is NOT certified for use in actual nuclear plant operations or safety systems. All predictions must be validated against certified simulation tools and experimental data.

## 📧 Contact

For questions or collaborations:
- punyansh26

## 📄 License

Educational and research use only.

---


**Last Updated**: March 2026
