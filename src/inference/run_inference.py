"""
End-to-End Inference Pipeline
Input parameters → DeepONet → Feature extraction → LOCAC detection
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import pickle
import time

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.deeponet.model import DeepONet
from src.feature_translation.translator import FeatureTranslator


class DigitalTwinInference:
    """End-to-end digital twin inference"""
    
    def __init__(self, config_path, deeponet_path, locac_detector_path, scalers_path):
        """Initialize inference pipeline"""
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = project_root
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load DeepONet
        print("Loading DeepONet...")
        self.deeponet = DeepONet(self.config).to(self.device)
        checkpoint = torch.load(deeponet_path, map_location=self.device)
        self.deeponet.load_state_dict(checkpoint['model_state_dict'])
        self.deeponet.eval()
        print("✓ DeepONet loaded")
        
        # Load scalers
        print("Loading scalers...")
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        print("✓ Scalers loaded")
        
        # Load LOCAC detector
        print("Loading LOCAC detector...")
        with open(locac_detector_path, 'rb') as f:
            locac_data = pickle.load(f)
            self.locac_detector = locac_data['model']
            self.locac_scaler = locac_data['scaler']
        print("✓ LOCAC detector loaded")
        
        # Feature translator
        self.translator = FeatureTranslator(config_path)
        
        # Load trunk coordinates from dataset
        import h5py
        h5_path = self.project_root / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
        with h5py.File(h5_path, 'r') as f:
            self.trunk_coords_norm = torch.FloatTensor(f['train']['trunk'][:]).to(self.device)
        
        # Denormalize coordinates
        self.trunk_coords = self.scalers['trunk'].inverse_transform(
            self.trunk_coords_norm.cpu().numpy()
        )
        
        self.field_names = self.config['deeponet']['output_fields']
    
    def denormalize_field(self, data, field_name):
        """Denormalize field using scaler"""
        scaler = self.scalers['targets'][field_name]
        shape = data.shape
        data_flat = data.reshape(-1, 1)
        denorm = scaler.inverse_transform(data_flat)
        return denorm.reshape(shape)
    
    def run_inference(self, velocity, break_size, temperature, verbose=True):
        """
        Run complete inference pipeline
        
        Args:
            velocity: Flow velocity (m/s)
            break_size: Break size (% of diameter)
            temperature: Temperature (°C)
            verbose: Print details
            
        Returns:
            results: Dictionary with all results
        """
        
        if verbose:
            print("\n" + "="*60)
            print("Running Digital Twin Inference")
            print("="*60)
            print(f"Input Parameters:")
            print(f"  Velocity: {velocity} m/s")
            print(f"  Break size: {break_size}% of diameter")
            print(f"  Temperature: {temperature}°C")
        
        # Step 1: Prepare branch input
        branch_input = np.array([[velocity, break_size, temperature]])
        branch_input_norm = self.scalers['branch'].transform(branch_input)
        branch_tensor = torch.FloatTensor(branch_input_norm).to(self.device)
        
        # Step 2: DeepONet prediction
        start_time = time.time()
        with torch.no_grad():
            predictions = self.deeponet(branch_tensor, self.trunk_coords_norm)
        deeponet_time = time.time() - start_time
        
        predictions_np = predictions.cpu().numpy()[0]  # [n_fields, n_nodes]
        
        if verbose:
            print(f"\n✓ DeepONet prediction: {deeponet_time*1000:.2f} ms")
        
        # Step 3: Denormalize predictions
        fields_dict = {}
        for i, field_name in enumerate(self.field_names):
            fields_dict[field_name] = self.denormalize_field(
                predictions_np[i], field_name
            )
        
        # Step 4: Extract features
        features = self.translator.extract_features(fields_dict, self.trunk_coords)
        
        if verbose:
            print(f"\n✓ Extracted features:")
            for key, value in features.items():
                print(f"    {key}: {value:.4f}")
        
        # Step 5: LOCAC detection
        # Prepare feature vector for classifier
        feature_vector = np.array([
            features['average_pressure'],
            features['mass_flow_rate'],
            features['avg_temperature'],
            features['pressure_drop'],
            features['max_turbulence'],
            features['temperature_difference'],
            features['velocity_std']
        ]).reshape(1, -1)
        
        feature_vector_scaled = self.locac_scaler.transform(feature_vector)
        locac_probability = self.locac_detector.predict_proba(feature_vector_scaled)[0, 1]
        locac_decision = locac_probability > 0.5
        
        if verbose:
            print(f"\n✓ LOCAC Detection:")
            print(f"    Probability: {locac_probability:.4f}")
            print(f"    Decision: {'LOCAC DETECTED' if locac_decision else 'NORMAL'}")
        
        # Compile results
        results = {
            'input_params': {
                'velocity': velocity,
                'break_size': break_size,
                'temperature': temperature
            },
            'fields': fields_dict,
            'features': features,
            'locac_probability': locac_probability,
            'locac_detected': locac_decision,
            'inference_time_ms': deeponet_time * 1000,
            'coordinates': self.trunk_coords
        }
        
        return results
    
    def run_time_series_simulation(self, param_sequence, duration=60):
        """
        Run simulation over time series
        
        Args:
            param_sequence: List of (velocity, break_size, temperature) tuples
            duration: Simulation duration in seconds
            
        Returns:
            time_series_results: List of results over time
        """
        
        print("\n" + "="*60)
        print(f"Time Series Simulation ({duration}s)")
        print("="*60)
        
        n_steps = len(param_sequence)
        dt = duration / n_steps
        
        time_series_results = []
        
        for i, (velocity, break_size, temperature) in enumerate(param_sequence):
            t = i * dt
            
            print(f"\nTime step {i+1}/{n_steps} (t={t:.1f}s)")
            
            results = self.run_inference(velocity, break_size, temperature, verbose=False)
            results['time'] = t
            
            time_series_results.append(results)
            
            # Print summary
            print(f"  LOCAC probability: {results['locac_probability']:.4f}")
            if results['locac_detected']:
                print(f"  ⚠ LOCAC DETECTED!")
        
        return time_series_results
    
    def compare_with_cfd(self, cfd_time_estimate=3600):
        """Compare inference time with CFD simulation"""
        
        # Run single inference
        results = self.run_inference(5.0, 5.0, 305.0, verbose=False)
        inference_time = results['inference_time_ms'] / 1000  # seconds
        
        speedup = cfd_time_estimate / inference_time
        
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)
        print(f"CFD simulation time (estimated): {cfd_time_estimate:.1f} seconds")
        print(f"DeepONet inference time: {inference_time:.4f} seconds")
        print(f"Speedup: {speedup:.0f}x")
        
        if speedup > 1000:
            print(f"\n✓ Achieved >1000x speedup target!")
        
        return speedup


def test_single_case():
    """Test single case inference"""
    
    config_path = project_root / "configs" / "config.yaml"
    deeponet_path = project_root / "results" / "models" / "best_model.pth"
    locac_path = project_root / "results" / "models" / "locac_detector.pkl"
    scalers_path = project_root / "data" / "deeponet_dataset" / "scalers.pkl"
    
    # Check files exist
    for path in [deeponet_path, locac_path, scalers_path]:
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            return
    
    # Create pipeline
    pipeline = DigitalTwinInference(config_path, deeponet_path, locac_path, scalers_path)
    
    # Test cases
    test_cases = [
        {"name": "Normal Operation", "velocity": 5.0, "break_size": 0.0, "temperature": 305.0},
        {"name": "Small Break", "velocity": 4.8, "break_size": 2.0, "temperature": 300.0},
        {"name": "Medium Break", "velocity": 4.5, "break_size": 5.0, "temperature": 295.0},
        {"name": "Large Break", "velocity": 4.0, "break_size": 10.0, "temperature": 290.0},
    ]
    
    results_list = []
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test Case: {case['name']}")
        print('='*60)
        
        results = pipeline.run_inference(
            case['velocity'],
            case['break_size'],
            case['temperature']
        )
        
        results_list.append(results)
    
    # Performance benchmark
    pipeline.compare_with_cfd()
    
    return results_list


def test_time_series():
    """Test time series simulation"""
    
    config_path = project_root / "configs" / "config.yaml"
    deeponet_path = project_root / "results" / "models" / "best_model.pth"
    locac_path = project_root / "results" / "models" / "locac_detector.pkl"
    scalers_path = project_root / "data" / "deeponet_dataset" / "scalers.pkl"
    
    # Check files exist
    for path in [deeponet_path, locac_path, scalers_path]:
        if not path.exists():
            print(f"ERROR: Required file not found: {path}")
            return
    
    # Create pipeline
    pipeline = DigitalTwinInference(config_path, deeponet_path, locac_path, scalers_path)
    
    # Simulate LOCAC event over 60 seconds
    n_steps = 20
    
    # Start normal, then break occurs at t=20s
    param_sequence = []
    
    for i in range(n_steps):
        t = i * 3  # 3 second intervals
        
        if t < 20:
            # Normal operation
            velocity = 5.0 + np.random.normal(0, 0.05)
            break_size = 0.0
            temperature = 305.0 + np.random.normal(0, 1)
        else:
            # Break develops
            severity = (t - 20) / 40  # 0 to 1 over 40 seconds
            velocity = 5.0 - severity * 1.0 + np.random.normal(0, 0.1)
            break_size = severity * 10.0
            temperature = 305.0 - severity * 15 + np.random.normal(0, 2)
        
        param_sequence.append((velocity, break_size, temperature))
    
    # Run time series
    results = pipeline.run_time_series_simulation(param_sequence, duration=60)
    
    # Plot results
    plot_time_series(results)
    
    return results


def plot_time_series(results):
    """Plot time series results"""
    import matplotlib.pyplot as plt
    
    times = [r['time'] for r in results]
    probabilities = [r['locac_probability'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(times, probabilities, 'b-', linewidth=2, label='LOCAC Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Detection Threshold')
    plt.fill_between(times, 0, 1, where=[p > 0.5 for p in probabilities], 
                     alpha=0.3, color='red', label='LOCAC Detected')
    
    plt.xlabel('Time (s)')
    plt.ylabel('LOCAC Probability')
    plt.title('Digital Twin LOCAC Detection - Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plot_dir = project_root / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / 'locac_time_series.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved time series plot to {plot_path}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run digital twin inference')
    parser.add_argument('--mode', choices=['single', 'time_series'], 
                       default='single', help='Inference mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_single_case()
    else:
        test_time_series()


if __name__ == "__main__":
    main()
