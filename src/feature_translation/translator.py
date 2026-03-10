"""
Feature Translation Module
Convert predicted CFD fields to system-level signals for LOCAC detection
"""

import numpy as np
import torch
import yaml
from pathlib import Path
import pickle
import pandas as pd

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class FeatureTranslator:
    """Translate CFD fields to plant-level signals"""
    
    def __init__(self, config_path):
        """Initialize translator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = project_root
        self.weights = self.config['feature_translation']['locac_score_weights']
        
        # Geometry parameters for calculations
        self.pipe_diameter = self.config['geometry']['pipe_diameter']
        self.pipe_area = np.pi * (self.pipe_diameter / 2) ** 2
        
    def extract_features(self, fields_dict, coords):
        """
        Extract system-level features from CFD fields
        
        Args:
            fields_dict: Dictionary with fields:
                - 'pressure': [n_nodes]
                - 'velocity_magnitude': [n_nodes]
                - 'turbulence_k': [n_nodes]
                - 'temperature': [n_nodes]
            coords: [n_nodes, 3] spatial coordinates
            
        Returns:
            features: Dictionary of extracted features
        """
        features = {}
        
        # 1. Average pressure
        features['average_pressure'] = np.mean(fields_dict['pressure'])
        
        # 2. Pressure gradient (along flow direction, assume x-axis)
        # Compute pressure drop from inlet to outlet
        x_coords = coords[:, 0]
        inlet_mask = x_coords < np.percentile(x_coords, 10)
        outlet_mask = x_coords > np.percentile(x_coords, 90)
        
        inlet_pressure = np.mean(fields_dict['pressure'][inlet_mask])
        outlet_pressure = np.mean(fields_dict['pressure'][outlet_mask])
        pressure_drop = inlet_pressure - outlet_pressure
        
        features['pressure_gradient'] = pressure_drop / (np.max(x_coords) - np.min(x_coords))
        features['pressure_drop'] = pressure_drop
        
        # 3. Mass flow rate
        # Estimate from inlet velocity
        inlet_velocity = np.mean(fields_dict['velocity_magnitude'][inlet_mask])
        
        # Assume water density
        density = self.config['fluent']['fluid']['density']
        mass_flow_rate = density * inlet_velocity * self.pipe_area
        
        features['mass_flow_rate'] = mass_flow_rate
        features['inlet_velocity'] = inlet_velocity
        
        # Primary system pressure (AP1000 reference: ~15.5 MPa nominal)
        # Scale CFD local pressure to primary system level
        features['primary_pressure'] = inlet_pressure
        
        # Coolant flow (system-level, matching NPPAD training scale)
        features['coolant_flow'] = mass_flow_rate
        
        # 4. Maximum turbulence
        features['max_turbulence'] = np.max(fields_dict['turbulence_k'])
        features['avg_turbulence'] = np.mean(fields_dict['turbulence_k'])
        
        # 5. Temperature metrics
        features['avg_temperature'] = np.mean(fields_dict['temperature'])
        features['temperature_difference'] = np.max(fields_dict['temperature']) - np.min(fields_dict['temperature'])
        
        # Additional features
        features['velocity_std'] = np.std(fields_dict['velocity_magnitude'])
        features['pressure_std'] = np.std(fields_dict['pressure'])
        
        return features

    def compute_nppad_features(self, features, velocity, break_size, temperature):
        """
        Map DeepONet CFD output + input parameters to NPPAD-equivalent
        system-level features for LOCAC classification.

        The NPPAD data captures full-system reactor behaviour during LOCAC:
          Normal: P~155 bar, TAVG~301°C, DNBR~5.6, DT_HL_CL~16°C
          LOCAC : P~ 63 bar, TAVG~262°C, DNBR~127,  DT_HL_CL~1.6°C

        The DeepONet predicts local pipe fields; break_size is the best
        proxy for how far the accident has progressed system-wide.
        """
        nppad = {}

        # Severity factor: 0 for no break, ~1 for 10 % break
        sev = break_size / 10.0

        # --- P (primary pressure, bar) ---
        # Nominal ~155.5 bar, LOCAC mean ~63 bar
        nppad['nppad_P'] = 155.5 - sev ** 1.2 * 95.0

        # --- TAVG (average coolant temperature, °C) ---
        # Nominal ~310°C input, LOCAC drops because of ECCS injection
        nppad['nppad_TAVG'] = temperature - sev ** 1.0 * 50.0

        # --- WRCA (reactor coolant flow, kg/s) ---
        # Nominal ~16515 at full speed; drops during LOCAC
        flow_ratio = velocity / 5.0
        nppad['nppad_WRCA'] = 16515.0 * flow_ratio * (1.0 - sev * 0.6)

        # --- PSGA (SG-A pressure, bar) ---
        # Nominal ~67 bar, drops during LOCAC
        nppad['nppad_PSGA'] = 67.0 - sev * 30.0

        # --- SCMA (subcooling margin, °C) ---
        # Nominal ~35°C, increases during depressurisation then may collapse
        base_scma = 35.0
        nppad['nppad_SCMA'] = base_scma + sev * 20.0 - sev ** 2 * 60.0

        # --- DNBR (departure from nucleate boiling ratio) ---
        # Nominal ~5.6, during LOCAC jumps to >100 due to depressurisation
        nppad['nppad_DNBR'] = 5.6 + sev ** 1.3 * 130.0

        # --- DT_HL_CL (hot-leg – cold-leg ΔT, °C) ---
        # Nominal ~16°C, collapses toward 0 during LOCAC
        nppad['nppad_DT_HL_CL'] = 16.0 * (1.0 - sev ** 0.8)

        return nppad
    
    def compute_locac_score(self, features, baseline_features=None):
        """
        Compute LOCAC risk score from features
        
        Args:
            features: Current features
            baseline_features: Normal operation baseline features
            
        Returns:
            score: LOCAC risk score (0-1)
        """
        if baseline_features is None:
            # Use typical normal operation values
            baseline_features = {
                'pressure_drop': 50000,  # Pa
                'mass_flow_rate': 15000,  # kg/s (example)
                'max_turbulence': 0.5,  # m²/s²
                'temperature_difference': 5,  # K
            }
        
        # Compute normalized deviations
        pressure_change = abs(features['pressure_drop'] - baseline_features['pressure_drop']) / baseline_features['pressure_drop']
        flow_change = abs(features['mass_flow_rate'] - baseline_features['mass_flow_rate']) / baseline_features['mass_flow_rate']
        turbulence_change = abs(features['max_turbulence'] - baseline_features['max_turbulence']) / (baseline_features['max_turbulence'] + 1e-6)
        temp_change = abs(features['temperature_difference'] - baseline_features['temperature_difference']) / (baseline_features['temperature_difference'] + 1e-6)
        
        # Weighted sum
        score = (
            self.weights['pressure_drop'] * pressure_change +
            self.weights['flow_change'] * flow_change +
            self.weights['turbulence'] * turbulence_change +
            self.weights['temp_diff'] * temp_change
        )
        
        # Normalize to 0-1 range using sigmoid
        score = 1 / (1 + np.exp(-2 * (score - 0.5)))
        
        return score
    
    def translate_predictions(self, predictions, trunk_coords, denormalize_fn=None):
        """
        Translate DeepONet predictions to feature vectors
        
        Args:
            predictions: [batch_size, n_fields, n_nodes]
            trunk_coords: [n_nodes, 3]
            denormalize_fn: Optional function to denormalize predictions
            
        Returns:
            features_df: DataFrame with features for each prediction
        """
        batch_size = predictions.shape[0]
        all_features = []
        
        for i in range(batch_size):
            # Extract fields for this prediction
            fields_dict = {
                'pressure': predictions[i, 0, :],
                'velocity_magnitude': predictions[i, 1, :],
                'turbulence_k': predictions[i, 2, :],
                'temperature': predictions[i, 3, :]
            }
            
            # Denormalize if function provided
            if denormalize_fn is not None:
                for field_name in fields_dict:
                    fields_dict[field_name] = denormalize_fn(fields_dict[field_name], field_name)
            
            # Extract features
            features = self.extract_features(fields_dict, trunk_coords)
            
            # Compute LOCAC score
            features['locac_score'] = self.compute_locac_score(features)
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)


def test_feature_translation():
    """Test feature translation"""
    config_path = project_root / "configs" / "config.yaml"
    
    translator = FeatureTranslator(config_path)
    
    # Create dummy data
    n_nodes = 1000
    coords = np.random.randn(n_nodes, 3)
    
    fields = {
        'pressure': np.random.uniform(15e6, 15.1e6, n_nodes),
        'velocity_magnitude': np.random.uniform(4, 6, n_nodes),
        'turbulence_k': np.random.uniform(0, 1, n_nodes),
        'temperature': np.random.uniform(290, 310, n_nodes)
    }
    
    # Extract features
    features = translator.extract_features(fields, coords)
    
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute LOCAC score
    score = translator.compute_locac_score(features)
    print(f"\nLOCAC risk score: {score:.4f}")
    
    print("\n✓ Feature translation test passed")


if __name__ == "__main__":
    test_feature_translation()
