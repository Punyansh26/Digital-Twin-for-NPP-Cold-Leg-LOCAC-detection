"""
Data preprocessing for DeepONet training
Converts CFD results to DeepONet format
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm
import pickle


class DeepONetDataPreprocessor:
    """Preprocess CFD data for DeepONet training"""
    
    def __init__(self, config_path):
        """Initialize preprocessor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent.parent
        self.fluent_raw_dir = self.project_root / self.config['data_paths']['fluent_raw']
        self.fluent_processed_dir = self.project_root / self.config['data_paths']['fluent_processed']
        self.output_dir = self.project_root / self.config['data_paths']['deeponet_dataset']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scalers
        self.branch_scaler = MinMaxScaler()
        self.trunk_scaler = MinMaxScaler()
        self.target_scalers = {}
        
    def load_single_simulation(self, csv_path):
        """Load single CFD simulation CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            # Expected columns (adjust based on Fluent export)
            required_cols = ['x-coordinate', 'y-coordinate', 'z-coordinate',
                           'pressure', 'velocity-magnitude', 
                           'turb-kinetic-energy', 'temperature']
            
            # Check if columns exist (Fluent column names may vary)
            # Rename if necessary
            column_mapping = {
                'x-coordinate': 'x',
                'y-coordinate': 'y',
                'z-coordinate': 'z',
                'pressure': 'pressure',
                'velocity-magnitude': 'velocity_magnitude',
                'turb-kinetic-energy': 'turbulence_k',
                'temperature': 'temperature'
            }
            
            df = df.rename(columns=column_mapping)
            
            return df
            
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None
    
    def load_all_simulations(self):
        """Load all CFD simulations"""
        
        # Load parameter file
        params_file = self.fluent_processed_dir / "simulation_parameters.csv"
        params_df = pd.read_csv(params_file)
        
        print(f"Loading {len(params_df)} simulations...")
        
        simulations = []
        valid_params = []
        
        for _, row in tqdm(params_df.iterrows(), total=len(params_df)):
            case_id = row['case_id']
            csv_path = self.fluent_raw_dir / f"{case_id}.csv"
            
            if csv_path.exists():
                df = self.load_single_simulation(csv_path)
                if df is not None:
                    simulations.append(df)
                    valid_params.append(row)
        
        print(f"Successfully loaded {len(simulations)} simulations")
        
        return simulations, pd.DataFrame(valid_params)
    
    def prepare_deeponet_format(self, simulations, params_df):
        """Convert to DeepONet format"""
        
        print("Preparing DeepONet format...")
        
        # Branch inputs: [velocity, break_size, temperature]
        branch_inputs = params_df[['velocity', 'break_size', 'temperature']].values
        
        # Trunk inputs: [x, y, z] - use first simulation as reference
        trunk_inputs = simulations[0][['x', 'y', 'z']].values
        
        # Verify all simulations have same mesh
        n_nodes = len(trunk_inputs)
        for sim in simulations:
            if len(sim) != n_nodes:
                raise ValueError(f"Mesh mismatch: expected {n_nodes} nodes")
        
        # Targets: [pressure, velocity_magnitude, turbulence_k, temperature]
        target_fields = ['pressure', 'velocity_magnitude', 'turbulence_k', 'temperature']
        targets = np.zeros((len(simulations), len(target_fields), n_nodes))
        
        for i, sim in enumerate(simulations):
            for j, field in enumerate(target_fields):
                targets[i, j, :] = sim[field].values
        
        print(f"Branch inputs shape: {branch_inputs.shape}")
        print(f"Trunk inputs shape: {trunk_inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        
        return branch_inputs, trunk_inputs, targets
    
    def normalize_data(self, branch_inputs, trunk_inputs, targets):
        """Normalize all data"""
        
        print("Normalizing data...")
        
        # Normalize branch inputs
        branch_normalized = self.branch_scaler.fit_transform(branch_inputs)
        
        # Normalize trunk inputs
        trunk_normalized = self.trunk_scaler.fit_transform(trunk_inputs)
        
        # Normalize each target field separately
        target_fields = self.config['deeponet']['output_fields']
        targets_normalized = np.zeros_like(targets)
        
        for i, field in enumerate(target_fields):
            scaler = MinMaxScaler()
            # Reshape for scaling: (n_simulations * n_nodes,)
            field_data = targets[:, i, :].reshape(-1, 1)
            field_normalized = scaler.fit_transform(field_data)
            targets_normalized[:, i, :] = field_normalized.reshape(targets[:, i, :].shape)
            self.target_scalers[field] = scaler
        
        return branch_normalized, trunk_normalized, targets_normalized
    
    def split_dataset(self, branch_inputs, trunk_inputs, targets):
        """Split into train/val/test sets"""
        
        val_split = self.config['training']['validation_split']
        test_split = self.config['training']['test_split']
        
        # First split: train+val vs test
        indices = np.arange(len(branch_inputs))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_split, random_state=42
        )
        
        # Second split: train vs val
        val_size = val_split / (1 - test_split)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size, random_state=42
        )
        
        # Create splits
        splits = {
            'train': {
                'branch': branch_inputs[train_idx],
                'trunk': trunk_inputs,  # Same for all
                'targets': targets[train_idx]
            },
            'val': {
                'branch': branch_inputs[val_idx],
                'trunk': trunk_inputs,
                'targets': targets[val_idx]
            },
            'test': {
                'branch': branch_inputs[test_idx],
                'trunk': trunk_inputs,
                'targets': targets[test_idx]
            }
        }
        
        print(f"Train: {len(train_idx)} samples")
        print(f"Val: {len(val_idx)} samples")
        print(f"Test: {len(test_idx)} samples")
        
        return splits
    
    def save_dataset(self, splits):
        """Save dataset and scalers"""
        
        print("Saving dataset...")
        
        # Save to HDF5
        h5_path = self.output_dir / "deeponet_dataset.h5"
        with h5py.File(h5_path, 'w') as f:
            for split_name, split_data in splits.items():
                group = f.create_group(split_name)
                group.create_dataset('branch', data=split_data['branch'])
                group.create_dataset('trunk', data=split_data['trunk'])
                group.create_dataset('targets', data=split_data['targets'])
        
        print(f"✓ Saved dataset to {h5_path}")
        
        # Save scalers
        scalers = {
            'branch': self.branch_scaler,
            'trunk': self.trunk_scaler,
            'targets': self.target_scalers
        }
        
        scaler_path = self.output_dir / "scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        print(f"✓ Saved scalers to {scaler_path}")
    
    def process(self):
        """Main processing pipeline"""
        
        print("="*60)
        print("DeepONet Data Preprocessing")
        print("="*60)
        
        # Load simulations
        simulations, params_df = self.load_all_simulations()
        
        if len(simulations) == 0:
            print("\nWARNING: No simulation data found!")
            print("You need to run Fluent simulations first:")
            print("  python fluent/automation/generate_simulations.py --run")
            return
        
        # Convert to DeepONet format
        branch_inputs, trunk_inputs, targets = self.prepare_deeponet_format(
            simulations, params_df
        )
        
        # Normalize
        branch_norm, trunk_norm, targets_norm = self.normalize_data(
            branch_inputs, trunk_inputs, targets
        )
        
        # Split dataset
        splits = self.split_dataset(branch_norm, trunk_norm, targets_norm)
        
        # Save
        self.save_dataset(splits)
        
        print("\n" + "="*60)
        print("Preprocessing complete!")
        print("="*60)


def main():
    """Main execution"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "config.yaml"
    
    preprocessor = DeepONetDataPreprocessor(config_path)
    preprocessor.process()


if __name__ == "__main__":
    main()
