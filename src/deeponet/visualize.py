"""
Visualization Module
Generate contour plots and error heatmaps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import h5py
import yaml
from pathlib import Path
import pickle

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.deeponet.model import DeepONet


class DeepONetVisualizer:
    """Visualize DeepONet predictions"""
    
    def __init__(self, config_path, model_path, scalers_path):
        """Initialize visualizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = project_root
        self.output_dir = self.project_root / self.config['output_paths']['plots']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load scalers
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepONet(self.config).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.field_names = self.config['deeponet']['output_fields']
        
    def denormalize_field(self, data, field_name):
        """Denormalize a field using saved scaler"""
        scaler = self.scalers['targets'][field_name]
        shape = data.shape
        data_flat = data.reshape(-1, 1)
        denorm = scaler.inverse_transform(data_flat)
        return denorm.reshape(shape)
    
    def predict_sample(self, branch_input, trunk_input):
        """Make prediction for a single sample"""
        with torch.no_grad():
            branch = torch.FloatTensor(branch_input).unsqueeze(0).to(self.device)
            trunk = torch.FloatTensor(trunk_input).to(self.device)
            
            output = self.model(branch, trunk)
            
        return output.cpu().numpy()[0]  # [n_outputs, n_nodes]
    
    def create_contour_comparison(self, coords, cfd_values, deeponet_values, 
                                  field_name, case_id):
        """Create side-by-side contour comparison"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Denormalize values
        cfd_denorm = self.denormalize_field(cfd_values, field_name)
        deeponet_denorm = self.denormalize_field(deeponet_values, field_name)
        
        # Compute error
        error = np.abs(cfd_denorm - deeponet_denorm)
        relative_error = error / (np.abs(cfd_denorm) + 1e-10)
        
        # Use same color scale for CFD and DeepONet
        vmin = min(cfd_denorm.min(), deeponet_denorm.min())
        vmax = max(cfd_denorm.max(), deeponet_denorm.max())
        
        # CFD ground truth
        scatter1 = axes[0].tricontourf(coords[:, 0], coords[:, 1], cfd_denorm, 
                                       levels=20, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'CFD Ground Truth\n{field_name}')
        axes[0].set_xlabel('x (m)')
        axes[0].set_ylabel('y (m)')
        axes[0].axis('equal')
        plt.colorbar(scatter1, ax=axes[0])
        
        # DeepONet prediction
        scatter2 = axes[1].tricontourf(coords[:, 0], coords[:, 1], deeponet_denorm,
                                       levels=20, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'DeepONet Prediction\n{field_name}')
        axes[1].set_xlabel('x (m)')
        axes[1].set_ylabel('y (m)')
        axes[1].axis('equal')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Error heatmap
        scatter3 = axes[2].tricontourf(coords[:, 0], coords[:, 1], relative_error,
                                       levels=20, cmap='Reds')
        axes[2].set_title(f'Relative Error\n{field_name}')
        axes[2].set_xlabel('x (m)')
        axes[2].set_ylabel('y (m)')
        axes[2].axis('equal')
        plt.colorbar(scatter3, ax=axes[2], label='Relative Error')
        
        plt.suptitle(f'Case {case_id} - {field_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_path = self.output_dir / f'{case_id}_{field_name}_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def visualize_test_case(self, test_idx=0):
        """Visualize a test case"""
        
        # Load test data
        h5_path = self.project_root / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
        
        with h5py.File(h5_path, 'r') as f:
            branch = f['test']['branch'][test_idx]
            trunk = f['test']['trunk'][:]
            target = f['test']['targets'][test_idx]
        
        # Make prediction
        prediction = self.predict_sample(branch, trunk)
        
        # Denormalize coordinates
        trunk_denorm = self.scalers['trunk'].inverse_transform(trunk)
        
        # Create visualizations for each field
        print(f"\nGenerating visualizations for test case {test_idx}...")
        
        for i, field_name in enumerate(self.field_names):
            plot_path = self.create_contour_comparison(
                trunk_denorm,
                target[i],
                prediction[i],
                field_name,
                f"test_{test_idx}"
            )
            print(f"✓ Saved {field_name}: {plot_path}")
        
        # Compute metrics
        print("\nMetrics for this case:")
        for i, field_name in enumerate(self.field_names):
            pred = prediction[i]
            targ = target[i]
            
            rel_error = np.linalg.norm(pred - targ) / np.linalg.norm(targ)
            mae = np.mean(np.abs(pred - targ))
            
            print(f"  {field_name}:")
            print(f"    Relative L2 Error: {rel_error:.4f}")
            print(f"    MAE (normalized): {mae:.6f}")
    
    def visualize_multiple_cases(self, n_cases=5):
        """Visualize multiple test cases"""
        for i in range(n_cases):
            print(f"\n{'='*60}")
            print(f"Test Case {i}")
            print('='*60)
            self.visualize_test_case(i)


def main():
    """Main execution"""
    config_path = project_root / "configs" / "config.yaml"
    model_path = project_root / "results" / "models" / "best_model.pth"
    scalers_path = project_root / "data" / "deeponet_dataset" / "scalers.pkl"
    
    # Check if files exist
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python src/deeponet/train.py")
        return
    
    if not scalers_path.exists():
        print(f"ERROR: Scalers not found at {scalers_path}")
        print("\nPlease run preprocessing first:")
        print("  python src/preprocessing/prepare_deeponet_data.py")
        return
    
    # Create visualizer
    visualizer = DeepONetVisualizer(config_path, model_path, scalers_path)
    
    # Visualize test cases
    visualizer.visualize_multiple_cases(n_cases=5)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
