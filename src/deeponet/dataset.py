"""
PyTorch Dataset for DeepONet
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


class DeepONetDataset(Dataset):
    """Dataset for DeepONet training"""
    
    def __init__(self, h5_path, split='train'):
        """
        Args:
            h5_path: Path to HDF5 dataset file
            split: 'train', 'val', or 'test'
        """
        self.h5_path = h5_path
        self.split = split
        
        # Load data into memory (assuming it fits)
        with h5py.File(h5_path, 'r') as f:
            self.branch_data = f[split]['branch'][:]
            self.trunk_data = f[split]['trunk'][:]
            self.targets = f[split]['targets'][:]
        
        self.n_samples = len(self.branch_data)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            branch: [branch_dim]
            trunk: [n_nodes, trunk_dim]
            target: [n_outputs, n_nodes]
        """
        branch = torch.FloatTensor(self.branch_data[idx])
        trunk = torch.FloatTensor(self.trunk_data)  # Same for all samples
        target = torch.FloatTensor(self.targets[idx])
        
        return branch, trunk, target


def create_dataloaders(h5_path, batch_size, num_workers=4):
    """Create train, val, test dataloaders"""
    
    # Create datasets
    train_dataset = DeepONetDataset(h5_path, 'train')
    val_dataset = DeepONetDataset(h5_path, 'val')
    test_dataset = DeepONetDataset(h5_path, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    h5_path = project_root / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
    
    if h5_path.exists():
        dataset = DeepONetDataset(h5_path, 'train')
        print(f"Dataset size: {len(dataset)}")
        
        branch, trunk, target = dataset[0]
        print(f"Branch shape: {branch.shape}")
        print(f"Trunk shape: {trunk.shape}")
        print(f"Target shape: {target.shape}")
    else:
        print(f"Dataset not found: {h5_path}")
        print("Run preprocessing first:")
        print("  python src/preprocessing/prepare_deeponet_data.py")
