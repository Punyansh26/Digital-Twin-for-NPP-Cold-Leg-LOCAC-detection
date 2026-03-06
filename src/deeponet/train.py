"""
DeepONet Training Script
Includes metrics, early stopping, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.deeponet.model import DeepONet, DeepONetLoss
from src.deeponet.dataset import create_dataloaders


class MetricsCalculator:
    """Calculate various accuracy metrics"""
    
    @staticmethod
    def relative_l2_error(pred, target):
        """Compute relative L2 error"""
        numerator = torch.norm(pred - target, p=2)
        denominator = torch.norm(target, p=2)
        return (numerator / denominator).item()
    
    @staticmethod
    def r2_score(pred, target):
        """Compute R² score"""
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()
    
    @staticmethod
    def mae(pred, target):
        """Compute mean absolute error"""
        return torch.mean(torch.abs(pred - target)).item()
    
    @staticmethod
    def compute_all_metrics(pred, target, field_names):
        """Compute all metrics for each field"""
        metrics = {}
        
        n_outputs = pred.shape[1]
        for i in range(n_outputs):
            field = field_names[i]
            pred_field = pred[:, i, :]
            target_field = target[:, i, :]
            
            metrics[f'{field}_rel_l2'] = MetricsCalculator.relative_l2_error(pred_field, target_field)
            metrics[f'{field}_r2'] = MetricsCalculator.r2_score(pred_field, target_field)
            metrics[f'{field}_mae'] = MetricsCalculator.mae(pred_field, target_field)
        
        return metrics


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=50, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class DeepONetTrainer:
    """Train DeepONet model"""
    
    def __init__(self, config_path):
        """Initialize trainer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = project_root
        self.output_dir = self.project_root / self.config['output_paths']['models']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['device']['use_cuda'] else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = DeepONet(self.config).to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Loss function
        self.criterion = DeepONetLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['training']['scheduler']['patience'],
            factor=self.config['training']['scheduler']['factor']
        )
        
        # Early stopping
        early_stop_config = self.config['training']['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta']
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.config['training']['mixed_precision'] and torch.cuda.is_available() else None
        
        # Field names
        self.field_names = self.config['deeponet']['output_fields']
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for branch, trunk, target in pbar:
            branch = branch.to(self.device)
            trunk = trunk.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler:
                with autocast('cuda'):
                    output = self.model(branch, trunk)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(branch, trunk)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for branch, trunk, target in tqdm(val_loader, desc='Validation'):
                branch = branch.to(self.device)
                trunk = trunk.to(self.device)
                target = target.to(self.device)
                
                output = self.model(branch, trunk)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = MetricsCalculator.compute_all_metrics(predictions, targets, self.field_names)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        
        print("\n" + "="*60)
        print("Starting DeepONet Training")
        print("="*60)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print("\nField Metrics:")
            for field in self.field_names:
                print(f"  {field}:")
                print(f"    R²: {metrics[f'{field}_r2']:.4f}")
                print(f"    Rel L2: {metrics[f'{field}_rel_l2']:.4f}")
                print(f"    MAE: {metrics[f'{field}_mae']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, metrics)
                print("✓ Saved best model")
            
            # Check early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth', epochs-1, metrics)
        
        # Save training history
        self.save_history()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot loss curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate
        axes[1].plot(self.history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = self.project_root / self.config['output_paths']['plots'] / 'training_curves.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training curves to {plot_path}")


def main():
    """Main execution"""
    config_path = project_root / "configs" / "config.yaml"
    
    # Load data
    h5_path = project_root / "data" / "deeponet_dataset" / "deeponet_dataset.h5"
    
    if not h5_path.exists():
        print(f"ERROR: Dataset not found at {h5_path}")
        print("\nPlease run preprocessing first:")
        print("  python src/preprocessing/prepare_deeponet_data.py")
        return
    
    # Load config to get batch size
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path, batch_size, num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create trainer
    trainer = DeepONetTrainer(config_path)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
