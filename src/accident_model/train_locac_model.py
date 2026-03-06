"""
LOCAC Detection Model
Train classifier on NPPAD dataset
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class LOCACDetector:
    """LOCAC detection model"""
    
    def __init__(self, config_path):
        """Initialize detector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = project_root
        self.model_type = self.config['locac_model']['type']
        
        # Initialize model
        if self.model_type == 'gradient_boosting':
            params = self.config['locac_model']['gb_params']
            self.model = GradientBoostingClassifier(**params, random_state=42)
        else:  # neural_network
            params = self.config['locac_model']['nn_params']
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(params['hidden_dims']),
                max_iter=1000,
                random_state=42
            )
        
        self.scaler = StandardScaler()
        
    def load_nppad_data(self):
        """
        Load and process NPPAD dataset
        
        Note: NPPAD dataset structure needs to be adapted to your actual data
        This is a template that should be modified based on actual NPPAD format
        """
        nppad_dir = self.project_root / "data" / "nppad"
        
        # Check if NPPAD data exists
        if not nppad_dir.exists():
            print(f"WARNING: NPPAD directory not found: {nppad_dir}")
            print("Using synthetic data for demonstration...")
            return self.generate_synthetic_nppad_data()
        
        # Load operation and dose data
        operation_dir = nppad_dir / "operation_csv_data"
        dose_dir = nppad_dir / "dose_csv_data"
        
        if not operation_dir.exists():
            print("WARNING: NPPAD data not found, using synthetic data...")
            return self.generate_synthetic_nppad_data()
        
        # Load CSV files
        all_data = []
        all_labels = []
        
        for csv_file in operation_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            
            # Extract relevant features (adjust column names based on actual data)
            # Example columns: 'primary_pressure', 'coolant_flow', 'temperature', etc.
            
            # Determine if LOCAC based on filename or column
            is_locac = 'loca' in csv_file.stem.lower()
            
            all_data.append(df)
            all_labels.extend([int(is_locac)] * len(df))
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return combined_df, np.array(all_labels)
    
    def generate_synthetic_nppad_data(self):
        """Generate synthetic NPPAD-like data for demonstration"""
        
        print("Generating synthetic NPPAD data...")
        
        n_normal = 1000
        n_locac = 400
        
        # Normal operation
        normal_data = {
            'primary_pressure': np.random.normal(15.5e6, 0.1e6, n_normal),
            'coolant_flow': np.random.normal(15000, 500, n_normal),
            'temperature': np.random.normal(305, 5, n_normal),
            'pressure_drop': np.random.normal(50000, 5000, n_normal),
            'max_turbulence': np.random.normal(0.5, 0.1, n_normal),
            'temperature_difference': np.random.normal(5, 1, n_normal),
            'velocity_std': np.random.normal(0.2, 0.05, n_normal),
        }
        
        # LOCAC events
        locac_data = {
            'primary_pressure': np.random.normal(13.0e6, 1.0e6, n_locac),  # Lower pressure
            'coolant_flow': np.random.normal(10000, 2000, n_locac),  # Reduced flow
            'temperature': np.random.normal(280, 15, n_locac),  # Lower temp
            'pressure_drop': np.random.normal(100000, 20000, n_locac),  # Higher drop
            'max_turbulence': np.random.normal(2.0, 0.5, n_locac),  # Higher turbulence
            'temperature_difference': np.random.normal(20, 5, n_locac),  # Larger variation
            'velocity_std': np.random.normal(0.8, 0.2, n_locac),  # Higher variation
        }
        
        # Combine
        normal_df = pd.DataFrame(normal_data)
        locac_df = pd.DataFrame(locac_data)
        
        combined_df = pd.concat([normal_df, locac_df], ignore_index=True)
        labels = np.array([0] * n_normal + [1] * n_locac)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(labels))
        combined_df = combined_df.iloc[shuffle_idx].reset_index(drop=True)
        labels = labels[shuffle_idx]
        
        print(f"✓ Generated {n_normal} normal and {n_locac} LOCAC samples")
        
        return combined_df, labels
    
    def train(self, X, y):
        """Train LOCAC detection model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"LOCAC ratio: {np.mean(y_train):.2%}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n" + "="*60)
        print("LOCAC Detection Model Performance")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'LOCAC']))
        
        # Check target accuracy
        target_acc = self.config['locac_model']['target_accuracy']
        if metrics['accuracy'] >= target_acc:
            print(f"\n✓ Target accuracy ({target_acc:.0%}) achieved!")
        else:
            print(f"\n⚠ Below target accuracy ({target_acc:.0%})")
        
        # Store results
        self.metrics = metrics
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.confusion_matrix = cm
        
        return metrics
    
    def plot_performance(self):
        """Plot performance metrics"""
        
        output_dir = self.project_root / self.config['output_paths']['plots']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(15, 5))
        
        # ROC Curve
        ax1 = plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {self.metrics["roc_auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Precision-Recall Curve
        ax2 = plt.subplot(1, 3, 2)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        ax2.plot(recall, precision)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True)
        
        # Confusion Matrix
        ax3 = plt.subplot(1, 3, 3)
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'LOCAC'],
                   yticklabels=['Normal', 'LOCAC'],
                   ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        plot_path = output_dir / 'locac_detection_performance.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved performance plots to {plot_path}")
    
    def save_model(self):
        """Save trained model"""
        
        model_dir = self.project_root / self.config['output_paths']['models']
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'config': self.config
        }
        
        model_path = model_dir / 'locac_detector.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved model to {model_path}")
    
    def predict(self, features):
        """
        Predict LOCAC probability from features
        
        Args:
            features: Feature array or DataFrame
            
        Returns:
            probability: LOCAC probability (0-1)
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(features_scaled)[0, 1]
        
        return proba


def main():
    """Main execution"""
    config_path = project_root / "configs" / "config.yaml"
    
    # Create detector
    detector = LOCACDetector(config_path)
    
    # Load data
    print("Loading NPPAD data...")
    X, y = detector.load_nppad_data()
    
    # Train
    metrics = detector.train(X, y)
    
    # Plot performance
    detector.plot_performance()
    
    # Save model
    detector.save_model()
    
    print("\n" + "="*60)
    print("LOCAC Detection Model Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
