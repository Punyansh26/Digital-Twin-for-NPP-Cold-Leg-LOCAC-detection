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
        
    # NPPAD columns used for LOCAC detection (must match inference feature order)
    FEATURE_COLUMNS = ['P', 'TAVG', 'WRCA', 'PSGA', 'SCMA', 'DNBR', 'DT_HL_CL']

    def load_nppad_data(self):
        """
        Load and process real NPPAD dataset from data/nppad/operation_csv_data/.
        Uses Normal/ and LOCAC/ subdirectories. Each CSV has time-series rows
        with columns like P, TAVG, WRCA, PSGA, SCMA, DNBR, THA, TCA, etc.
        """
        nppad_dir = self.project_root / "data" / "nppad" / "operation_csv_data"

        # Fall back to scripts/data/nppad if main data dir missing
        if not nppad_dir.exists():
            nppad_dir = self.project_root / "scripts" / "data" / "nppad" / "operation_csv_data"

        normal_dir = nppad_dir / "Normal"
        locac_dir = nppad_dir / "LOCAC"

        if not normal_dir.exists() or not locac_dir.exists():
            print("WARNING: NPPAD Normal/LOCAC dirs not found, using synthetic data...")
            return self.generate_synthetic_nppad_data()

        all_rows = []
        all_labels = []

        # Load Normal data (label=0)
        for csv_file in sorted(normal_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df = self._extract_nppad_features(df)
            all_rows.append(df)
            all_labels.extend([0] * len(df))

        # Load LOCAC data (label=1)
        for csv_file in sorted(locac_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df = self._extract_nppad_features(df)
            all_rows.append(df)
            all_labels.extend([1] * len(df))

        combined_df = pd.concat(all_rows, ignore_index=True)
        labels = np.array(all_labels)

        # Add transitional samples that bridge the Normal↔LOCAC gap
        trans_df, trans_labels = self._generate_transitional_data(combined_df, labels)
        combined_df = pd.concat([combined_df, trans_df], ignore_index=True)
        labels = np.concatenate([labels, trans_labels])

        # Shuffle
        shuffle_idx = np.random.RandomState(42).permutation(len(labels))
        combined_df = combined_df.iloc[shuffle_idx].reset_index(drop=True)
        labels = labels[shuffle_idx]

        n_normal = int((labels == 0).sum())
        n_locac = int((labels == 1).sum())
        n_trans = len(trans_labels)
        print(f"✓ Loaded NPPAD data: {n_normal} normal, {n_locac} LOCAC rows")
        print(f"  (includes {n_trans} transitional samples for probability calibration)")

        return combined_df, labels

    @staticmethod
    def _generate_transitional_data(combined_df, labels):
        """
        Generate synthetic transitional samples between Normal and LOCAC
        distributions.  This fills the gap in the training data so the
        GradientBoosting classifier learns to output graded probabilities
        (0.1, 0.3, 0.5, …) instead of snapping to 0 or 1.

        For each severity level s ∈ {0.05, 0.10, …, 0.95} we interpolate
        the feature means/stds and assign label=1 with probability s.
        """
        rng = np.random.RandomState(123)
        cols = LOCACDetector.FEATURE_COLUMNS

        normal_mask = labels == 0
        locac_mask = labels == 1
        normal_mean = combined_df.loc[normal_mask, cols].mean().values
        normal_std  = combined_df.loc[normal_mask, cols].std().values.clip(min=1e-6)
        locac_mean  = combined_df.loc[locac_mask, cols].mean().values
        locac_std   = combined_df.loc[locac_mask, cols].std().values.clip(min=1e-6)

        rows = []
        row_labels = []
        n_per_level = 200

        for s in np.arange(0.05, 1.0, 0.05):  # 19 severity levels
            mean_interp = normal_mean * (1 - s) + locac_mean * s
            std_interp  = normal_std  * (1 - s) + locac_std  * s
            samples = rng.normal(mean_interp, std_interp,
                                 size=(n_per_level, len(cols)))
            # Label each sample as LOCAC with probability = s
            sample_labels = (rng.rand(n_per_level) < s).astype(int)
            rows.append(samples)
            row_labels.append(sample_labels)

        trans_features = np.vstack(rows)
        trans_labels   = np.concatenate(row_labels)
        trans_df = pd.DataFrame(trans_features, columns=cols)
        return trans_df, trans_labels

    @staticmethod
    def _extract_nppad_features(df):
        """Select / compute the 7 features used for classification."""
        out = pd.DataFrame()
        out['P'] = df['P']                    # Primary pressure (bar)
        out['TAVG'] = df['TAVG']              # Average temperature (°C)
        out['WRCA'] = df['WRCA']              # Reactor coolant flow loop-A (kg/s)
        out['PSGA'] = df['PSGA']              # SG-A pressure (bar)
        out['SCMA'] = df['SCMA']              # Subcooling margin A (°C)
        out['DNBR'] = df['DNBR']              # Departure from nucleate boiling ratio
        out['DT_HL_CL'] = df['THA'] - df['TCA']  # Hot-leg minus cold-leg temp (°C)
        return out
    
    def generate_synthetic_nppad_data(self):
        """Generate synthetic NPPAD-like data matching real column names."""

        print("Generating synthetic NPPAD data...")

        n_normal = 1000
        n_locac = 400
        rng = np.random.RandomState(42)

        # Normal operation (based on real NPPAD ranges)
        normal_data = {
            'P':         rng.normal(155.5, 0.3, n_normal),       # bar
            'TAVG':      rng.normal(310.0, 2.0, n_normal),       # °C
            'WRCA':      rng.normal(16515, 100, n_normal),       # kg/s
            'PSGA':      rng.normal(67.0, 0.5, n_normal),        # bar
            'SCMA':      rng.normal(17.2, 0.3, n_normal),        # °C
            'DNBR':      rng.normal(2.3, 0.05, n_normal),        # -
            'DT_HL_CL':  rng.normal(35.6, 1.0, n_normal),        # °C
        }

        # LOCAC events (pressure drops, flows change, margins shrink)
        locac_data = {
            'P':         rng.normal(150.0, 3.0, n_locac),
            'TAVG':      rng.normal(308.0, 5.0, n_locac),
            'WRCA':      rng.normal(14000, 2000, n_locac),
            'PSGA':      rng.normal(65.0, 2.0, n_locac),
            'SCMA':      rng.normal(12.0, 3.0, n_locac),
            'DNBR':      rng.normal(1.8, 0.3, n_locac),
            'DT_HL_CL':  rng.normal(28.0, 5.0, n_locac),
        }

        normal_df = pd.DataFrame(normal_data)
        locac_df = pd.DataFrame(locac_data)

        combined_df = pd.concat([normal_df, locac_df], ignore_index=True)
        labels = np.array([0] * n_normal + [1] * n_locac)

        shuffle_idx = rng.permutation(len(labels))
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
