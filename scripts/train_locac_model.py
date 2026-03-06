"""
Train LOCAC detection model
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.accident_model.train_locac_model import main as train_locac_main


if __name__ == "__main__":
    print("="*60)
    print("STEP 6: Train LOCAC Detection Model")
    print("="*60)
    
    train_locac_main()
