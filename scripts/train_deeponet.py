"""
Train DeepONet model
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deeponet.train import main as train_main


if __name__ == "__main__":
    print("="*60)
    print("STEP 3: Train DeepONet Model")
    print("="*60)
    
    train_main()
