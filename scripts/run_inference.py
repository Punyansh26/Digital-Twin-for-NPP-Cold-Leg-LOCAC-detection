"""
Run inference pipeline
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.run_inference import main as inference_main


if __name__ == "__main__":
    print("="*60)
    print("STEP 8: Run Inference Pipeline")
    print("="*60)
    
    inference_main()
