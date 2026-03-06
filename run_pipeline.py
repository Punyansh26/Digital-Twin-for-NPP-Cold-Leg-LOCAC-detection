"""Master run script - Execute entire pipeline"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent


def run_command(script_path, description):
    """Run a Python script and check for errors"""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70 + "\n")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error in {description}")
        return False
    
    print(f"\n✅ {description} completed successfully\n")
    return True


def main():
    """Run full pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AP1000 Digital Twin Pipeline')
    parser.add_argument('--use-mock-data', action='store_true',
                       help='Use mock data instead of Fluent simulations')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing models)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AP1000 DIGITAL TWIN - FULL PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Mock Data: {args.use_mock_data}")
    print(f"  Skip Training: {args.skip_training}")
    print()
    
    steps = []
    
    # Step 1: Generate data
    if args.use_mock_data:
        steps.append((
            project_root / "scripts" / "generate_mock_data.py",
            "STEP 1: Generate Mock CFD Data"
        ))
    else:
        print("\nWARNING: Using real Fluent data requires ANSYS Fluent")
        print("Journals will be generated but not executed")
        print("Use --use-mock-data flag for automated testing\n")
        steps.append((
            project_root / "scripts" / "generate_dataset.py",
            "STEP 1: Generate Fluent Journals"
        ))
    
    # Step 2: Preprocess
    steps.append((
        project_root / "src" / "preprocessing" / "prepare_deeponet_data.py",
        "STEP 2: Preprocess Data for DeepONet"
    ))
    
    # Step 3: Train DeepONet
    if not args.skip_training:
        steps.append((
            project_root / "scripts" / "train_deeponet.py",
            "STEP 3: Train DeepONet Model"
        ))
    
    # Step 4: Visualize
    if not args.skip_training:
        steps.append((
            project_root / "src" / "deeponet" / "visualize.py",
            "STEP 4: Visualize DeepONet Predictions"
        ))
    
    # Step 5: Train LOCAC detector
    if not args.skip_training:
        steps.append((
            project_root / "scripts" / "train_locac_model.py",
            "STEP 5: Train LOCAC Detection Model"
        ))
    
    # Step 6: Run inference
    steps.append((
        project_root / "scripts" / "run_inference.py",
        "STEP 6: Run Inference Pipeline"
    ))
    
    # Execute steps
    for script_path, description in steps:
        success = run_command(script_path, description)
        if not success:
            print("\n" + "="*70)
            print("PIPELINE ABORTED DUE TO ERROR")
            print("="*70)
            sys.exit(1)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved in:")
    print(f"  Models: results/models/")
    print(f"  Plots: results/plots/")
    print(f"  Metrics: results/metrics/")
    print("\nNext steps:")
    print("  1. Review visualizations in results/plots/")
    print("  2. Check model performance metrics")
    print("  3. Run custom inference with: python scripts/run_inference.py")
    print("="*70)


if __name__ == "__main__":
    main()
