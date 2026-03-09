"""Master run script - Execute entire pipeline"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(script_path, description, extra_args=None):
    """Run a Python script and check for errors"""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70 + "\n")

    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=project_root, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Error in {description}")
        return False

    print(f"\n✅ {description} completed successfully\n")
    return True


def _resolve_model_version(cli_version, config_path):
    """Return the active model version: CLI arg > config file > default."""
    import yaml
    if cli_version:
        return cli_version
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model_version") or cfg.get("operator") or "deeponet_fourier"
    except Exception:
        return "deeponet_fourier"


def _print_banner(model_version):
    """Print the version/tier banner."""
    import torch
    from src.core.model_versions import get_tier_label

    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = f"cuda ({torch.cuda.get_device_name(0)})"

    print("\n" + "=" * 43)
    print("   AP1000 LOCAC DIGITAL TWIN SYSTEM   ")
    print("=" * 43)
    print(f"  Selected Operator : {model_version}")
    print(f"  Tier              : {get_tier_label(model_version)}")
    print(f"  Device            : {device_str}")
    print(f"  Dataset           : AP1000 Cold-Leg CFD")
    print("=" * 43 + "\n")


def main():
    """Run full pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Run AP1000 Digital Twin Pipeline')
    parser.add_argument('--use-mock-data', action='store_true',
                       help='Use mock data instead of Fluent simulations')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing models)')
    parser.add_argument(
        '--model-version',
        type=str,
        default=None,
        dest='model_version',
        choices=['deeponet', 'deeponet_fourier', 'transolver', 'clifford'],
        help='Select neural operator architecture (overrides configs/model_config.yaml)',
    )

    args = parser.parse_args()

    config_path   = project_root / 'configs' / 'model_config.yaml'
    model_version = _resolve_model_version(args.model_version, config_path)

    _print_banner(model_version)

    # Extra args propagated to sub-scripts
    version_args = ['--model-version', model_version]

    print(f"Configuration:")
    print(f"  Mock Data     : {args.use_mock_data}")
    print(f"  Skip Training : {args.skip_training}")
    print(f"  Model Version : {model_version}")
    print()
    
    steps = []

    # Step 1: Generate data
    if args.use_mock_data:
        steps.append((
            project_root / "scripts" / "generate_mock_data.py",
            "STEP 1: Generate Mock CFD Data",
            [],
        ))
    else:
        print("\nWARNING: Using real Fluent data requires ANSYS Fluent")
        print("Journals will be generated but not executed")
        print("Use --use-mock-data flag for automated testing\n")
        steps.append((
            project_root / "scripts" / "generate_dataset.py",
            "STEP 1: Generate Fluent Journals",
            [],
        ))

    # Step 2: Preprocess
    steps.append((
        project_root / "src" / "preprocessing" / "prepare_deeponet_data.py",
        "STEP 2: Preprocess Data for DeepONet",
        [],
    ))

    # Step 3: Train operator (version-aware)
    if not args.skip_training:
        if model_version in ('transolver', 'clifford'):
            steps.append((
                project_root / "scripts" / "train_operator.py",
                f"STEP 3: Train {model_version.capitalize()} Operator",
                ['--model-version', model_version],
            ))
        else:
            steps.append((
                project_root / "scripts" / "train_deeponet.py",
                f"STEP 3: Train DeepONet Model ({model_version})",
                ['--model-version', model_version],
            ))

    # Step 4: Visualize
    if not args.skip_training:
        steps.append((
            project_root / "src" / "deeponet" / "visualize.py",
            "STEP 4: Visualize DeepONet Predictions",
            [],
        ))

    # Step 5: Train LOCAC detector
    if not args.skip_training:
        steps.append((
            project_root / "scripts" / "train_locac_model.py",
            "STEP 5: Train LOCAC Detection Model",
            [],
        ))

    # Step 6: Run inference (version-aware)
    steps.append((
        project_root / "scripts" / "run_inference.py",
        f"STEP 6: Run Inference Pipeline ({model_version})",
        version_args,
    ))

    # Execute steps
    for script_path, description, extra_args in steps:
        success = run_command(script_path, description, extra_args)
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
