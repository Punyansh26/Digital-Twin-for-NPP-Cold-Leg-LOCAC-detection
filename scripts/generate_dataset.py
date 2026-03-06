"""
Master script to generate CFD dataset
Orchestrates Fluent simulation generation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fluent.automation.generate_simulations import FluentSimulationGenerator


def main():
    """Generate CFD dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CFD dataset')
    parser.add_argument('--run-fluent', action='store_true',
                       help='Actually run Fluent simulations (requires ANSYS Fluent)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    config_path = project_root / args.config
    
    print("="*60)
    print("STEP 1: Generate CFD Dataset")
    print("="*60)
    
    generator = FluentSimulationGenerator(config_path)
    generator.generate_all_simulations(run_fluent=args.run_fluent)
    
    if not args.run_fluent:
        print("\n" + "="*60)
        print("IMPORTANT:")
        print("="*60)
        print("Journal files have been generated in fluent/journals/")
        print("\nTo run actual CFD simulations, you need ANSYS Fluent installed.")
        print("Options:")
        print("  1. Run with --run-fluent flag (automated)")
        print("  2. Manually run each journal file in Fluent")
        print("  3. Use mock data for demonstration (see generate_mock_data.py)")
        print("="*60)


if __name__ == "__main__":
    main()
