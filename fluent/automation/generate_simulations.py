"""
ANSYS Fluent Automation Script
Generates parameter sweep for AP1000 cold-leg LOCA simulations
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import subprocess
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class FluentSimulationGenerator:
    """Generate and run Fluent simulations with parameter sweep"""
    
    def __init__(self, config_path):
        """Initialize generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = project_root
        self.fluent_dir = self.project_root / "fluent"
        self.data_dir = self.project_root / "data"
        self.output_dir = self.data_dir / "fluent_raw"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_parameter_combinations(self):
        """Generate all parameter combinations for sweep"""
        sweep_config = self.config['parameter_sweep']
        
        # Create ranges
        velocities = np.linspace(
            sweep_config['velocity']['min'],
            sweep_config['velocity']['max'],
            sweep_config['velocity']['samples']
        )
        
        break_sizes = np.linspace(
            sweep_config['break_size']['min'],
            sweep_config['break_size']['max'],
            sweep_config['break_size']['samples']
        )
        
        temperatures = np.linspace(
            sweep_config['temperature']['min'],
            sweep_config['temperature']['max'],
            sweep_config['temperature']['samples']
        )
        
        # Generate all combinations
        combinations = list(product(velocities, break_sizes, temperatures))
        
        # Create DataFrame
        params_df = pd.DataFrame(combinations, 
                                columns=['velocity', 'break_size', 'temperature'])
        params_df['case_id'] = [f"case_{i:04d}" for i in range(len(params_df))]
        
        print(f"Generated {len(params_df)} parameter combinations")
        
        # Save parameter file
        params_file = self.data_dir / "fluent_processed" / "simulation_parameters.csv"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(params_file, index=False)
        
        return params_df
    
    def create_journal_file(self, case_id, velocity, break_size, temperature):
        """Create Fluent journal file for specific parameters"""
        
        # Load template
        template_path = self.fluent_dir / "journals" / "template_simulation.jou"
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Calculate properties
        density = self.config['fluent']['fluid']['density']
        specific_heat = 5000  # J/kg-K for pressurized water
        thermal_conductivity = 0.6  # W/m-K
        viscosity = 0.0001  # kg/m-s
        outlet_pressure = self.config['fluent']['boundary_conditions']['outlet']['pressure']
        
        # Convert temperature to Kelvin
        temp_kelvin = temperature + 273.15
        
        # Break diameter in meters
        pipe_diameter = self.config['geometry']['pipe_diameter']
        break_diameter = (break_size / 100.0) * pipe_diameter
        
        # Break boundary conditions (if applicable)
        break_bc_commands = ""
        if break_size > 0:
            # Add mass flow outlet for break
            break_bc_commands = f"; Break opening\n/define/boundary-conditions/pressure-outlet break yes no 101325 no {temp_kelvin} no yes no no yes 5 10\n"
        
        # Substitute parameters
        journal_content = template.format(
            case_file="cold_leg_base.cas.h5",
            density=density,
            specific_heat=specific_heat,
            thermal_conductivity=thermal_conductivity,
            viscosity=viscosity,
            velocity=velocity,
            temperature=temp_kelvin,
            outlet_pressure=outlet_pressure,
            break_bc_commands=break_bc_commands,
            iterations=1000,
            output_file=str(self.output_dir / f"{case_id}.csv"),
            case_name=f"solved_{case_id}"
        )
        
        # Save journal file
        journal_path = self.fluent_dir / "journals" / f"{case_id}.jou"
        with open(journal_path, 'w') as f:
            f.write(journal_content)
        
        return journal_path
    
    def run_fluent_batch(self, journal_path, case_id):
        """Run Fluent in batch mode with journal file"""
        
        # Fluent command (adjust path for your installation)
        # Example: "C:/Program Files/ANSYS Inc/v232/fluent/ntbin/win64/fluent.exe"
        fluent_exe = "fluent"  # Assumes Fluent is in PATH
        
        # Command arguments
        cmd = [
            fluent_exe,
            "3d",  # 3D mode
            "-g",  # No GUI
            "-i", str(journal_path),  # Input journal
        ]
        
        print(f"Running Fluent for {case_id}...")
        
        try:
            # Run Fluent
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"✓ {case_id} completed successfully")
                return True
            else:
                print(f"✗ {case_id} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ {case_id} timed out")
            return False
        except FileNotFoundError:
            print(f"Error: Fluent executable not found. Please set correct path.")
            print(f"NOTE: You need ANSYS Fluent installed to run CFD simulations.")
            return False
    
    def generate_all_simulations(self, run_fluent=False):
        """Generate all simulation journals and optionally run them"""
        
        # Generate parameter combinations
        params_df = self.generate_parameter_combinations()
        
        # Create journal files
        print("\nGenerating journal files...")
        for _, row in params_df.iterrows():
            self.create_journal_file(
                row['case_id'],
                row['velocity'],
                row['break_size'],
                row['temperature']
            )
        
        print(f"✓ Generated {len(params_df)} journal files")
        
        # Run simulations if requested
        if run_fluent:
            print("\nRunning Fluent simulations...")
            print("NOTE: This will take several hours for 2000 simulations")
            print("Ensure ANSYS Fluent is installed and in PATH\n")
            
            results = []
            for _, row in params_df.iterrows():
                journal_path = self.fluent_dir / "journals" / f"{row['case_id']}.jou"
                success = self.run_fluent_batch(journal_path, row['case_id'])
                results.append(success)
            
            # Summary
            successful = sum(results)
            print(f"\n{'='*60}")
            print(f"Simulation Summary:")
            print(f"Total: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"{'='*60}")
        else:
            print("\nJournal files generated. To run Fluent simulations:")
            print("  python fluent/automation/generate_simulations.py --run")
            print("\nOr run individual simulations with:")
            print("  fluent 3d -g -i fluent/journals/case_XXXX.jou")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Fluent simulations')
    parser.add_argument('--run', action='store_true', 
                       help='Run Fluent simulations (requires ANSYS Fluent)')
    parser.add_argument('--config', type=str, 
                       default='configs/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    config_path = project_root / args.config
    
    # Create generator
    generator = FluentSimulationGenerator(config_path)
    
    # Generate (and optionally run) simulations
    generator.generate_all_simulations(run_fluent=args.run)


if __name__ == "__main__":
    main()
