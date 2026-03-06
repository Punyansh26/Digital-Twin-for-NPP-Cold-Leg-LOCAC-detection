"""
Generate mock CFD data for testing
Use this if you don't have ANSYS Fluent installed
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm


def generate_mock_cfd_simulation(velocity, break_size, temperature, n_nodes=25000):
    """
    Generate synthetic CFD data that mimics Fluent output
    
    Args:
        velocity: Flow velocity (m/s)
        break_size: Break size (% of diameter)
        temperature: Temperature (°C)
        n_nodes: Number of mesh nodes
        
    Returns:
        DataFrame with CFD results
    """
    
    # Generate mesh coordinates for 90-degree elbow pipe
    # Simplified geometry: straight pipe + elbow
    
    pipe_diameter = 0.7  # meters
    pipe_length = 7.0  # 10D total
    
    # Random distribution in pipe volume (simplified)
    # In reality, mesh would be structured
    
    # Cartesian coordinates
    x = np.random.uniform(0, pipe_length, n_nodes)
    y = np.random.uniform(-pipe_diameter/2, pipe_diameter/2, n_nodes)
    z = np.random.uniform(-pipe_diameter/2, pipe_diameter/2, n_nodes)
    
    # Filter to keep only points inside pipe (simplified cylinder check)
    radius = np.sqrt(y**2 + z**2)
    mask = radius <= pipe_diameter/2
    
    x = x[mask][:n_nodes]
    y = y[mask][:n_nodes]
    z = z[mask][:n_nodes]
    
    # Pad if needed
    if len(x) < n_nodes:
        deficit = n_nodes - len(x)
        x = np.concatenate([x, x[:deficit]])
        y = np.concatenate([y, y[:deficit]])
        z = np.concatenate([z, z[:deficit]])
    
    # Generate field values
    
    # Pressure: decreases along flow, affected by break
    baseline_pressure = 15.5e6  # Pa
    pressure_drop_factor = 50000 * (1 + break_size / 10)
    pressure = baseline_pressure - (x / pipe_length) * pressure_drop_factor
    pressure += np.random.normal(0, 10000, n_nodes)  # Noise
    
    # Velocity: based on input, varies radially, affected by break
    velocity_factor = 1.0 - (break_size / 100) * 0.3  # Flow reduces with break
    radial_position = np.sqrt(y**2 + z**2) / (pipe_diameter/2)
    velocity_profile = velocity * velocity_factor * (1 - radial_position**2)  # Parabolic
    velocity_magnitude = np.maximum(velocity_profile + np.random.normal(0, 0.1, n_nodes), 0)
    
    # Velocity components (simplified, mostly in x-direction)
    velocity_x = velocity_magnitude * 0.9
    velocity_y = velocity_magnitude * 0.05 * np.random.randn(n_nodes)
    velocity_z = velocity_magnitude * 0.05 * np.random.randn(n_nodes)
    
    # Turbulence kinetic energy: higher at break, elbow
    base_turbulence = 0.01 * velocity**2  # Typical k value
    turbulence_k = base_turbulence * (1 + break_size / 10) * (1 + 2 * radial_position)
    turbulence_k += np.random.uniform(0, 0.1, n_nodes)
    
    # Temperature: based on input, slight variations
    temp_kelvin = temperature + 273.15
    temperature_field = temp_kelvin + np.random.normal(0, 2, n_nodes)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x-coordinate': x,
        'y-coordinate': y,
        'z-coordinate': z,
        'pressure': pressure,
        'velocity-magnitude': velocity_magnitude,
        'x-velocity': velocity_x,
        'y-velocity': velocity_y,
        'z-velocity': velocity_z,
        'turb-kinetic-energy': turbulence_k,
        'temperature': temperature_field
    })
    
    return df


def main():
    """Generate mock dataset"""
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Output directory
    output_dir = project_root / "data" / "fluent_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parameters
    params_file = project_root / "data" / "fluent_processed" / "simulation_parameters.csv"
    
    if params_file.exists():
        params_df = pd.read_csv(params_file)
    else:
        print("Generating parameter combinations...")
        # Generate parameters
        sweep_config = config['parameter_sweep']
        
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
        
        from itertools import product
        combinations = list(product(velocities, break_sizes, temperatures))
        
        params_df = pd.DataFrame(combinations, 
                                columns=['velocity', 'break_size', 'temperature'])
        params_df['case_id'] = [f"case_{i:04d}" for i in range(len(params_df))]
        
        params_file.parent.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(params_file, index=False)
        
        print(f"✓ Generated {len(params_df)} parameter combinations")
    
    print(f"\nGenerating {len(params_df)} mock CFD simulations...")
    print("This may take a few minutes...\n")
    
    # Generate simulations
    for _, row in tqdm(params_df.iterrows(), total=len(params_df)):
        case_id = row['case_id']
        output_file = output_dir / f"{case_id}.csv"
        
        if output_file.exists():
            continue  # Skip if already exists
        
        # Generate mock data
        df = generate_mock_cfd_simulation(
            row['velocity'],
            row['break_size'],
            row['temperature']
        )
        
        # Save
        df.to_csv(output_file, index=False)
    
    print(f"\n✓ Generated {len(params_df)} mock CFD files in {output_dir}")
    print("\nNOTE: This is synthetic data for demonstration.")
    print("For actual CFD data, run ANSYS Fluent simulations.")


if __name__ == "__main__":
    main()
