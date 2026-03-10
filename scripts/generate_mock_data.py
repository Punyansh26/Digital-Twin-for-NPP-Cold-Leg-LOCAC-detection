"""
Generate mock CFD data for testing
Use this if you don't have ANSYS Fluent installed
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm


def generate_mesh(n_nodes=25000, seed=42):
    """
    Generate a fixed mesh inside a cylindrical pipe.
    Called once; reused for every simulation so all cases share coordinates.
    """
    rng = np.random.RandomState(seed)
    pipe_diameter = 0.7   # metres
    pipe_length = 7.0     # 10 D

    # Over-sample then filter to interior of cylinder
    n_oversample = int(n_nodes * 1.35)
    x = rng.uniform(0, pipe_length, n_oversample)
    y = rng.uniform(-pipe_diameter / 2, pipe_diameter / 2, n_oversample)
    z = rng.uniform(-pipe_diameter / 2, pipe_diameter / 2, n_oversample)
    radius = np.sqrt(y ** 2 + z ** 2)
    mask = radius <= pipe_diameter / 2
    x, y, z = x[mask][:n_nodes], y[mask][:n_nodes], z[mask][:n_nodes]

    # Pad if a few nodes are missing
    while len(x) < n_nodes:
        deficit = n_nodes - len(x)
        x = np.concatenate([x, x[:deficit]])
        y = np.concatenate([y, y[:deficit]])
        z = np.concatenate([z, z[:deficit]])
    return x[:n_nodes], y[:n_nodes], z[:n_nodes]


def generate_mock_cfd_simulation(x, y, z, velocity, break_size, temperature,
                                 rng=None):
    """
    Generate synthetic CFD field values on a *fixed* mesh.

    Physics improvements over the original version:
    - Break creates a localized pressure drop + turbulence spike near the break
    - Velocity profile flattens near the break region
    - Temperature develops a hot-spot downstream of the break
    """
    if rng is None:
        rng = np.random.RandomState()

    n_nodes = len(x)
    pipe_diameter = 0.7
    pipe_length = 7.0
    radial_position = np.sqrt(y ** 2 + z ** 2) / (pipe_diameter / 2)

    # -- Break location & influence envelope --------------------------------
    break_x = 3.5                       # break at pipe midpoint
    break_sigma = 0.8                   # axial spread of break effect
    break_strength = break_size / 10.0  # 0‒1 normalised severity
    break_envelope = np.exp(-0.5 * ((x - break_x) / break_sigma) ** 2)

    # -- Pressure -----------------------------------------------------------
    baseline_pressure = 15.5e6  # Pa
    # Global linear drop along pipe
    dp_global = 50000 * velocity / 5.0
    pressure = baseline_pressure - (x / pipe_length) * dp_global
    # Localised pressure drop at break (up to 200 kPa extra at full break)
    pressure -= break_strength * 200000 * break_envelope
    # Radial pressure gradient (centrifugal effect)
    pressure += 5000 * (1 - radial_position ** 2)
    pressure += rng.normal(0, 2000, n_nodes)

    # -- Velocity -----------------------------------------------------------
    velocity_factor = 1.0 - break_strength * 0.30
    # Parabolic profile, flattened near break
    profile_exp = 2.0 - break_strength * break_envelope * 0.8  # flatter near break
    vel_profile = velocity * velocity_factor * np.maximum(1 - radial_position ** profile_exp, 0)
    # Axial acceleration/deceleration around break
    vel_profile *= (1 - 0.3 * break_strength * break_envelope)
    velocity_magnitude = np.maximum(vel_profile + rng.normal(0, 0.05, n_nodes), 0)

    velocity_x = velocity_magnitude * (0.9 + 0.05 * break_envelope * break_strength)
    velocity_y = velocity_magnitude * 0.05 * rng.randn(n_nodes)
    velocity_z = velocity_magnitude * 0.05 * rng.randn(n_nodes)

    # -- Turbulence kinetic energy ------------------------------------------
    base_k = 0.01 * velocity ** 2
    # Wall‐layer increase + break‐localised spike
    turbulence_k = base_k * (1 + 1.5 * radial_position)
    turbulence_k += base_k * 5.0 * break_strength * break_envelope * (1 + radial_position)
    turbulence_k += rng.uniform(0, 0.02, n_nodes)

    # -- Temperature --------------------------------------------------------
    temp_kelvin = temperature + 273.15
    # Slight radial gradient (warmer core) + break‐induced hot spot downstream
    temperature_field = temp_kelvin + 3.0 * (1 - radial_position ** 2)
    downstream = np.clip((x - break_x) / pipe_length, 0, 1)
    temperature_field += break_strength * 15.0 * downstream * break_envelope
    # Axial cooling along pipe
    temperature_field -= 2.0 * (x / pipe_length)
    temperature_field += rng.normal(0, 0.5, n_nodes)

    return pd.DataFrame({
        'x-coordinate': x,
        'y-coordinate': y,
        'z-coordinate': z,
        'pressure': pressure,
        'velocity-magnitude': velocity_magnitude,
        'x-velocity': velocity_x,
        'y-velocity': velocity_y,
        'z-velocity': velocity_z,
        'turb-kinetic-energy': turbulence_k,
        'temperature': temperature_field,
    })


def main():
    """Generate mock dataset"""

    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = project_root / "data" / "fluent_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_file = project_root / "data" / "fluent_processed" / "simulation_parameters.csv"

    if params_file.exists():
        params_df = pd.read_csv(params_file)
    else:
        print("Generating parameter combinations...")
        sweep_config = config['parameter_sweep']

        velocities = np.linspace(
            sweep_config['velocity']['min'],
            sweep_config['velocity']['max'],
            sweep_config['velocity']['samples'],
        )
        break_sizes = np.linspace(
            sweep_config['break_size']['min'],
            sweep_config['break_size']['max'],
            sweep_config['break_size']['samples'],
        )
        temperatures = np.linspace(
            sweep_config['temperature']['min'],
            sweep_config['temperature']['max'],
            sweep_config['temperature']['samples'],
        )

        from itertools import product
        combinations = list(product(velocities, break_sizes, temperatures))

        params_df = pd.DataFrame(
            combinations, columns=['velocity', 'break_size', 'temperature']
        )
        params_df['case_id'] = [f"case_{i:04d}" for i in range(len(params_df))]

        params_file.parent.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(params_file, index=False)
        print(f"✓ Generated {len(params_df)} parameter combinations")

    # ---- Generate a SINGLE shared mesh ----
    print("Generating shared mesh (25 000 nodes)...")
    x, y, z = generate_mesh(n_nodes=25000)

    print(f"\nGenerating {len(params_df)} mock CFD simulations...")
    print("(All cases share the same mesh coordinates)\n")

    rng = np.random.RandomState(0)
    for _, row in tqdm(params_df.iterrows(), total=len(params_df)):
        case_id = row['case_id']
        output_file = output_dir / f"{case_id}.csv"

        df = generate_mock_cfd_simulation(
            x, y, z,
            row['velocity'],
            row['break_size'],
            row['temperature'],
            rng=rng,
        )
        
        # Save
        df.to_csv(output_file, index=False)
    
    print(f"\n✓ Generated {len(params_df)} mock CFD files in {output_dir}")
    print("\nNOTE: This is synthetic data for demonstration.")
    print("For actual CFD data, run ANSYS Fluent simulations.")


if __name__ == "__main__":
    main()
