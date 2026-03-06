"""
Create .gitkeep files in empty directories to preserve structure
"""

from pathlib import Path

def create_gitkeep_files():
    """Create .gitkeep in data directories"""
    
    project_root = Path(__file__).parent
    
    directories = [
        "data/fluent_raw",
        "data/fluent_processed",
        "data/deeponet_dataset",
        "data/nppad/operation_csv_data",
        "data/nppad/dose_csv_data",
        "fluent/geometry",
        "fluent/mesh",
        "results/models",
        "results/plots",
        "results/metrics",
        "results/predictions",
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        gitkeep = full_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"✓ Created {gitkeep}")


if __name__ == "__main__":
    create_gitkeep_files()
    print("\n✓ Directory structure preserved with .gitkeep files")
