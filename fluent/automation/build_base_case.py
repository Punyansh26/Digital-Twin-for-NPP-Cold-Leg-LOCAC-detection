"""
build_base_case.py
==================
Generates the AP1000 Cold-Leg base case file:  cold_leg_base.cas.h5

Pipeline:
  1. Build geometry + mesh  → via gmsh (pure Python, no CAD software needed)
  2. Write mesh to          → cold_leg_mesh.msh   (Fluent-readable ASCII MSH)
  3. Launch Fluent in batch → reads mesh, sets up physics, saves cold_leg_base.cas.h5

Requirements:
  pip install gmsh ansys-fluent-core

Geometry (AP1000 Cold-Leg):
  - Pipe inner diameter   D  = 0.787 m  (~31 in nominal)
  - Straight inlet section   = 5 D
  - 90-degree elbow, radius  = 1.5 D
  - Straight outlet section  = 5 D
  - "Break" patch on elbow outer wall (separate named surface for BC)

Run:
  python fluent/automation/build_base_case.py
  python fluent/automation/build_base_case.py --fluent-path "C:/Program Files/ANSYS Inc/v232/fluent/ntbin/win64/fluent.exe"
"""

import math
import subprocess
import argparse
import sys
import os
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
FLUENT_DIR   = PROJECT_ROOT / "fluent"
JOURNALS_DIR = FLUENT_DIR / "journals"
MESH_FILE    = FLUENT_DIR / "cold_leg_mesh.msh"
CASE_FILE    = FLUENT_DIR / "cold_leg_base.cas.h5"

# ── Geometry constants (match config.yaml / generate_simulations.py) ─────────
D              = 0.787          # pipe inner diameter [m]  AP1000 ~31-inch ID
R_ELBOW        = 1.5 * D        # elbow centreline radius
L_INLET        = 5.0 * D        # straight inlet length
L_OUTLET       = 5.0 * D        # straight outlet length
MESH_SIZE_FINE = D / 8          # element size near walls
MESH_SIZE_COARSE = D / 4        # element size in bulk

# ── Fluid base properties (water at ~290 °C, 15.5 MPa) ──────────────────────
DENSITY             = 732.0     # kg/m³
SPECIFIC_HEAT       = 5500.0    # J/kg-K
THERMAL_CONDUCTIVITY= 0.558     # W/m-K
VISCOSITY           = 9.0e-5    # kg/m-s
BASE_VELOCITY       = 5.0       # m/s  (nominal, overridden per simulation)
BASE_TEMPERATURE    = 563.15    # K    (290 °C)
OUTLET_PRESSURE     = 15500000  # Pa   (15.5 MPa)


# ============================================================
#  STEP 1 – Build geometry and mesh with gmsh
# ============================================================

def build_mesh():
    """
    Build a 3-D cold-leg pipe (straight + 90° elbow + straight) using gmsh
    and export a Fluent-compatible ASCII MSH file.

    Named surfaces:
        inlet   – velocity inlet
        outlet  – pressure outlet
        break   – small patch on elbow outer wall (break location)
        wall    – all remaining pipe walls
    """
    try:
        import gmsh
    except ImportError:
        print("ERROR: gmsh Python package not found.")
        print("Install it with:  pip install gmsh")
        sys.exit(1)

    print("\n[1/3] Building geometry and mesh with gmsh ...")
    gmsh.initialize()
    gmsh.model.add("cold_leg")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm",   5)   # Delaunay
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # Frontal-Delaunay

    # ── Characteristic lengths ────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE_FINE)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE_COARSE)

    geo = gmsh.model.occ          # use OpenCASCADE kernel for boolean ops

    R = D / 2.0                   # pipe radius

    # ── Three pipe segments via revolution / sweep ───────────────────────────
    # Segment 1: straight inlet  (along +X axis)
    disk1 = geo.addDisk(0, 0, 0,  R, R)                  # inlet face  @ origin
    geo.synchronize()
    vol1, _ = geo.extrude([(2, disk1)], L_INLET, 0, 0)   # extrude along X

    # Segment 2: 90° elbow
    # Elbow centre is at (L_INLET + R_ELBOW, 0, 0), revolving around Z axis
    # inlet disk of elbow = outlet disk of segment 1  → we reuse its face
    # after extrude vol1[1] is the outlet face tag
    elbow_inlet_face = vol1[1][1]                         # face tag
    cx = L_INLET + R_ELBOW                                # elbow centre X
    cy = 0.0
    vol2, _ = geo.revolve(
        [(2, elbow_inlet_face)],
        cx, cy, 0,          # axis point
        0,  0,  1,          # axis direction (+Z)
        -math.pi / 2        # -90° → rotates outlet to point in +Y direction
    )

    # Segment 3: straight outlet  (along +Y from elbow exit)
    # elbow outlet face = vol2[1]
    elbow_outlet_face = vol2[1][1]
    vol3, _ = geo.extrude([(2, elbow_outlet_face)], 0, L_OUTLET, 0)

    # ── Fuse all three volumes into one ──────────────────────────────────────
    geo.synchronize()
    fused, _ = geo.fuse(
        [(3, vol1[0][1])],
        [(3, vol2[0][1]), (3, vol3[0][1])]
    )
    geo.synchronize()

    # ── Identify Named Surfaces ──────────────────────────────────────────────
    # After fusion gmsh renumbers; use bounding-box queries to find faces.
    all_surfaces = gmsh.model.getBoundary(fused, oriented=False)
    surf_tags = [s[1] for s in all_surfaces]

    def face_centroid(tag):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        return ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)

    inlet_tag  = min(surf_tags, key=lambda t: face_centroid(t)[0])   # most -X
    outlet_tag = min(surf_tags, key=lambda t: -face_centroid(t)[1])  # most +Y

    # Break patch: small area on elbow outer wall (max X in elbow zone)
    #   We approximate by picking the face whose centroid is closest to
    #   (L_INLET + R_ELBOW + R, 0, 0) — the outermost elbow wall point.
    elbow_outer_target = (L_INLET + R_ELBOW + R, 0.0, 0.0)
    def dist(tag):
        cx_, cy_, cz_ = face_centroid(tag)
        return math.sqrt((cx_ - elbow_outer_target[0])**2 +
                         (cy_ - elbow_outer_target[1])**2 +
                         (cz_ - elbow_outer_target[2])**2)
    
    wall_candidates = [t for t in surf_tags if t not in (inlet_tag, outlet_tag)]
    break_tag = min(wall_candidates, key=dist)
    wall_tags = [t for t in wall_candidates if t != break_tag]

    # ── Assign Physical Groups (Fluent Named Selections) ─────────────────────
    gmsh.model.addPhysicalGroup(2, [inlet_tag],  name="inlet")
    gmsh.model.addPhysicalGroup(2, [outlet_tag], name="outlet")
    gmsh.model.addPhysicalGroup(2, [break_tag],  name="break")
    gmsh.model.addPhysicalGroup(2, wall_tags,    name="wall")
    gmsh.model.addPhysicalGroup(3, [t[1] for t in fused], name="fluid")

    # ── Generate 3-D mesh ────────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    # ── Write Fluent-format MSH ──────────────────────────────────────────────
    MESH_FILE.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(MESH_FILE))          # gmsh auto-selects format from ext
    print(f"    ✓ Mesh written → {MESH_FILE}")

    gmsh.finalize()
    return str(MESH_FILE)


# ============================================================
#  STEP 2 – Write the Fluent journal that reads the mesh,
#            sets up physics and saves cold_leg_base.cas.h5
# ============================================================

SETUP_JOURNAL = JOURNALS_DIR / "create_base_case.jou"

def write_setup_journal(mesh_file: str, output_case: str):
    """Write the Fluent TUI journal for full physics setup."""
    print("\n[2/3] Writing Fluent setup journal ...")

    content = f"""; ============================================================
; create_base_case.jou
; AP1000 Cold-Leg  –  base case setup journal
; Auto-generated by build_base_case.py
; ============================================================

/file/set-batch-options no yes

; ── 1. Read mesh ─────────────────────────────────────────────
/file/read-case "{mesh_file}"

; ── 2. Solver: pressure-based, steady ────────────────────────
/define/models/solver/pressure-based yes
/define/models/steady yes

; ── 3. Energy equation ───────────────────────────────────────
/define/models/energy yes

; ── 4. Turbulence: Realizable k-epsilon + enhanced wall ──────
/define/models/viscous/ke-realizable yes
/define/models/viscous/near-wall-treatment enhanced-wall-treatment

; ── 5. Material: liquid water (pressurised) ──────────────────
/define/materials/change-create \\
    water water \\
    yes constant {DENSITY} \\
    no no \\
    yes constant {SPECIFIC_HEAT} \\
    no no \\
    yes constant {THERMAL_CONDUCTIVITY} \\
    no no \\
    yes constant {VISCOSITY} \\
    no no no

; ── 6. Cell-zone conditions: fluid ───────────────────────────
/define/boundary-conditions/fluid fluid no 0 no 0 no no 1 no yes water () no 0 no 0

; ── 7. Boundary conditions ───────────────────────────────────
; Inlet: velocity inlet  (nominal operating values;
;   actual values are overridden per simulation run)
/define/boundary-conditions/velocity-inlet \\
    inlet yes no {BASE_VELOCITY} no 0 no 0 \\
    no {BASE_TEMPERATURE} no no yes 5 10

; Outlet: pressure outlet
/define/boundary-conditions/pressure-outlet \\
    outlet yes no {OUTLET_PRESSURE} \\
    no {BASE_TEMPERATURE} no yes no no yes 5 10

; Break patch: pressure outlet (represents break opening)
;   Will be activated/deactivated per simulation run
/define/boundary-conditions/pressure-outlet \\
    break yes no 101325 \\
    no {BASE_TEMPERATURE} no yes no no yes 5 10

; Wall: adiabatic no-slip
/define/boundary-conditions/wall wall 0 no 0 no no no 0 no 0.5 no 1

; ── 8. Solution methods (SIMPLE, second-order) ───────────────
/solve/set/p-v-coupling 20
/solve/set/discretization-scheme pressure 12
/solve/set/discretization-scheme mom 1
/solve/set/discretization-scheme k 1
/solve/set/discretization-scheme epsilon 1
/solve/set/discretization-scheme energy 1

; Under-relaxation factors
/solve/set/under-relaxation pressure 0.3
/solve/set/under-relaxation mom 0.7
/solve/set/under-relaxation k 0.8
/solve/set/under-relaxation epsilon 0.8
/solve/set/under-relaxation energy 1.0
/solve/set/under-relaxation density 1.0

; ── 9. Convergence monitors ──────────────────────────────────
/solve/monitors/residual/convergence-criteria 1e-4 1e-4 1e-4 1e-4 1e-4 1e-6

; ── 10. Hybrid initialisation + 50 steady iterations ─────────
;   (creates a stable starting field for the parameter sweep)
/solve/initialize/hyb-initialization
/solve/iterate 50

; ── 11. Save base case (NO data – simulations write their own) ─
/file/write-case "{output_case}" yes

/exit yes
"""
    JOURNALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETUP_JOURNAL, 'w') as f:
        f.write(content)
    print(f"    ✓ Journal written → {SETUP_JOURNAL}")


# ============================================================
#  STEP 3 – Launch Fluent in batch mode to run the journal
# ============================================================

def run_fluent(fluent_exe: str):
    """Launch ANSYS Fluent in 3-D batch mode with the setup journal."""
    print(f"\n[3/3] Launching Fluent: {fluent_exe}")
    print("      This will take a few minutes (mesh import + 50 iterations) ...")

    cmd = [
        fluent_exe,
        "3d",          # 3-D solver
        "-g",          # no GUI (batch)
        "-t4",         # use 4 CPU cores (adjust as needed)
        "-i", str(SETUP_JOURNAL),
    ]

    log_file = FLUENT_DIR / "build_base_case.log"
    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(FLUENT_DIR)
        )

    if result.returncode == 0 and CASE_FILE.exists():
        print(f"\n{'='*60}")
        print(f"  ✓  cold_leg_base.cas.h5 created successfully!")
        print(f"     Location: {CASE_FILE}")
        print(f"  ✓  Fluent log: {log_file}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  ✗  Fluent exited with code {result.returncode}")
        print(f"     Check the log for details: {log_file}")
        print(f"{'='*60}\n")
        sys.exit(1)


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build cold_leg_base.cas.h5 for the AP1000 Digital Twin"
    )
    parser.add_argument(
        "--fluent-path",
        default="fluent",
        help=(
            "Full path to the Fluent executable. "
            "Default 'fluent' assumes it is on PATH. "
            "Example (Windows): "
            r'"C:/Program Files/ANSYS Inc/v232/fluent/ntbin/win64/fluent.exe"'
        )
    )
    parser.add_argument(
        "--cores", type=int, default=4,
        help="Number of CPU cores for Fluent (default: 4)"
    )
    parser.add_argument(
        "--skip-mesh", action="store_true",
        help=f"Skip gmsh step (use existing mesh at {MESH_FILE})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AP1000 Cold-Leg  –  Base Case Builder")
    print("  Target file: cold_leg_base.cas.h5")
    print("=" * 60)

    # Step 1: mesh
    if not args.skip_mesh:
        build_mesh()
    else:
        if not MESH_FILE.exists():
            print(f"ERROR: --skip-mesh specified but {MESH_FILE} not found.")
            sys.exit(1)
        print(f"[1/3] Skipping mesh (using existing: {MESH_FILE})")

    # Step 2: journal
    write_setup_journal(str(MESH_FILE), str(CASE_FILE))

    # Step 3: run Fluent
    run_fluent(args.fluent_path)


if __name__ == "__main__":
    main()
