"""
Merge fitted parameters from gait14dof (11 lumped muscles) and Rajagopal (40 muscles).

Strategy:
  - Direct 1:1 matches  → use all gait14dof params (preferred: validated 2-D planar model)
  - 1:many splits       → Rajagopal geometry (F_max, l_opt, l_slack, v_max)
                          + gait14dof curve shape (W, C, N, K, E_REF) from parent muscle
  - Rajagopal-only      → keep Rajagopal params unchanged

Outputs:
    mujoco_muscle_data/merged_fitted_params.csv
Then applies it to the compliant XML via script 03.
"""

import csv
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# MyoSim name -> gait14dof name  (direct 1:1 functional equivalents)
DIRECT_MAP = {
    "soleus_r": "soleus_r",
    "recfem_r": "rect_fem_r",
    "tibant_r": "tib_ant_r",
    "bfsh_r":   "bifemsh_r",
}

# MyoSim name -> gait14dof parent muscle  (sub-muscle gets parent's curve shape)
SHAPE_MAP = {
    # Gastrocnemius split
    "gasmed_r":     "gastroc_r",
    "gaslat_r":     "gastroc_r",
    # Hamstrings split (long-head components; bfsh_r handled via DIRECT_MAP)
    "bflh_r":       "hamstrings_r",
    "semimem_r":    "hamstrings_r",
    "semiten_r":    "hamstrings_r",
    # Vasti split
    "vasint_r":     "vasti_r",
    "vaslat_r":     "vasti_r",
    "vasmed_r":     "vasti_r",
    # Gluteus maximus split
    "glmax1_r":     "glut_max_r",
    "glmax2_r":     "glut_max_r",
    "glmax3_r":     "glut_max_r",
    # Iliopsoas split
    "iliacus_r":    "iliopsoas_r",
    "psoas_r":      "iliopsoas_r",
    # Hip abductors
    "glmed1_r":     "abd_r",
    "glmed2_r":     "abd_r",
    "glmed3_r":     "abd_r",
    # Hip adductors
    "addbrev_r":    "add_r",
    "addlong_r":    "add_r",
    "addmagDist_r": "add_r",
    "addmagIsch_r": "add_r",
    "addmagMid_r":  "add_r",
    "addmagProx_r": "add_r",
}

FIELDS = ["muscle", "F_max", "l_opt", "l_slack", "v_max", "W", "C", "N", "K", "E_REF"]
GEO_FIELDS   = ["F_max", "l_opt", "l_slack", "v_max"]
SHAPE_FIELDS = ["W", "C", "N", "K", "E_REF"]


def load_csv(path: Path) -> dict:
    """Return {muscle_name: {field: float}} from a fitted_params CSV."""
    data = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["muscle"].strip()
            if name:
                data[name] = {k: float(row[k]) for k in FIELDS[1:]}
    return data


def merge(gait14dof: dict, rajagopal: dict) -> list[dict]:
    rows = []
    for muscle in rajagopal:
        row = {"muscle": muscle}

        if muscle in DIRECT_MAP:
            # Full substitution from gait14dof
            src = gait14dof[DIRECT_MAP[muscle]]
            for f in FIELDS[1:]:
                row[f] = src[f]
            label = f"gait14dof ({DIRECT_MAP[muscle]})"

        elif muscle in SHAPE_MAP:
            # Rajagopal geometry + gait14dof curve shape
            parent = SHAPE_MAP[muscle]
            for f in GEO_FIELDS:
                row[f] = rajagopal[muscle][f]
            for f in SHAPE_FIELDS:
                row[f] = gait14dof[parent][f]
            label = f"Rajagopal geo + gait14dof shape ({parent})"

        else:
            # Pure Rajagopal
            for f in FIELDS[1:]:
                row[f] = rajagopal[muscle][f]
            label = "Rajagopal"

        print(f"  {muscle:<20} <- {label}")
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved merged params: {path}")


def main():
    root = Path(__file__).resolve().parent.parent

    gait14_csv = root / "mujoco_muscle_data/gait14dof22musc_planar_20170320/fitted_params_length_only.csv"
    raja_csv   = root / "mujoco_muscle_data/Rajagopal/fitted_params_length_only.csv"
    merged_csv = root / "mujoco_muscle_data/merged_fitted_params.csv"
    xml_path   = root / "myosim_convert/myo_sim/leg/assets/myolegs_muscle_compliant.xml"

    print("Loading gait14dof params ...")
    gait14dof = load_csv(gait14_csv)
    print(f"  {len(gait14dof)} muscles: {', '.join(sorted(gait14dof))}\n")

    print("Loading Rajagopal params ...")
    rajagopal = load_csv(raja_csv)
    print(f"  {len(rajagopal)} muscles\n")

    print("Merging ...")
    rows = merge(gait14dof, rajagopal)

    write_csv(rows, merged_csv)

    print("\nApplying to XML via script 03 ...")
    result = subprocess.run(
        [sys.executable, str(root / "scripts/03_apply_fitted_params.py"),
         str(xml_path), str(merged_csv), str(xml_path)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
