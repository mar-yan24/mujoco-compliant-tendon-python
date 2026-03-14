"""
08_hybrid_model_builder.py — Build a hybrid rigid/compliant muscle XML.

For muscles with short tendons (l_slack/l_opt < 1.0), use standard MuJoCo
gaintype="muscle" (rigid tendon) to avoid spurious compliance and Newton solver
overhead. For muscles with long tendons, use gaintype="compliant_mtu" with
fitted E_REF.

Usage:
    python scripts/08_hybrid_model_builder.py \
        [base_xml] [fitted_csv] [out_xml]

    base_xml:   MuJoCo muscle XML (e.g., myolegs_muscle_rigid.xml)
    fitted_csv: CSV from 02_fit_mujoco_params.py with 9 fitted parameters
    out_xml:    Output hybrid XML path
"""

import sys
import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

if sys.version_info < (3, 11):
    print(f"ERROR: Requires Python 3.11+, using {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)


# Threshold: muscles with l_slack/l_opt below this use rigid tendon
COMPLIANT_RATIO_THRESHOLD = 1.0


def load_fitted_params(csv_path):
    """Load fitted parameters from CSV, return dict of muscle_name -> param dict."""
    param_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            muscle = row["muscle"]
            if not muscle:
                continue
            param_map[muscle] = {
                "F_max": float(row["F_max"]),
                "l_opt": float(row["l_opt"]),
                "l_slack": float(row["l_slack"]),
                "v_max": float(row["v_max"]),
                "W": float(row["W"]),
                "C": float(row["C"]),
                "N": float(row["N"]),
                "K": float(row["K"]),
                "E_REF": float(row["E_REF"]),
            }
    return param_map


def counterpart_name(name):
    """Map _l <-> _r for bilateral mirroring."""
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return None


def classify_muscle(params):
    """Return 'rigid' or 'compliant' based on tendon compliance ratio."""
    l_opt = params["l_opt"]
    l_slack = params["l_slack"]
    if l_opt <= 0:
        return "rigid"
    ratio = l_slack / l_opt
    return "compliant" if ratio >= COMPLIANT_RATIO_THRESHOLD else "rigid"


def format_gainprm_compliant(params):
    """Format 9 compliant_mtu parameters as a space-separated string."""
    keys = ["F_max", "l_opt", "l_slack", "v_max", "W", "C", "N", "K", "E_REF"]
    return " ".join(f"{params[k]:.12g}" for k in keys)


def format_gainprm_rigid(params):
    """Format standard MuJoCo muscle gainprm (timeconst1 timeconst2 ... range[2])."""
    # For rigid tendon, use MuJoCo's built-in muscle model.
    # gainprm for gaintype="muscle" is:
    #   range[0] range[1] force scale lmin lmax vmax fpmax fvmax
    # We set reasonable defaults, scaling F_max via the tendon's force attribute.
    # Actually, for gaintype="muscle", the gain is just the muscle force scaling.
    # The simplest approach: set gainprm="1" and let MuJoCo use its defaults,
    # but override the force via the tendon or actuator force attribute.
    #
    # For a clean hybrid, we encode F_max as the gain and let MuJoCo handle
    # the default Hill model internally.
    return f"{params['F_max']:.12g}"


def build_hybrid_xml(base_xml_path, param_map, out_xml_path):
    """Build hybrid XML: compliant for long-tendon muscles, rigid for short-tendon."""
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    stats = {"compliant": 0, "rigid": 0, "missing": 0, "mirrored": 0}

    for general in root.findall(".//general"):
        name = general.attrib.get("name", "")
        params = None

        if name in param_map:
            params = param_map[name]
        else:
            mirror = counterpart_name(name)
            if mirror and mirror in param_map:
                params = param_map[mirror]
                stats["mirrored"] += 1

        if params is None:
            stats["missing"] += 1
            continue

        muscle_type = classify_muscle(params)
        ratio = params["l_slack"] / params["l_opt"] if params["l_opt"] > 0 else 0

        if muscle_type == "compliant":
            general.set("gainprm", format_gainprm_compliant(params))
            general.set("gaintype", "compliant_mtu")
            general.set("biasprm", "0")
            # Ensure dyntype is set for compliant muscles
            if "dyntype" not in general.attrib or general.attrib["dyntype"] != "compliant_mtu":
                general.set("dyntype", "muscle")
            stats["compliant"] += 1
        else:
            # Rigid tendon: use MuJoCo's built-in muscle model
            general.set("gaintype", "muscle")
            general.set("biastype", "muscle")
            # Remove compliant-specific attributes
            if "biasprm" in general.attrib:
                general.set("biasprm", "0")
            # gainprm for built-in muscle: just needs reasonable range
            # MuJoCo muscle defaults handle the Hill model internally
            general.set("gainprm", "1 0 0 0 0 0 0 0 0 0")
            general.set("dyntype", "muscle")
            stats["rigid"] += 1

        print(f"  {name}: {muscle_type} (ratio={ratio:.2f}, E_REF={params['E_REF']:.4f})")

    tree.write(out_xml_path, encoding="utf-8", xml_declaration=False)
    return stats


def main():
    repo_root = Path(__file__).resolve().parent.parent

    base_xml = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else repo_root / "myosim_convert" / "myo_sim" / "leg" / "assets" / "myolegs_muscle_rigid.xml"
    )
    fitted_csv = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else repo_root / "mujoco_muscle_data" / "fitted_params_length_only.csv"
    )
    out_xml = (
        Path(sys.argv[3]) if len(sys.argv) > 3
        else repo_root / "myosim_convert" / "myo_sim" / "leg" / "assets" / "myolegs_muscle_hybrid.xml"
    )

    print(f"Base XML:    {base_xml}")
    print(f"Fitted CSV:  {fitted_csv}")
    print(f"Output XML:  {out_xml}")
    print(f"Compliant threshold: l_slack/l_opt >= {COMPLIANT_RATIO_THRESHOLD}")
    print()

    param_map = load_fitted_params(fitted_csv)
    print(f"Loaded {len(param_map)} muscles from CSV\n")

    stats = build_hybrid_xml(base_xml, param_map, out_xml)

    print(f"\nHybrid model built: {out_xml}")
    print(f"  Compliant (long tendon): {stats['compliant']}")
    print(f"  Rigid (short tendon):    {stats['rigid']}")
    print(f"  Mirrored L/R:            {stats['mirrored']}")
    print(f"  Missing from CSV:        {stats['missing']}")


if __name__ == "__main__":
    main()
