"""
09_validate_dynamic.py — Velocity-sweep validation: compare MuJoCo CMTU vs OpenSim
force output across multiple velocities.

For each muscle, compares force at 5 velocities across the MTU length range.
Computes per-muscle dynamic error metrics and generates overlay plots.

Usage:
    python scripts/09_validate_dynamic.py [data_dir] [fitted_csv] [out_dir]

    data_dir:   Directory with OpenSim extracted data (from 01_extract_opensim_data.py
                with multi-velocity extraction)
    fitted_csv: CSV from 02_fit_mujoco_params.py
    out_dir:    Output directory for plots and error reports
"""

import sys
import os
import csv
import numpy as np

if sys.version_info < (3, 11):
    print(f"ERROR: Requires Python 3.11+, using {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

import mujoco
import matplotlib.pyplot as plt

# Import shared functions from 02
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
_mod = SourceFileLoader("fit_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_fit_mujoco_params.py")).load_module()

CompliantTendonParams = _mod.CompliantTendonParams
create_model = _mod.create_model
compute_forces_at_velocity = _mod.compute_forces_at_velocity
load_length_force_sim = _mod.load_length_force_sim


def load_fitted_params_csv(csv_path):
    """Load fitted parameters from CSV."""
    params = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            muscle = row["muscle"]
            if not muscle:
                continue
            params[muscle] = [
                float(row["F_max"]), float(row["l_opt"]), float(row["l_slack"]),
                float(row["v_max"]), float(row["W"]), float(row["C"]),
                float(row["N"]), float(row["K"]), float(row["E_REF"]),
            ]
    return params


def validate_muscle_dynamic(muscle_name, target_data, fitted_params, out_dir):
    """
    Compare MuJoCo CMTU vs OpenSim at all available velocities.
    Returns per-velocity RMS errors.
    """
    F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF = fitted_params
    ref_f_max = target_data["f_max"]

    cp = CompliantTendonParams(F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF)
    model = create_model(cp)
    data = mujoco.MjData(model)

    mtu_lengths = target_data["mtu_lengths"]
    norm_velocities = target_data["norm_velocities"]
    force_matrix = target_data["force_matrix"]  # (n_lengths, n_velocities)

    has_passive = target_data.get("passive_force_matrix") is not None
    if has_passive:
        passive_matrix = target_data["passive_force_matrix"]

    errors = {}  # v_norm -> {"rms_active": ..., "rms_passive": ..., "max_active": ...}

    # Setup figure: one subplot per velocity
    n_vel = len(norm_velocities)
    fig, axes = plt.subplots(1, n_vel, figsize=(5 * n_vel, 4), squeeze=False)
    axes = axes[0]

    for j, v_norm in enumerate(norm_velocities):
        v_phy = v_norm * v_max
        ax = axes[j]

        # Target (OpenSim)
        f_target_active = force_matrix[:, j]

        # Simulated (MuJoCo CMTU)
        f_sim_active = compute_forces_at_velocity(model, data, v_phy, mtu_lengths, activation=1.0)

        # Filter valid points
        valid = np.isfinite(f_sim_active) & np.isfinite(f_target_active)

        if np.sum(valid) > 0:
            diff_active = f_sim_active[valid] - f_target_active[valid]
            rms_active = np.sqrt(np.mean(diff_active ** 2))
            max_active = np.max(np.abs(diff_active))
            nrms_active = rms_active / ref_f_max * 100  # as % of F_max
        else:
            rms_active = np.nan
            max_active = np.nan
            nrms_active = np.nan

        rms_passive = np.nan
        if has_passive:
            f_target_passive = passive_matrix[:, j]
            f_sim_passive = compute_forces_at_velocity(model, data, v_phy, mtu_lengths, activation=0.0)
            valid_p = np.isfinite(f_sim_passive) & np.isfinite(f_target_passive)
            if np.sum(valid_p) > 0:
                diff_passive = f_sim_passive[valid_p] - f_target_passive[valid_p]
                rms_passive = np.sqrt(np.mean(diff_passive ** 2))
                ax.plot(mtu_lengths, f_target_passive, "b.", alpha=0.4, markersize=2, label="Passive (OpenSim)")
                ax.plot(mtu_lengths, f_sim_passive, "b--", linewidth=1, label="Passive (MuJoCo)")

        errors[v_norm] = {
            "rms_active": rms_active,
            "nrms_active_pct": nrms_active,
            "max_active": max_active,
            "rms_passive": rms_passive,
        }

        # Plot
        ax.plot(mtu_lengths, f_target_active, "k.", markersize=3, label="Active (OpenSim)")
        ax.plot(mtu_lengths, f_sim_active, "r-", linewidth=1, label="Active (MuJoCo)")
        ax.set_title(f"v_norm={v_norm:.2f}\nRMS={nrms_active:.1f}% F_max")
        ax.set_xlabel("MTU length (m)")
        if j == 0:
            ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize="x-small")
    fig.suptitle(f"{muscle_name} — Dynamic Validation", fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{muscle_name}_dynamic_validation.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    return errors


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "osim_muscle_data/Rajagopal"
    fitted_csv = sys.argv[2] if len(sys.argv) > 2 else "mujoco_muscle_data/Rajagopal/fitted_params_length_only.csv"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs_dynamic_validation"

    params_csv = os.path.join(data_dir, "all_muscle_parameters.csv")

    # Load fitted parameters
    fitted_params = load_fitted_params_csv(fitted_csv)
    print(f"Loaded {len(fitted_params)} fitted muscles from {fitted_csv}")

    # Discover available muscles
    files = [f for f in os.listdir(data_dir) if f.endswith("_sim_total.csv")]
    muscles = sorted(f.replace("_sim_total.csv", "") for f in files)

    all_errors = []

    for mname in muscles:
        if mname not in fitted_params:
            print(f"Skipping {mname} (no fitted params)")
            continue

        print(f"\nValidating {mname}...")
        try:
            target_data = load_length_force_sim(mname, params_csv, data_dir)
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue

        # Check if multi-velocity data exists
        if len(target_data["norm_velocities"]) < 2:
            print(f"  Only v=0 data available — re-run 01_extract_opensim_data.py with multi-velocity")
            continue

        errors = validate_muscle_dynamic(mname, target_data, fitted_params[mname], out_dir)

        for v_norm, err in errors.items():
            all_errors.append({
                "muscle": mname,
                "v_norm": v_norm,
                "rms_active_N": err["rms_active"],
                "nrms_active_pct_fmax": err["nrms_active_pct"],
                "max_active_N": err["max_active"],
                "rms_passive_N": err["rms_passive"],
            })

    # Save error report
    if all_errors:
        report_path = os.path.join(out_dir, "dynamic_validation_errors.csv")
        os.makedirs(out_dir, exist_ok=True)
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_errors[0].keys())
            writer.writeheader()
            writer.writerows(all_errors)
        print(f"\nError report saved: {report_path}")

        # Summary: identify worst muscles at v≠0
        v_nonzero = [e for e in all_errors if abs(e["v_norm"]) > 0.01]
        if v_nonzero:
            worst = sorted(v_nonzero, key=lambda e: e["nrms_active_pct_fmax"]
                           if not np.isnan(e["nrms_active_pct_fmax"]) else 0, reverse=True)
            print(f"\nTop 10 worst dynamic errors (v≠0):")
            print(f"  {'Muscle':<20s} {'v_norm':>6s} {'NRMS%':>8s} {'RMS(N)':>8s}")
            for e in worst[:10]:
                print(f"  {e['muscle']:<20s} {e['v_norm']:>6.2f} {e['nrms_active_pct_fmax']:>8.1f} {e['rms_active_N']:>8.1f}")


if __name__ == "__main__":
    main()
