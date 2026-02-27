"""
Refit glut_max_r and rect_fem_r with improved bounds and passive force weighting.

Changes from 02_fit_mujoco_params.py:
  1. C bounds tightened to [-5, -0.5] (prevent extreme values)
  2. N, K, E_REF unlocked (allows passive force curve fitting)
  3. Passive residuals weighted 2x (prioritize passive force matching)

Usage:
    py -3.11 scripts/02b_refit_problem_muscles.py <data_dir> <out_dir>
"""

import sys

if sys.version_info < (3, 11):
    print(f"ERROR: Requires Python 3.11+, using {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import least_squares
import time

# Import shared classes/functions from 02
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
_mod = SourceFileLoader("fit_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_fit_mujoco_params.py")).load_module()

CompliantTendonParams = _mod.CompliantTendonParams
create_model = _mod.create_model
compute_forces_at_velocity = _mod.compute_forces_at_velocity
load_length_force_sim = _mod.load_length_force_sim
get_fitting_range = _mod.get_fitting_range
plot_fitting_range_on_ax = _mod.plot_fitting_range_on_ax


PASSIVE_WEIGHT = 2.0  # Weight multiplier for passive force residuals


def objective_improved(x, target_data, verbose=0):
    """Objective with passive force weighting."""
    F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF = x

    params = [F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF]
    for p in params:
        if np.isnan(p) or np.isinf(p):
            raise ValueError(f"Invalid parameter: {p}")
    if F_max <= 0 or l_opt <= 0 or l_slack <= 0 or v_max <= 0:
        raise ValueError("Non-positive physical parameter")

    ref_f_max = target_data['f_max']
    mtu_lengths = target_data['mtu_lengths']
    norm_velocities = target_data['norm_velocities']
    force_matrix = target_data['force_matrix']

    min_len, max_len = get_fitting_range(target_data)
    range_mask = (mtu_lengths >= min_len) & (mtu_lengths <= max_len)
    if not np.any(range_mask):
        range_mask = np.ones_like(mtu_lengths, dtype=bool)

    target_l_phys = mtu_lengths[range_mask]
    target_force_subset = force_matrix[range_mask, :]
    v0_idx = np.argmin(np.abs(norm_velocities))

    cp = CompliantTendonParams(F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF)
    model = create_model(cp)
    data = mujoco.MjData(model)

    f_target_active = target_force_subset[:, v0_idx]
    f_sim_active = compute_forces_at_velocity(model, data, 0.0, target_l_phys, activation=1.0)

    if target_data.get('passive_force_matrix') is not None:
        passive_subset = target_data['passive_force_matrix'][range_mask, :]
        f_target_passive = passive_subset[:, v0_idx]
        f_sim_passive = compute_forces_at_velocity(model, data, 0.0, target_l_phys, activation=0.0)
    else:
        f_target_passive = np.zeros_like(f_target_active)
        f_sim_passive = np.zeros_like(f_sim_active)

    residuals_active = f_sim_active - f_target_active
    residuals_passive = (f_sim_passive - f_target_passive) * PASSIVE_WEIGHT

    # Validity filter
    valid_mask = np.isfinite(f_sim_active) & (np.abs(f_sim_active) <= 5.0 * ref_f_max)
    valid_mask &= np.isfinite(f_sim_passive) & (np.abs(f_sim_passive) <= 5.0 * ref_f_max)

    if f_sim_active.size > 1:
        jump_thresh = 0.1 * ref_f_max
        jumps = np.abs(np.diff(f_sim_active)) > jump_thresh
        jumps |= np.abs(np.diff(f_sim_passive)) > jump_thresh
        bad_indices = np.where(jumps)[0] + 1
        valid_mask[bad_indices] = False

    penalty = 5.0 * ref_f_max
    residuals_active[~np.isfinite(residuals_active)] = penalty
    residuals_active[~valid_mask] = np.sign(residuals_active[~valid_mask]) * penalty
    residuals_passive[~np.isfinite(residuals_passive)] = penalty
    residuals_passive[~valid_mask] = np.sign(residuals_passive[~valid_mask]) * penalty

    all_residuals = np.concatenate([residuals_active, residuals_passive])

    if verbose >= 1:
        mse = np.mean(all_residuals**2)
        print(f"  MSE={mse:.1f} | F={F_max:.1f} lo={l_opt:.4f} ls={l_slack:.4f} "
              f"W={W:.3f} C={C:.3f} N={N:.3f} K={K:.3f} E={E_REF:.4f}")

    return all_residuals


def fit_muscle_improved(muscle_name, data_dir, params_csv, out_dir, verbose=0):
    """Fit with tightened C bounds, unlocked N/K/E_REF, passive weighting."""
    print(f"\n{'='*60}")
    print(f"REFIT: {muscle_name} (improved bounds + passive weighting)")
    print(f"{'='*60}")

    target_data = load_length_force_sim(muscle_name, params_csv, data_dir)

    base_F = target_data['f_max']
    base_L_opt = target_data['l_opt']
    base_L_slack = target_data['l_slack']
    base_V_max = target_data['v_max']

    x0 = [
        base_F,
        base_L_opt,
        base_L_slack,
        base_V_max,
        0.56, -2.995732274, 1.5, 5.0, 0.04
    ]

    bounds = [
        (base_F * 0.5, base_F * 1.5),              # F_max
        (base_L_opt * 0.8, base_L_opt * 1.2),       # l_opt (slightly wider)
        (base_L_slack * 0.8, base_L_slack * 1.2),    # l_slack (slightly wider)
        (base_V_max, base_V_max + 1e-7),             # v_max fixed

        (0.1, 3.0),                                   # W (tightened upper)
        (-5.0, -0.5),                                 # C (TIGHTENED from [-100, -0.01])
        (0.5, 3.0),                                   # N (UNLOCKED from 1.5 fixed)
        (2.0, 10.0),                                  # K (UNLOCKED from 5.0 fixed)
        (0.01, 0.15),                                 # E_REF (UNLOCKED from 0.04 fixed)
    ]

    print(f"  Initial: F={x0[0]:.1f} lo={x0[1]:.4f} ls={x0[2]:.4f}")
    print(f"  C bounds: [{bounds[5][0]}, {bounds[5][1]}] (was [-100, -0.01])")
    print(f"  N bounds: [{bounds[6][0]}, {bounds[6][1]}] (was fixed at 1.5)")
    print(f"  K bounds: [{bounds[7][0]}, {bounds[7][1]}] (was fixed at 5.0)")
    print(f"  E_REF bounds: [{bounds[8][0]}, {bounds[8][1]}] (was fixed at 0.04)")
    print(f"  Passive weight: {PASSIVE_WEIGHT}x")

    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]

    def obj(x):
        return objective_improved(x, target_data, verbose=verbose)

    t0 = time.time()
    res = least_squares(
        obj, x0,
        bounds=(lower, upper),
        max_nfev=2000,
        jac='3-point',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        verbose=2 if verbose >= 1 else 0,
    )
    elapsed = time.time() - t0

    print(f"\n  Finished in {elapsed:.1f}s — {res.message}")
    print(f"  Final MSE: {np.mean(res.fun**2):.1f}")
    names = ['F_max', 'l_opt', 'l_slack', 'v_max', 'W', 'C', 'N', 'K', 'E_REF']
    for n, v, v0 in zip(names, res.x, x0):
        print(f"    {n}: {v:.6f} (init: {v0:.6f})")

    # Plot
    plot_refit(res.x, x0, target_data, muscle_name, out_dir)

    return res.x


def plot_refit(best, initial, target_data, muscle_name, out_dir):
    """Plot fitted vs data with both active and passive."""
    F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF = best
    cp = CompliantTendonParams(F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF)
    model = create_model(cp)
    data = mujoco.MjData(model)

    mtu_lengths = target_data['mtu_lengths']
    dense = np.linspace(mtu_lengths.min(), mtu_lengths.max(), 80)

    f_sim_active = compute_forces_at_velocity(model, data, 0.0, dense, activation=1.0)
    f_sim_passive = compute_forces_at_velocity(model, data, 0.0, dense, activation=0.0)

    f_data_active = target_data['force_matrix'][:, 0]
    f_data_passive = target_data['passive_force_matrix'][:, 0] if target_data.get('passive_force_matrix') is not None else None

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    min_len, max_len = get_fitting_range(target_data)
    plot_fitting_range_on_ax(ax, min_len, max_len, label='Fitting Range')

    # Reference lines
    val_ref = target_data['l_opt'] + target_data['l_slack']
    ax.axvline(val_ref, color='gray', linestyle='--', label='Ref lo+ls', alpha=0.8)
    val_fit = l_opt + l_slack
    ax.axvline(val_fit, color='r', linestyle=':', label='Fit lo+ls', alpha=0.8)

    if f_data_passive is not None:
        ax.plot(mtu_lengths, f_data_passive, "b.", label="Passive (Data)", alpha=0.5)
        ax.plot(dense, f_sim_passive, "b--", label="Passive (Fit)", linewidth=2)

    ax.plot(mtu_lengths, f_data_active, "k.", label="Active (Data)")
    ax.plot(dense, f_sim_active, "r-", label="Active (Fit)", linewidth=2)

    ax.set_title(f"{muscle_name} (REFIT: C=[{-5},{-0.5}], N/K/E unlocked, passive 2x)")
    ax.set_xlabel("MTU length (m)")
    ax.set_ylabel("Force (N)")
    ax.legend(fontsize="small", loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    # Add parameter text
    F0, l0, ls0, v0, W0, C0, N0, K0, E0 = initial
    txt = f"Fitted: F={F_max:.1f} lo={l_opt:.4f} ls={l_slack:.4f} W={W:.3f} C={C:.3f} N={N:.3f} K={K:.3f} E={E_REF:.4f}\n"
    txt += f"Initial: F={F0:.1f} lo={l0:.4f} ls={ls0:.4f} W={W0:.3f} C={C0:.3f} N={N0:.3f} K={K0:.3f} E={E0:.4f}"
    ax.text(0.5, -0.15, txt, transform=ax.transAxes, fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.20)
    path = os.path.join(out_dir, f"{muscle_name}_refit.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved plot: {path}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs_2026-02-24_21-27-45/01_extract_opensim_data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs_refit"

    params_csv = os.path.join(data_dir, "all_muscle_parameters.csv")
    target_muscles = ["glut_max_r", "rect_fem_r"]

    fitted_rows = []
    for mname in target_muscles:
        res = fit_muscle_improved(mname, data_dir, params_csv, out_dir, verbose=0)
        if res is not None:
            fitted_rows.append([mname] + list(res))

    # Save refit CSV
    if fitted_rows:
        csv_path = os.path.join(out_dir, "refit_params.csv")
        header = ["muscle", "F_max", "l_opt", "l_slack", "v_max", "W", "C", "N", "K", "E_REF"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(fitted_rows)
        print(f"\nSaved refit parameters: {csv_path}")

    # Also create a merged CSV with original params + refit overrides
    orig_csv = os.path.join(os.path.dirname(out_dir), "outputs_2026-02-24_21-27-45",
                            "03_apply_fitted_params", "fitted_params_myoleg22_names.csv")
    if os.path.exists(orig_csv):
        # Read original
        orig_rows = {}
        with open(orig_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_rows[row['muscle']] = row

        # Map refit names to myoleg22 names
        name_map = {"glut_max_r": "glutmax_r", "rect_fem_r": "rectfem_r"}

        # Override with refit
        for row in fitted_rows:
            mname = row[0]
            myoleg_name = name_map.get(mname, mname)
            orig_rows[myoleg_name] = {
                'muscle': myoleg_name,
                'F_max': row[1], 'l_opt': row[2], 'l_slack': row[3], 'v_max': row[4],
                'W': row[5], 'C': row[6], 'N': row[7], 'K': row[8], 'E_REF': row[9]
            }

        merged_csv = os.path.join(out_dir, "merged_params_myoleg22.csv")
        header = ["muscle", "F_max", "l_opt", "l_slack", "v_max", "W", "C", "N", "K", "E_REF"]
        with open(merged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in orig_rows.values():
                writer.writerow(row)
        print(f"Saved merged parameters: {merged_csv}")

        # Also add edl_r/fdl_r from rajagopal supplement
        raj_csv = os.path.join(os.path.dirname(out_dir), "outputs_2026-02-24_21-27-45",
                               "02_fit_mujoco_params", "rajagopal_supplement", "fitted_params_length_only.csv")
        if os.path.exists(raj_csv):
            with open(raj_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig_rows[row['muscle']] = row

            with open(merged_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in orig_rows.values():
                    writer.writerow(row)
            print(f"Updated merged CSV with edl_r/fdl_r: {merged_csv}")
