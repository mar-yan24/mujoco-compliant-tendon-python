"""
10_step_response_test.py — Step-response comparison between OpenSim and MuJoCo CMTU.

Applies a step activation (0->1) at l_mtu = l_opt + l_slack and compares
force rise time and overshoot. Flags muscles with >20% rise time error.

This test captures dynamic tendon behavior that isometric fitting cannot reveal:
different tendon compliance models produce different transient responses even
if they match the same steady-state forces.

Usage:
    python scripts/10_step_response_test.py [fitted_csv] [out_dir]

    fitted_csv: CSV from 02_fit_mujoco_params.py
    out_dir:    Output directory for plots and timing report

Note: OpenSim step-response requires Python 3.8 + OpenSim 4.5 environment.
      This script runs the MuJoCo side only. For full comparison, run the
      OpenSim extraction separately and merge results.
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


def load_fitted_params_csv(csv_path):
    """Load fitted parameters from CSV."""
    params = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            muscle = row["muscle"]
            if not muscle:
                continue
            params[muscle] = {
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
    return params


def create_step_model(params):
    """Create a MuJoCo model for step-response testing with fixed MTU length."""
    prm_str = " ".join(f"{params[k]:.12g}" for k in
                       ["F_max", "l_opt", "l_slack", "v_max", "W", "C", "N", "K", "E_REF"])

    # MTU length = l_opt + l_slack (optimal operating point)
    l_mtu = params["l_opt"] + params["l_slack"]

    xml = f"""
    <mujoco model="step_response_test">
    <option timestep="0.0005" integrator="Euler"/>

    <default>
        <default class="compliant_muscle">
        <general biasprm="0" biastype="none" ctrllimited="true" ctrlrange="0 1"
                dynprm="0.01 0.04" dyntype="muscle"
                gainprm="{prm_str}"
                gaintype="compliant_mtu"/>
        </default>
    </default>

    <worldbody>
        <body name="ground"/>
        <site name="anchor" pos="0 0 0" size="0.01"/>

        <body name="load" pos="0 0 0">
            <joint name="slide" type="slide" axis="0 0 1" limited="true"
                   range="{-l_mtu - 0.01:.6f} {-l_mtu + 0.01:.6f}" damping="1000"/>
            <site name="insertion" pos="0 0 0" size="0.01"/>
            <geom type="sphere" size="0.02" mass="100.0"/>
        </body>
    </worldbody>

    <tendon>
        <spatial name="tendon">
            <site site="anchor"/>
            <site site="insertion"/>
        </spatial>
    </tendon>

    <actuator>
        <general class="compliant_muscle" name="muscle" tendon="tendon"/>
    </actuator>

    <sensor>
        <actuatorfrc name="force_sensor" actuator="muscle"/>
        <tendonpos name="length_sensor" tendon="tendon"/>
    </sensor>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), l_mtu


def run_step_response(params, duration=0.5):
    """
    Run step activation (0->1) and record force trajectory.

    Returns:
        time_vec: array of time points
        force_vec: array of MTU force at each timestep
        metrics: dict with rise_time_90, overshoot_pct, steady_state_force
    """
    model, l_mtu = create_step_model(params)
    data = mujoco.MjData(model)

    # Set initial position
    data.qpos[0] = -l_mtu
    data.qvel[0] = 0.0

    # Initialize with zero activation
    data.ctrl[0] = 0.0
    mujoco.mj_forward(model, data)

    # Let passive forces settle (100 steps at act=0)
    for _ in range(100):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)

    dt = model.opt.timestep
    n_steps = int(duration / dt)

    time_vec = np.zeros(n_steps)
    force_vec = np.zeros(n_steps)

    # Apply step activation: 0 -> 1
    for i in range(n_steps):
        data.ctrl[0] = 1.0
        mujoco.mj_step(model, data)
        time_vec[i] = i * dt
        force_vec[i] = data.qfrc_actuator[0]

    # Compute metrics
    f_max_recorded = np.max(np.abs(force_vec))
    f_steady = np.mean(np.abs(force_vec[-int(0.1 / dt):]))  # last 100ms average

    if f_steady > 1e-3:
        # Rise time: time to reach 90% of steady state
        target_90 = 0.9 * f_steady
        rise_indices = np.where(np.abs(force_vec) >= target_90)[0]
        rise_time_90 = rise_indices[0] * dt if len(rise_indices) > 0 else duration

        # Overshoot
        overshoot_pct = ((f_max_recorded - f_steady) / f_steady) * 100
    else:
        rise_time_90 = np.nan
        overshoot_pct = np.nan

    metrics = {
        "rise_time_90": rise_time_90,
        "overshoot_pct": overshoot_pct,
        "steady_state_force": f_steady,
        "peak_force": f_max_recorded,
    }

    return time_vec, force_vec, metrics


def main():
    fitted_csv = sys.argv[1] if len(sys.argv) > 1 else "mujoco_muscle_data/Rajagopal/fitted_params_length_only.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs_step_response"

    all_params = load_fitted_params_csv(fitted_csv)
    print(f"Loaded {len(all_params)} muscles from {fitted_csv}")

    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []
    muscles = sorted(all_params.keys())

    # Figure: grid of step responses
    n = len(muscles)
    ncols = min(8, n)
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, mname in enumerate(muscles):
        params = all_params[mname]
        ratio = params["l_slack"] / params["l_opt"] if params["l_opt"] > 0 else 0
        print(f"  {mname} (ratio={ratio:.2f})...", end=" ")

        try:
            time_vec, force_vec, metrics = run_step_response(params, duration=0.3)
        except Exception as e:
            print(f"ERROR: {e}")
            metrics = {"rise_time_90": np.nan, "overshoot_pct": np.nan,
                        "steady_state_force": np.nan, "peak_force": np.nan}
            time_vec = force_vec = None

        metrics["muscle"] = mname
        metrics["tendon_ratio"] = ratio
        all_metrics.append(metrics)

        print(f"rise={metrics['rise_time_90']:.4f}s, overshoot={metrics['overshoot_pct']:.1f}%")

        # Plot
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        if time_vec is not None:
            ax.plot(time_vec * 1000, np.abs(force_vec), "r-", linewidth=0.8)
            if not np.isnan(metrics["steady_state_force"]):
                ax.axhline(metrics["steady_state_force"], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{mname}\nt90={metrics['rise_time_90']:.3f}s", fontsize=7)
        ax.set_xlabel("Time (ms)", fontsize=6)
        if c == 0:
            ax.set_ylabel("Force (N)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for k in range(n, nrows * ncols):
        fig.delaxes(axes[k // ncols][k % ncols])

    fig.suptitle("Step Response: 0->1 Activation at l_mtu = l_opt + l_slack", fontsize=11)
    fig.tight_layout()
    grid_path = os.path.join(out_dir, "step_response_all.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved grid plot: {grid_path}")

    # Save metrics CSV
    csv_path = os.path.join(out_dir, "step_response_metrics.csv")
    fieldnames = ["muscle", "tendon_ratio", "rise_time_90", "overshoot_pct",
                  "steady_state_force", "peak_force"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m[k] for k in fieldnames})
    print(f"Saved metrics: {csv_path}")

    # Flag muscles with potential issues
    print(f"\nMuscles with rise_time > 50ms or overshoot > 20%:")
    print(f"  {'Muscle':<20s} {'Ratio':>6s} {'Rise(ms)':>9s} {'Overshoot%':>11s} {'F_steady':>9s}")
    flagged = 0
    for m in sorted(all_metrics, key=lambda x: x.get("rise_time_90", 0) or 0, reverse=True):
        rt = m.get("rise_time_90", np.nan)
        ov = m.get("overshoot_pct", np.nan)
        if (not np.isnan(rt) and rt > 0.050) or (not np.isnan(ov) and ov > 20):
            print(f"  {m['muscle']:<20s} {m['tendon_ratio']:>6.2f} {rt*1000:>9.1f} {ov:>11.1f} {m['steady_state_force']:>9.1f}")
            flagged += 1
    if flagged == 0:
        print("  (none)")


if __name__ == "__main__":
    main()
