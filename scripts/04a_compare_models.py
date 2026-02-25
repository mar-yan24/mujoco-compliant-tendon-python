"""
Compare baseline (rigid tendon) vs compliant tendon MuJoCo models side-by-side.

This script launches two MuJoCo viewer windows to compare the muscle behavior
between the original rigid tendon model and the fitted compliant tendon model.

Usage:
    python scripts/04a_compare_models.py [baseline|compliant|both|record] [out_dir]

    record mode: headless rendering to video + comparison plots (no GUI needed)
"""

import mujoco
import numpy as np
import os
import time

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

BASELINE_PATH = os.path.join(repo_root, "myosim_convert", "myoassist", "myoLeg22", "myoLeg22_2D_BASELINE.xml")
COMPLIANT_PATH = os.path.join(repo_root, "myosim_convert", "myoassist", "myoLeg22", "myoLeg22_2D_COMPLIANT.xml")


def get_muscle_actuator_ids(model):
    """Get IDs of muscle actuators (right side only for display)."""
    muscle_ids = []
    muscle_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.endswith('_r'):
            muscle_ids.append(i)
            muscle_names.append(name)
    return muscle_ids, muscle_names


def run_headless(model_path, label, activation_pattern="sine", sim_duration=4.0, fps=30,
                  warmup_time=1.0, timestep=None):
    """
    Run a headless simulation, recording frames, forces, and lengths.

    Args:
        warmup_time: seconds to simulate before recording (ramps activation gradually)
        timestep: override model timestep (smaller = more stable for compliant tendons)

    Returns:
        dict with keys: frames, time, forces (dict), lengths (dict), muscle_names
    """
    print(f"[{label}] Loading: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    if timestep is not None:
        model.opt.timestep = timestep
        print(f"[{label}] Timestep overridden to {timestep}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    muscle_ids, muscle_names = get_muscle_actuator_ids(model)
    print(f"[{label}] Found {len(muscle_ids)} right-side muscles")

    activation_freq = 0.5

    frames = []
    time_steps = []
    force_history = {n: [] for n in muscle_names}
    length_history = {n: [] for n in muscle_names}

    render_dt = 1.0 / fps
    n_frames = int(sim_duration * fps)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Warmup phase: ramp activation from 0 to starting sine value gradually
    if warmup_time > 0:
        print(f"[{label}] Warming up for {warmup_time}s (gradual ramp)...")
        while data.time < warmup_time:
            # Ramp from 0 to the sine value at warmup_time
            ramp_frac = data.time / warmup_time
            sine_val = 0.5 * (1 + np.sin(2 * np.pi * activation_freq * data.time))
            activation = ramp_frac * sine_val
            data.ctrl[:] = activation
            mujoco.mj_step(model, data)
        print(f"[{label}] Warmup complete at t={data.time:.3f}s")

    # Record the time offset so plots start at 0
    record_start_time = data.time

    for frame_i in range(n_frames):
        target_time = record_start_time + frame_i * render_dt
        while data.time < target_time:
            t = data.time
            if activation_pattern == "sine":
                activation = 0.5 * (1 + np.sin(2 * np.pi * activation_freq * t))
            elif activation_pattern == "constant":
                activation = 0.5
            elif activation_pattern == "ramp":
                activation = min(1.0, t / 5.0)
            else:
                activation = 0.0
            data.ctrl[:] = activation
            mujoco.mj_step(model, data)

        # Render
        renderer.update_scene(data, camera=0)
        pixels = renderer.render()
        frames.append(pixels.copy())
        time_steps.append(data.time - record_start_time)

        # Record
        for name, mid in zip(muscle_names, muscle_ids):
            force_history[name].append(data.actuator_force[mid])
            length_history[name].append(data.actuator_length[mid])

    print(f"[{label}] Captured {len(frames)} frames over {sim_duration:.1f}s (sim time {data.time:.2f}s)")
    return {
        "frames": frames,
        "time": np.array(time_steps),
        "forces": {n: np.array(v) for n, v in force_history.items()},
        "lengths": {n: np.array(v) for n, v in length_history.items()},
        "muscle_names": muscle_names,
    }


def run_record_comparison(out_dir):
    """Headless: record videos + comparison plots for baseline vs compliant."""
    import imageio
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: Baseline model not found: {BASELINE_PATH}")
        return
    if not os.path.exists(COMPLIANT_PATH):
        print(f"ERROR: Compliant model not found: {COMPLIANT_PATH}")
        return

    # Run both headlessly with warmup and smaller timestep
    smaller_dt = 0.0005
    warmup = 1.0
    base_data = run_headless(BASELINE_PATH, "BASELINE", activation_pattern="sine",
                             warmup_time=warmup, timestep=smaller_dt)
    comp_data = run_headless(COMPLIANT_PATH, "COMPLIANT", activation_pattern="sine",
                             warmup_time=warmup, timestep=smaller_dt)

    # Save videos
    for label, rec in [("baseline", base_data), ("compliant", comp_data)]:
        vpath = os.path.join(out_dir, f"comparison_{label}.mp4")
        imageio.mimsave(vpath, rec["frames"], fps=30)
        print(f"Saved video: {vpath}")

    # Comparison plots: force and length per muscle
    muscles = base_data["muscle_names"]
    n = len(muscles)
    ncols = min(n, 6)
    nrows = int(np.ceil(n / ncols))

    # --- Force comparison ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, mname in enumerate(muscles):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        ax.plot(base_data["time"], -base_data["forces"][mname], label="Baseline", alpha=0.8)
        ax.plot(comp_data["time"], -comp_data["forces"][mname], label="Compliant", alpha=0.8, linestyle="--")
        ax.set_title(mname, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Force (N)", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    for k in range(n, nrows * ncols):
        fig.delaxes(axes[k // ncols][k % ncols])
    fig.suptitle("Force Comparison: Baseline vs Compliant (sine activation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(out_dir, "compare_forces.png")
    fig.savefig(fpath, dpi=200)
    plt.close(fig)
    print(f"Saved force comparison: {fpath}")

    # --- Length comparison ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, mname in enumerate(muscles):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        ax.plot(base_data["time"], base_data["lengths"][mname], label="Baseline", alpha=0.8)
        ax.plot(comp_data["time"], comp_data["lengths"][mname], label="Compliant", alpha=0.8, linestyle="--")
        ax.set_title(mname, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Length (m)", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    for k in range(n, nrows * ncols):
        fig.delaxes(axes[k // ncols][k % ncols])
    fig.suptitle("Length Comparison: Baseline vs Compliant (sine activation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lpath = os.path.join(out_dir, "compare_lengths.png")
    fig.savefig(lpath, dpi=200)
    plt.close(fig)
    print(f"Saved length comparison: {lpath}")

    print("\nComparison recording complete.")


def run_viewer(model_path, title, activation_pattern="sine"):
    """
    Run a MuJoCo viewer for the given model.

    Args:
        model_path: Path to the MuJoCo XML model
        title: Window title
        activation_pattern: "sine" for oscillating activation, "constant" for fixed activation
    """
    print(f"Loading: {title}")
    print(f"  Path: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Get muscle actuators
    muscle_ids, muscle_names = get_muscle_actuator_ids(model)
    print(f"  Found {len(muscle_ids)} right-side muscles")

    # Simulation parameters
    activation_freq = 0.5  # Hz - slow oscillation to observe muscle behavior

    def controller(model, data):
        """Apply muscle activations."""
        t = data.time

        if activation_pattern == "sine":
            # Oscillating activation (0 to 1) for all muscles
            activation = 0.5 * (1 + np.sin(2 * np.pi * activation_freq * t))
        elif activation_pattern == "constant":
            activation = 0.5
        elif activation_pattern == "ramp":
            # Ramp up over 5 seconds, then hold
            activation = min(1.0, t / 5.0)
        else:
            activation = 0.0

        # Apply to all actuators
        data.ctrl[:] = activation

    # Launch viewer with controller callback
    print(f"  Launching viewer: {title}")
    print(f"  - Press SPACE to pause/resume")
    print(f"  - Press R to reset")
    print(f"  - Press ESC to close")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set window title (if supported)
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120

        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # Apply controller
            controller(model, data)

            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Maintain realtime
            elapsed = time.time() - step_start
            dt = model.opt.timestep
            if elapsed < dt:
                time.sleep(dt - elapsed)


def run_dual_comparison():
    """Run both models in separate threads for side-by-side comparison."""

    print("=" * 60)
    print("MuJoCo Model Comparison: Baseline vs Compliant Tendon")
    print("=" * 60)
    print()
    print("This will launch TWO viewer windows:")
    print("  1. BASELINE (rigid tendon) - original muscle model")
    print("  2. COMPLIANT (fitted) - compliant tendon muscle model")
    print()
    print("Both models will have synchronized muscle activations")
    print("(sinusoidal pattern) so you can compare the force/length behavior.")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume simulation")
    print("  R     - Reset simulation")
    print("  ESC   - Close viewer")
    print()

    # Check files exist
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: Baseline model not found: {BASELINE_PATH}")
        return
    if not os.path.exists(COMPLIANT_PATH):
        print(f"ERROR: Compliant model not found: {COMPLIANT_PATH}")
        return

    # Run viewers sequentially (MuJoCo viewer doesn't support multiple windows well in threads)
    print("Starting BASELINE model viewer first...")
    print("Close the BASELINE viewer to open the COMPLIANT viewer.")
    print()

    # Run baseline first
    run_viewer(BASELINE_PATH, "BASELINE (Rigid Tendon)", activation_pattern="sine")

    print()
    print("Starting COMPLIANT model viewer...")
    print()

    # Then run compliant
    run_viewer(COMPLIANT_PATH, "COMPLIANT (Fitted Tendon)", activation_pattern="sine")

    print()
    print("Comparison complete.")


def run_single_model(model_type="compliant"):
    """Run a single model viewer."""
    if model_type.lower() == "baseline":
        run_viewer(BASELINE_PATH, "BASELINE (Rigid Tendon)", activation_pattern="sine")
    else:
        run_viewer(COMPLIANT_PATH, "COMPLIANT (Fitted Tendon)", activation_pattern="sine")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "baseline":
            import mujoco.viewer
            run_single_model("baseline")
        elif arg == "compliant":
            import mujoco.viewer
            run_single_model("compliant")
        elif arg == "both":
            import mujoco.viewer
            run_dual_comparison()
        elif arg == "record":
            _out_dir = sys.argv[2] if len(sys.argv) > 2 else "comparison_output"
            run_record_comparison(_out_dir)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python 04a_compare_models.py [baseline|compliant|both|record] [out_dir]")
    else:
        # Default: run comparison
        import mujoco.viewer
        run_dual_comparison()
