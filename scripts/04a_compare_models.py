"""
Compare baseline (rigid tendon) vs compliant tendon MuJoCo models side-by-side.

This script launches two MuJoCo viewer windows to compare the muscle behavior
between the original rigid tendon model and the fitted compliant tendon model.

Usage:
    python scripts/05_compare_models.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import threading
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
            run_single_model("baseline")
        elif arg == "compliant":
            run_single_model("compliant")
        elif arg == "both":
            run_dual_comparison()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python 05_compare_models.py [baseline|compliant|both]")
    else:
        # Default: run comparison
        run_dual_comparison()
