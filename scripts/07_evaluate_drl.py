"""
Evaluate trained DRL walking policies: record videos + comparison plots.

Loads best SAC models for baseline (rigid) and compliant (fitted) myoLeg22,
runs deterministic rollouts, and produces:
  - Side-by-side walking videos (MP4)
  - Per-muscle activation heatmaps
  - Joint angle trajectories
  - Pelvis height / forward velocity / reward time-series
  - Summary statistics

Usage:
    python scripts/07_evaluate_drl.py [--out-dir outputs_drl/walk/eval]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import SAC

# Reuse the environment from the training script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib.machinery import SourceFileLoader

_train_mod = SourceFileLoader("drl", str(SCRIPT_DIR / "06_drl_comparison.py")).load_module()
MyoLeg22Env = _train_mod.MyoLeg22Env

MODEL_DIR = REPO_ROOT / "myosim_convert" / "myoassist" / "myoLeg22"
BASELINE_XML = MODEL_DIR / "myoLeg22_2D_BASELINE.xml"
COMPLIANT_XML = MODEL_DIR / "myoLeg22_2D_COMPLIANT.xml"

BEST_BASELINE = REPO_ROOT / "outputs_drl" / "walk" / "best_models" / "baseline_walk" / "best_model.zip"
BEST_COMPLIANT = REPO_ROOT / "outputs_drl" / "walk" / "best_models" / "compliant_walk" / "best_model.zip"


# ---------------------------------------------------------------------------
# Rollout with full recording
# ---------------------------------------------------------------------------
def run_rollout(agent, env, model_mj, data_mj, renderer, n_episodes=3):
    """Run deterministic rollouts and record everything."""
    muscle_names = []
    for i in range(model_mj.nu):
        name = mujoco.mj_id2name(model_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        muscle_names.append(name if name else f"act_{i}")

    # Joint names for the main leg DOFs
    leg_joint_names = [
        "hip_flexion_r", "knee_angle_r", "ankle_angle_r", "mtp_angle_r",
        "hip_flexion_l", "knee_angle_l", "ankle_angle_l", "mtp_angle_l",
    ]
    leg_joint_ids = []
    for jn in leg_joint_names:
        jid = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, jn)
        leg_joint_ids.append(jid)

    pelvis_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    all_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        frames = []
        activations = []     # (T, nu)
        joint_angles = []    # (T, n_leg_joints)
        pelvis_heights = []
        forward_vels = []
        rewards = []
        actuator_forces = []
        times = []
        step = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1

            # Record muscle activations
            activations.append(env.data.ctrl.copy())
            actuator_forces.append(env.data.actuator_force.copy())

            # Record joint angles
            angles = []
            for jid in leg_joint_ids:
                if jid >= 0:
                    qadr = model_mj.jnt_qposadr[jid]
                    angles.append(env.data.qpos[qadr])
                else:
                    angles.append(0.0)
            joint_angles.append(angles)

            pelvis_heights.append(info["pelvis_height"])
            forward_vels.append(info["forward_vel"])
            rewards.append(r)
            times.append(step * env.frame_skip * model_mj.opt.timestep)

            # Render frame (every 2nd step for reasonable video size)
            if step % 2 == 0:
                renderer.update_scene(env.data, camera=-1)
                frames.append(renderer.render().copy())

        all_episodes.append({
            "frames": frames,
            "activations": np.array(activations),
            "actuator_forces": np.array(actuator_forces),
            "joint_angles": np.array(joint_angles),
            "pelvis_heights": np.array(pelvis_heights),
            "forward_vels": np.array(forward_vels),
            "rewards": np.array(rewards),
            "times": np.array(times),
            "total_reward": sum(rewards),
            "episode_length": step,
        })
        print(f"  Episode {ep+1}: {step} steps, reward={sum(rewards):.1f}, "
              f"avg_vel={np.mean(forward_vels):.3f} m/s")

    return all_episodes, muscle_names, leg_joint_names


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_timeseries(baseline_ep, compliant_ep, out_dir):
    """Plot pelvis height, forward velocity, and reward over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for data, label, color in [
        (baseline_ep, "Baseline (rigid)", "#2196F3"),
        (compliant_ep, "Compliant (fitted)", "#FF5722"),
    ]:
        t = data["times"]
        axes[0].plot(t, data["pelvis_heights"], label=label, color=color, alpha=0.85)
        axes[1].plot(t, data["forward_vels"], label=label, color=color, alpha=0.85)
        axes[2].plot(t, np.cumsum(data["rewards"]), label=label, color=color, alpha=0.85)

    axes[0].set_ylabel("Pelvis Height (m)")
    axes[0].set_title("Pelvis Height Over Time")
    axes[0].axhline(y=0.91, color="gray", linestyle="--", alpha=0.5, label="Standing height")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Forward Velocity (m/s)")
    axes[1].set_title("Forward Velocity Over Time")
    axes[1].axhline(y=1.2, color="gray", linestyle="--", alpha=0.5, label="Target vel")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Cumulative Reward")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Cumulative Reward")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "timeseries_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_activation_heatmaps(baseline_ep, compliant_ep, muscle_names, out_dir):
    """Plot muscle activation heatmaps for both models."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    for ax, data, title in [
        (axes[0], baseline_ep, "Baseline (rigid) — Muscle Activations"),
        (axes[1], compliant_ep, "Compliant (fitted) — Muscle Activations"),
    ]:
        act = data["activations"].T  # (nu, T)
        im = ax.imshow(act, aspect="auto", cmap="hot", vmin=0, vmax=1,
                        interpolation="nearest")
        ax.set_yticks(range(len(muscle_names)))
        ax.set_yticklabels(muscle_names, fontsize=7)
        ax.set_xlabel("Timestep")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Activation")

    fig.tight_layout()
    path = out_dir / "activation_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_joint_angles(baseline_ep, compliant_ep, joint_names, out_dir):
    """Plot leg joint angle trajectories."""
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 2.5 * n_joints), sharex=True)

    for i, jn in enumerate(joint_names):
        ax = axes[i]
        for data, label, color in [
            (baseline_ep, "Baseline", "#2196F3"),
            (compliant_ep, "Compliant", "#FF5722"),
        ]:
            t = data["times"]
            ax.plot(t, np.degrees(data["joint_angles"][:, i]),
                    label=label, color=color, alpha=0.85)
        ax.set_ylabel(f"{jn}\n(deg)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Leg Joint Angles During Walking", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "joint_angles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_muscle_forces(baseline_ep, compliant_ep, muscle_names, out_dir):
    """Plot actuator forces for right-leg muscles."""
    # Right-side muscles only
    r_indices = [i for i, n in enumerate(muscle_names) if n.endswith("_r")]
    r_names = [muscle_names[i] for i in r_indices]
    n = len(r_names)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    for idx, (ri, rn) in enumerate(zip(r_indices, r_names)):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        for data, label, color in [
            (baseline_ep, "Baseline", "#2196F3"),
            (compliant_ep, "Compliant", "#FF5722"),
        ]:
            t = data["times"]
            forces = -data["actuator_forces"][:, ri]  # negate for tension
            ax.plot(t, forces, label=label, color=color, alpha=0.7, linewidth=0.8)
        ax.set_title(rn, fontsize=9)
        ax.set_ylabel("Force (N)", fontsize=7)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for k in range(n, nrows * ncols):
        fig.delaxes(axes[k // ncols][k % ncols])

    axes[-1][0].set_xlabel("Time (s)")
    fig.suptitle("Muscle Forces During DRL Walking (Right Leg)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "muscle_forces.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def make_side_by_side_video(baseline_frames, compliant_frames, out_path, fps=30):
    """Stitch frames side-by-side into one video."""
    min_len = min(len(baseline_frames), len(compliant_frames))
    combined = []
    for i in range(min_len):
        bf = baseline_frames[i]
        cf = compliant_frames[i]
        # Resize to same height if needed
        h = min(bf.shape[0], cf.shape[0])
        bf = bf[:h]
        cf = cf[:h]
        # Add labels
        frame = np.concatenate([bf, cf], axis=1)
        combined.append(frame)

    imageio.mimsave(str(out_path), combined, fps=fps)
    print(f"Saved: {out_path} ({min_len} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DRL walking policies")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs_drl" / "walk" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify files
    for p in [BEST_BASELINE, BEST_COMPLIANT, BASELINE_XML, COMPLIANT_XML]:
        if not p.exists():
            print(f"ERROR: Not found: {p}")
            sys.exit(1)

    results = {}

    for variant, model_zip, xml_path in [
        ("baseline", BEST_BASELINE, BASELINE_XML),
        ("compliant", BEST_COMPLIANT, COMPLIANT_XML),
    ]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {variant}")
        print(f"  Policy: {model_zip}")
        print(f"  Model:  {xml_path}")
        print(f"{'='*60}")

        env = MyoLeg22Env(model_path=str(xml_path), task="walk")
        agent = SAC.load(str(model_zip), env=env)
        renderer = mujoco.Renderer(env.model, height=720, width=1280)

        episodes, muscle_names, joint_names = run_rollout(
            agent, env, env.model, env.data, renderer, n_episodes=args.n_episodes
        )

        # Save individual videos
        for i, ep in enumerate(episodes):
            vpath = out_dir / f"{variant}_walk_ep{i}.mp4"
            imageio.mimsave(str(vpath), ep["frames"], fps=30)
            print(f"  Video: {vpath}")

        results[variant] = {
            "episodes": episodes,
            "muscle_names": muscle_names,
            "joint_names": joint_names,
        }

        renderer.close()
        env.close()

    # Pick best episode from each for comparison plots
    best_b = max(results["baseline"]["episodes"], key=lambda e: e["total_reward"])
    best_c = max(results["compliant"]["episodes"], key=lambda e: e["total_reward"])
    muscle_names = results["baseline"]["muscle_names"]
    joint_names = results["baseline"]["joint_names"]

    print(f"\nBest baseline episode: {best_b['episode_length']} steps, reward={best_b['total_reward']:.1f}")
    print(f"Best compliant episode: {best_c['episode_length']} steps, reward={best_c['total_reward']:.1f}")

    # Plots
    print("\nGenerating comparison plots...")
    plot_timeseries(best_b, best_c, out_dir)
    plot_activation_heatmaps(best_b, best_c, muscle_names, out_dir)
    plot_joint_angles(best_b, best_c, joint_names, out_dir)
    plot_muscle_forces(best_b, best_c, muscle_names, out_dir)

    # Side-by-side video
    print("\nGenerating side-by-side video...")
    make_side_by_side_video(
        best_b["frames"], best_c["frames"],
        out_dir / "side_by_side_walk.mp4", fps=30,
    )

    # Summary stats
    summary = {}
    for variant in ["baseline", "compliant"]:
        eps = results[variant]["episodes"]
        summary[variant] = {
            "reward_mean": float(np.mean([e["total_reward"] for e in eps])),
            "reward_std": float(np.std([e["total_reward"] for e in eps])),
            "episode_length_mean": float(np.mean([e["episode_length"] for e in eps])),
            "avg_forward_vel": float(np.mean([np.mean(e["forward_vels"]) for e in eps])),
            "avg_pelvis_height": float(np.mean([np.mean(e["pelvis_heights"]) for e in eps])),
            "avg_ctrl_norm": float(np.mean([np.mean(np.linalg.norm(e["activations"], axis=1)) for e in eps])),
            "avg_force_norm": float(np.mean([np.mean(np.linalg.norm(e["actuator_forces"], axis=1)) for e in eps])),
        }

    print(f"\n{'='*70}")
    print(f"{'Metric':<30} {'Baseline':>15} {'Compliant':>15}")
    print(f"{'-'*70}")
    for key in summary["baseline"]:
        b = summary["baseline"][key]
        c = summary["compliant"][key]
        print(f"{key:<30} {b:>15.3f} {c:>15.3f}")
    print(f"{'='*70}")

    stats_path = out_dir / "eval_summary.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {stats_path}")
    print(f"All outputs in {out_dir}")


if __name__ == "__main__":
    main()
