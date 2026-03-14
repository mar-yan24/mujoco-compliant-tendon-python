"""
Overnight DRL comparison: Compliant vs Rigid tendon myoLeg22 models.

Trains SAC policies on both models for standing and walking tasks, then
compares learning curves and final performance metrics.

Usage:
    pip install stable-baselines3 tensorboard
    python scripts/06_drl_comparison.py [--task stand|walk] [--timesteps 5000000]

    # Monitor training live:
    tensorboard --logdir outputs_drl/

Requirements (in addition to existing requirements.txt):
    stable-baselines3>=2.0.0
    tensorboard>=2.14.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_DIR = REPO_ROOT / "myosim_convert" / "myoassist" / "myoLeg22"
BASELINE_PATH = MODEL_DIR / "myoLeg22_2D_BASELINE.xml"
COMPLIANT_PATH = MODEL_DIR / "myoLeg22_2D_COMPLIANT.xml"


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
class MyoLeg22Env(gym.Env):
    """
    Gymnasium environment for the myoLeg22 2D sagittal-plane model.

    Supports two tasks:
        - "stand": maintain upright balance from the standing keyframe
        - "walk":  walk forward at a target velocity while staying upright

    Observation (dim = nq + nv + nu):
        [qpos, clipped_qvel, muscle_activations]
    Action (dim = nu = 22):
        Muscle activations clipped to [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        model_path: str,
        task: str = "stand",
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        target_walk_speed: float = 1.2,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.task = task
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self._max_episode_steps = max_episode_steps
        self.target_walk_speed = target_walk_speed

        # Load model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Use the "stand" keyframe as the initial state
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id >= 0:
            self.init_qpos = self.model.key_qpos[key_id].copy()
            self.init_qvel = self.model.key_qvel[key_id].copy()
        else:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.init_qpos = self.data.qpos.copy()
            self.init_qvel = self.data.qvel.copy()

        # Body IDs for reward computation
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Remember standing pelvis height from keyframe
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        mujoco.mj_forward(self.model, self.data)
        self.stand_pelvis_height = self.data.xpos[self.pelvis_id, 2]

        # Spaces
        obs_dim = self.model.nq + self.model.nv + self.model.nu
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )

        # Renderer
        self._renderer = None
        if render_mode in ("human", "rgb_array"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Step counter
        self._step_count = 0

    # -- observation ----------------------------------------------------------
    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = np.clip(self.data.qvel.flat.copy(), -50.0, 50.0)
        act = (
            self.data.act.flat.copy()
            if self.data.act.size > 0
            else self.data.ctrl.flat.copy()
        )
        return np.concatenate([qpos, qvel, act])

    # -- reward ---------------------------------------------------------------
    def _reward_stand(self):
        pelvis_h = self.data.xpos[self.pelvis_id, 2]
        # Height maintenance
        height_rew = np.exp(-40.0 * (pelvis_h - self.stand_pelvis_height) ** 2)
        # Penalise large velocities (want to be still)
        vel_penalty = -0.005 * np.sum(self.data.qvel[:3] ** 2)
        # Penalise excessive control
        ctrl_penalty = -0.0005 * np.sum(self.data.ctrl ** 2)
        # Alive bonus
        alive = 0.2
        return height_rew + vel_penalty + ctrl_penalty + alive

    def _reward_walk(self):
        pelvis_h = self.data.xpos[self.pelvis_id, 2]
        # Forward velocity (pelvis_tx is qvel[0] in the 2D model)
        forward_vel = self.data.qvel[0]
        vel_rew = np.exp(-2.0 * (forward_vel - self.target_walk_speed) ** 2)
        # Stay upright
        height_rew = np.exp(-40.0 * (pelvis_h - self.stand_pelvis_height) ** 2)
        # Control cost
        ctrl_penalty = -0.0005 * np.sum(self.data.ctrl ** 2)
        alive = 0.1
        return vel_rew + 0.5 * height_rew + ctrl_penalty + alive

    # -- termination ----------------------------------------------------------
    def _is_terminated(self):
        pelvis_h = self.data.xpos[self.pelvis_id, 2]
        return pelvis_h < 0.35  # fell over

    # -- step -----------------------------------------------------------------
    def step(self, action):
        self.data.ctrl[:] = np.clip(action, 0.0, 1.0)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._reward_stand() if self.task == "stand" else self._reward_walk()
        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_episode_steps
        info = {
            "pelvis_height": float(self.data.xpos[self.pelvis_id, 2]),
            "forward_vel": float(self.data.qvel[0]),
            "ctrl_norm": float(np.linalg.norm(self.data.ctrl)),
            "actuator_force_norm": float(np.linalg.norm(self.data.actuator_force)),
        }
        return obs, reward, terminated, truncated, info

    # -- reset ----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise_q = self.np_random.uniform(-0.005, 0.005, self.init_qpos.shape)
        noise_v = self.np_random.uniform(-0.005, 0.005, self.init_qvel.shape)
        self.data.qpos[:] = self.init_qpos + noise_q
        self.data.qvel[:] = self.init_qvel + noise_v
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    # -- render ---------------------------------------------------------------
    def render(self):
        if self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------------
# Extra TensorBoard logging callback
# ---------------------------------------------------------------------------
class BiomechMetricsCallback(BaseCallback):
    """Logs biomechanics-specific metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_pelvis_heights = []
        self._ep_forward_vels = []
        self._ep_ctrl_norms = []
        self._ep_force_norms = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                self._ep_pelvis_heights.append(info.get("pelvis_height", 0))
                self._ep_forward_vels.append(info.get("forward_vel", 0))
                self._ep_ctrl_norms.append(info.get("ctrl_norm", 0))
                self._ep_force_norms.append(info.get("actuator_force_norm", 0))

        # Log every 5000 steps
        if self.num_timesteps % 5000 == 0 and self._ep_pelvis_heights:
            self.logger.record(
                "biomech/pelvis_height_mean", np.mean(self._ep_pelvis_heights)
            )
            self.logger.record(
                "biomech/forward_vel_mean", np.mean(self._ep_forward_vels)
            )
            self.logger.record(
                "biomech/ctrl_norm_mean", np.mean(self._ep_ctrl_norms)
            )
            self.logger.record(
                "biomech/actuator_force_norm_mean", np.mean(self._ep_force_norms)
            )
            self._ep_pelvis_heights.clear()
            self._ep_forward_vels.clear()
            self._ep_ctrl_norms.clear()
            self._ep_force_norms.clear()
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(model_path, task, seed=0):
    """Factory that returns a thunk for SubprocVecEnv / DummyVecEnv."""

    def _init():
        env = MyoLeg22Env(model_path=model_path, task=task)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    label: str,
    model_path: Path,
    task: str,
    total_timesteps: int,
    out_dir: Path,
    n_envs: int = 4,
    seed: int = 42,
):
    """Train a SAC policy and return the path to the best model."""
    print(f"\n{'='*60}")
    print(f"Training: {label} | task={task} | timesteps={total_timesteps:,}")
    print(f"Model:    {model_path}")
    print(f"Output:   {out_dir}")
    print(f"{'='*60}\n")

    log_dir = out_dir / "logs" / label
    best_dir = out_dir / "best_models" / label
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Vectorised training environments
    env_fns = [make_env(str(model_path), task, seed=seed + i) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)

    # Single eval environment
    eval_env = DummyVecEnv([make_env(str(model_path), task, seed=seed + 100)])

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    biomech_cb = BiomechMetricsCallback()

    # SAC with sensible defaults for musculoskeletal control
    agent = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(out_dir / "tb_logs"),
        seed=seed,
        verbose=1,
    )

    t0 = time.time()
    agent.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, biomech_cb],
        tb_log_name=label,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # Save final model
    final_path = out_dir / "final_models" / label
    final_path.mkdir(parents=True, exist_ok=True)
    agent.save(str(final_path / "model"))

    print(f"\n[{label}] Training complete in {elapsed/3600:.1f}h")
    print(f"  Best model: {best_dir / 'best_model.zip'}")
    print(f"  Final model: {final_path / 'model.zip'}")

    train_env.close()
    eval_env.close()
    return final_path / "model.zip"


# ---------------------------------------------------------------------------
# Evaluation & comparison
# ---------------------------------------------------------------------------
def evaluate_model(model_zip: Path, model_xml: Path, task: str, n_episodes: int = 20):
    """Load a trained SAC model and evaluate it, returning summary stats."""
    env = MyoLeg22Env(model_path=str(model_xml), task=task)
    env = Monitor(env)
    agent = SAC.load(str(model_zip), env=env)

    ep_rewards = []
    ep_lengths = []
    ep_heights = []
    ep_vels = []
    ep_ctrls = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        heights, vels, ctrls = [], [], []
        steps = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            steps += 1
            heights.append(info["pelvis_height"])
            vels.append(info["forward_vel"])
            ctrls.append(info["ctrl_norm"])
            done = term or trunc

        ep_rewards.append(total_r)
        ep_lengths.append(steps)
        ep_heights.append(np.mean(heights))
        ep_vels.append(np.mean(vels))
        ep_ctrls.append(np.mean(ctrls))

    env.close()
    return {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "episode_length_mean": float(np.mean(ep_lengths)),
        "pelvis_height_mean": float(np.mean(ep_heights)),
        "forward_vel_mean": float(np.mean(ep_vels)),
        "ctrl_norm_mean": float(np.mean(ep_ctrls)),
    }


def print_comparison(baseline_stats, compliant_stats):
    """Pretty-print side-by-side comparison."""
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Baseline':>15} {'Compliant':>15}")
    print("-" * 70)
    for key in baseline_stats:
        b = baseline_stats[key]
        c = compliant_stats[key]
        print(f"{key:<30} {b:>15.3f} {c:>15.3f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DRL comparison of myoLeg22 models")
    parser.add_argument(
        "--task",
        choices=["stand", "walk"],
        default="stand",
        help="Task to train on (default: stand). 'stand' trains faster and "
        "is a good overnight smoke test; 'walk' is the gold standard but "
        "needs more timesteps.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps per model (default: 2M). "
        "For overnight runs, try 5M-10M for 'walk'.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs_drl/<task>)",
    )
    parser.add_argument(
        "--models",
        choices=["both", "baseline", "compliant"],
        default="both",
        help="Which model(s) to train (default: both)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only evaluate existing models in --out-dir",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs_drl" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify model files
    for p in [BASELINE_PATH, COMPLIANT_PATH]:
        if not p.exists():
            print(f"ERROR: Model not found: {p}")
            sys.exit(1)

    # ---- Training -----------------------------------------------------------
    final_models = {}

    if not args.eval_only:
        if args.models in ("both", "baseline"):
            final_models["baseline"] = train_model(
                label=f"baseline_{args.task}",
                model_path=BASELINE_PATH,
                task=args.task,
                total_timesteps=args.timesteps,
                out_dir=out_dir,
                n_envs=args.n_envs,
                seed=args.seed,
            )

        if args.models in ("both", "compliant"):
            final_models["compliant"] = train_model(
                label=f"compliant_{args.task}",
                model_path=COMPLIANT_PATH,
                task=args.task,
                total_timesteps=args.timesteps,
                out_dir=out_dir,
                n_envs=args.n_envs,
                seed=args.seed,
            )
    else:
        # Look for existing models
        for variant in ["baseline", "compliant"]:
            p = out_dir / "final_models" / f"{variant}_{args.task}" / "model.zip"
            if p.exists():
                final_models[variant] = p
            else:
                print(f"No saved model found at {p}, skipping {variant}")

    # ---- Evaluation ---------------------------------------------------------
    if len(final_models) == 0:
        print("No models to evaluate.")
        return

    results = {}
    for variant, model_zip in final_models.items():
        xml = BASELINE_PATH if variant == "baseline" else COMPLIANT_PATH
        print(f"\nEvaluating {variant} ...")
        results[variant] = evaluate_model(model_zip, xml, args.task)

    if len(results) == 2:
        print_comparison(results["baseline"], results["compliant"])

    # Save results
    results_path = out_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"TensorBoard logs: tensorboard --logdir {out_dir / 'tb_logs'}")


if __name__ == "__main__":
    main()
