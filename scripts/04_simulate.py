import mujoco
import numpy as np
import imageio
import os
import csv
import matplotlib.pyplot as plt

def main():
    # 1. Define Model Path
    # Assuming repository root is parent of 'scripts'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Path based on previous file search
    model_rel_path = os.path.join("myosim_convert", "myo_sim", "leg", "myolegs_hang_compliant.xml")
    model_path = os.path.join(repo_root, model_rel_path)
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        # Fallback search or exit
        return

    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    # Disable gravity as requested
    model.opt.gravity = (0, 0, 0)
    data = mujoco.MjData(model)
    # Set resolution to FHD (1920x1080) for high quality rendering
    renderer = mujoco.Renderer(model, height=1080, width=1920)

    # 2. Identify Right-side Muscles
    # Usually ending in '_r'
    right_muscles = []
    right_muscle_ids = []
    
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.endswith('_r'):
            right_muscles.append(name)
            right_muscle_ids.append(i)
            
    print(f"Found {len(right_muscles)} Right-side muscles: {right_muscles[:5]} ...")

    # 3. Define Simulation & Capture Function
    fps = 600
    render_dt = 1.0 / fps
    n_frames = 10
    
    # Ensure output dir
    output_dir = os.path.join(repo_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 2b. Load Muscle Parameters (L_opt, L_slack) for Reference Lines & Limits
    params_csv = os.path.join(repo_root, "osim_muscle_data", "all_muscle_parameters.csv")
    muscle_params = {}
    if os.path.exists(params_csv):
        with open(params_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['muscle']
                try:
                    l_opt = float(row['optimal_fiber_length'])
                    l_slack = float(row['tendon_slack_length'])
                    muscle_params[name] = {'l_opt': l_opt, 'l_slack': l_slack}
                except ValueError:
                    continue
        print(f"Loaded reference parameters for {len(muscle_params)} muscles.")
    else:
        print(f"Warning: Parameters file not found at {params_csv}")

    def run_and_capture(segment_name, do_warmup=False, warmup_time=3.0, keyframe_name=None):
        print(f"\n[{segment_name}] Starting sequence...")
        mujoco.mj_resetData(model, data)
        
        # Apply Keyframe if requested
        if keyframe_name:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
            if key_id != -1:
                print(f"[{segment_name}] Applying keyframe '{keyframe_name}' (ID: {key_id})...")
                # Copy keyframe qpos to data.qpos
                data.qpos[:] = model.key_qpos[key_id]
            else:
                print(f"[{segment_name}] Warning: Keyframe '{keyframe_name}' not found!")
        
        mujoco.mj_forward(model, data)
        
        # Warmup Phase
        if do_warmup:
            print(f"[{segment_name}] Warming up for {warmup_time}s...")
            start_t = data.time
            while data.time - start_t < warmup_time:
                mujoco.mj_step(model, data)
            print(f"[{segment_name}] Warmup complete. Current time: {data.time:.3f}s")
            
        # Capture Phase
        print(f"[{segment_name}] Capturing {n_frames} frames...")
        frames = []
        force_history = {name: [] for name in right_muscles}
        length_history = {name: [] for name in right_muscles}
        frame_count = 0
        
        capture_start_time = data.time
        
        while frame_count < n_frames:
            # Step until next render time
            # Note: We need to maintain sync relative to capture start
            target_time = capture_start_time + (frame_count * render_dt)
            while data.time < target_time:
                mujoco.mj_step(model, data)
                
            # Render
            renderer.update_scene(data, camera=0)
            pixels = renderer.render()
            frames.append(pixels)
            
            # Record Force & Length
            for name, id in zip(right_muscles, right_muscle_ids):
                # Force
                val = data.actuator_force[id]
                force_history[name].append(val)
                # Length (MTU length)
                l_val = data.actuator_length[id]
                length_history[name].append(l_val)
                
            frame_count += 1
            
        # Save Video
        video_path = os.path.join(output_dir, f"simulation_video_{segment_name}.mp4")
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"[{segment_name}] Video saved: {video_path}")
        
        # Plot Forces
        print(f"[{segment_name}] Plotting forces...")
        n_muscles = len(right_muscles)
        n_cols = 8
        n_rows = int(np.ceil(n_muscles / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), constrained_layout=True)
        axes = axes.flatten()
        x_axis = np.arange(n_frames)
        
        # Calculate global max force for unified scaling
        all_max_forces = []
        for name in right_muscles:
            f_vals = -np.array(force_history[name])
            all_max_forces.append(np.max(f_vals) if len(f_vals) > 0 else 0)
        
        global_max = max(all_max_forces) if all_max_forces else 0
        # Unified top limit: at least 10, otherwise global max + 5% padding
        unified_top = max(global_max * 1.05, 10)
        
        for i, name in enumerate(right_muscles):
            ax = axes[i]
            forces = np.array(force_history[name])
            line, = ax.plot(x_axis, -forces, label=name)
            ax.fill_between(x_axis, 0, -forces, color=line.get_color(), alpha=0.4)
            
            # Set unified Y-axis limit
            ax.set_ylim(bottom=0, top=unified_top)
                
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Force (N)")
            ax.grid(True, alpha=0.3)
            
        for i in range(n_muscles, len(axes)):
            axes[i].axis('off')
            
        plot_path = os.path.join(output_dir, f"muscle_forces_{segment_name}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[{segment_name}] Plot saved: {plot_path}")

        # Plot Lengths
        print(f"[{segment_name}] Plotting lengths...")
        # Use same grid layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), constrained_layout=True)
        axes = axes.flatten()
        
        for i, name in enumerate(right_muscles):
            ax = axes[i]
            lens = np.array(length_history[name])
            ax.plot(x_axis, lens, label=name, color='tab:orange')
            
            # Add L_opt + L_slack reference line & Set Limit
            if name in muscle_params:
                p = muscle_params[name]
                l_opt = p['l_opt']
                l_slack = p['l_slack']
                
                ref_val = l_opt + l_slack
                ax.axhline(ref_val, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
                
                # Set requested limits: min = l_slack, max = l_slack + 2.0 * l_opt
                ax.set_ylim(bottom=l_slack, top=l_slack + 2.0 * l_opt)
            
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Length (m)")
            ax.grid(True, alpha=0.3)
            
        for i in range(n_muscles, len(axes)):
            axes[i].axis('off')
            
        plot_path = os.path.join(output_dir, f"muscle_lengths_{segment_name}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[{segment_name}] Length plot saved: {plot_path}")

        # Print Analysis for top forces (Explaining passive forces)
        print(f"\n[{segment_name}] Top 5 Forces at final frame:")
        final_forces = []
        for name in right_muscles:
            # force_history[name] may be empty if no frames? No, loop ensures frames.
            if force_history[name]:
                f = force_history[name][-1]
                final_forces.append((name, f))
        
        # Sort descending by force
        final_forces.sort(key=lambda x: x[1], reverse=True)
        
        for name, f in final_forces[:5]:
            info = f"{name}: {f:.2f} N"
            if name in muscle_params:
                p = muscle_params[name]
                if length_history[name]:
                    l_cur = length_history[name][-1]
                    l_sla = p['l_slack']
                    l_opt = p['l_opt']
                    # Calculate strain relative to l_opt
                    strain = (l_cur - l_sla) / l_opt if l_opt > 0 else 0
                    info += f" | L_mtu={l_cur:.4f} m, L_slack={l_sla:.4f} m, Normalized Strain={strain:.2%}"
            print(f"  - {info}")

    # 4. Execute Scenarios
    # Scenario A: Initial State (Directly capture)
    run_and_capture("initial", do_warmup=False)
    
    # Scenario B: Converged State (Warmup then capture)
    run_and_capture("converged", do_warmup=True, warmup_time=3.0)

    # Scenario C: Slight Knee Flexion (Initial)
    run_and_capture("knee_flex_initial", keyframe_name="slight_knee_flexion_only", do_warmup=False)
    
    # Scenario D: Slight Knee Flexion (Converged)
    run_and_capture("knee_flex_converged", keyframe_name="slight_knee_flexion_only", do_warmup=True, warmup_time=3.0)

if __name__ == "__main__":
    main()
