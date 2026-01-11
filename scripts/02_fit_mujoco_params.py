# %%
import sys

# Check Python version (MuJoCo and modern libraries require Python 3.11+)
if sys.version_info < (3, 11):
    print(f"ERROR: This script requires Python 3.11 or higher, but you are using {sys.version_info.major}.{sys.version_info.minor}")
    print("MuJoCo and modern scientific libraries require Python 3.11+.")
    print("Please activate the correct virtual environment (venv_mujoco) and try again.")
    sys.exit(1)

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import minimize, least_squares
import time
from tqdm import tqdm


# %%
# ==========================================
# 1. Class Definitions
# ==========================================
class CompliantTendonParams:
    def __init__(self, F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF):
        self.F_max = F_max
        self.l_opt = l_opt
        self.l_slack = l_slack
        self.v_max = v_max
        self.W = W
        self.C = C
        self.N = N
        self.K = K
        self.E_REF = E_REF
    
    def get_prm_str(self):
        # Validate all parameters before creating string
        params = [self.F_max, self.l_opt, self.l_slack, self.v_max, self.W, self.C, self.N, self.K, self.E_REF]
        for param in params:
            if np.isnan(param) or np.isinf(param):
                raise ValueError(f"Invalid parameter value (NaN or Inf) in CompliantTendonParams")
        return f"{self.F_max} {self.l_opt} {self.l_slack} {self.v_max} {self.W} {self.C} {self.N} {self.K} {self.E_REF}"


# %%
# ==========================================
# 2. MuJoCo Model Generation
# ==========================================
def create_model(cp_params: CompliantTendonParams):
    xml_string = f"""
    <mujoco model="fitting_scene">
    <option timestep="0.002" integrator="Euler"/>

    <default>
        <default class="compliant_muscle">
        <general biasprm="0" biastype="none" ctrllimited="true" ctrlrange="0 1" 
                dynprm="0.01 0.04" dyntype="muscle" 
                gainprm="{cp_params.get_prm_str()}"
                gaintype="compliant_mtu"/>
        </default>
    </default>

    <worldbody>
        <body name="ground"/>
        <site name="anchor" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>

        <body name="load" pos="0 0 0">
            <joint name="slide" type="slide" axis="0 0 1" limited="false" damping="0"/> 
            <site name="insertion" pos="0 0 0" size="0.01" rgba="0 1 0 1"/>
            <geom type="sphere" size="0.05" mass="1.0"/>
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
    return mujoco.MjModel.from_xml_string(xml_string)


# %%
# ==========================================
# 3. Simulation Function
# ==========================================
def compute_forces_at_velocity(model, data, velocity, lengths, activation=1.0):
    """
    Compute forces for a specific velocity across multiple lengths.
    Uses Kinematic Drive (mj_forward) to efficiently calculate steady-state forces.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data (reused)
        velocity: Physical velocity (m/s)
        lengths: List/Array of physical lengths (m)
        activation: Muscle activation (0.0-1.0)
        
    Returns:
        np.array of forces corresponding to 'lengths'
    """

    model.opt.timestep = 1/1200
    mujoco.mj_resetData(model, data)
    forces = []
    
    for length in lengths:
        for internal_it in range(1):
            data.qvel[0] = velocity
            data.act[0] = activation
            data.ctrl[0] = activation
            
            # Update length (qpos)
            # Ensure non-negative length if physical constraints require
            if length < 0.001: length = 0.001
            data.qpos[0] = -length
            
            # Compute dynamics
            # mj_forward computes position-dependent forces (active+passive) given qpos, qvel, act
            # mujoco.mj_forward(model, data)


            # mujoco.mj_fwdVelocity(model, data)
            # mujoco.mj_fwdVelocity(model, data)
            # mujoco.mj_fwdActuation(model, data)
            # mujoco.mj_kinematics(model, data)
            mujoco.mj_forward(model, data)

            # print(f"{data.ten_length[0]=}, {data.qpos[0]=}")

        
        
        # MuJoCo actuatorfrc sign can be opposite (contractile pull shows as negative).
        # Flip sign to store tendon tensile force as positive.
        forces.append(data.qfrc_actuator[0])
        
    return np.array(forces)


# %%
# ==========================================
# 4. Data Loading & Preprocessing
# ==========================================
def load_length_force_sim(muscle_name, params_csv, data_dir):
    if not os.path.exists(params_csv):
        raise FileNotFoundError(f"Parameter CSV not found: {params_csv}. Please supply the parameter CSV exported from extract_muscle_force_sim outputs.")
    # parameters from reference CSV
    p = None
    with open(params_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['muscle'] == muscle_name:
                p = row
                break
    if p is None:
        raise ValueError(f"Muscle {muscle_name} not found in parameters CSV.")

    f_max = float(p['max_isometric_force'])
    l_opt = float(p['optimal_fiber_length'])
    l_slack = float(p['tendon_slack_length'])
    v_max = float(p['max_contraction_velocity'])

    csv_path_total = os.path.join(data_dir, f"{muscle_name}_sim_total.csv")
    if not os.path.exists(csv_path_total):
        raise FileNotFoundError(f"Data file {csv_path_total} not found.")

    csv_path_passive = os.path.join(data_dir, f"{muscle_name}_sim_passive.csv")
    if not os.path.exists(csv_path_passive):
        # Fallback if passive file doesn't exist (though it should from 01 script)
        print(f"Warning: Passive force file {csv_path_passive} not found. Using total force for range finding logic might be inaccurate.")
        csv_path_passive = None

    def read_sim_data(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # rows[0] is header (velocities)
        # rows[1:] is data
        _lengths = []
        _matrix = []
        for r in rows[1:]:
            _lengths.append(float(r[0]))
            _matrix.append([float(x) for x in r[1:]])
        return np.array(_lengths), np.array(_matrix), np.array([float(v) for v in rows[0][1:]])

    mtu_lengths, total_force_matrix, norm_velocities = read_sim_data(csv_path_total)
    
    if csv_path_passive:
        _, passive_force_matrix, _ = read_sim_data(csv_path_passive)
    else:
        passive_force_matrix = None

    return {
        "f_max": f_max,
        "l_opt": l_opt,
        "l_slack": l_slack,
        "v_max": v_max,
        "mtu_lengths": mtu_lengths,
        "norm_velocities": norm_velocities,
        "force_matrix": total_force_matrix,
        "passive_force_matrix": passive_force_matrix
    }


# %%
# ==========================================
# 5. Fitting Logic (All Parameters)
# ==========================================
# Global variables for progress tracking
_obj_iter_count = 0
_obj_start_time = None
_obj_verbose = 1

def get_fitting_range(target_data):
    """
    Calculate fitting range based on Tendon linear transition point using Passive Force.
    
    Rationale:
    According to the OpenSim API Documentation for TendonForceLengthCurve:
    https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1TendonForceLengthCurve.html
    
    The default parameters for the Millard 2012 Equilibrium Muscle model are fitted to match 
    average in-vivo tendon curves reported by Maganaris et al. and Magnusson et al.
    Key properties include:
      - strainAtOneNormForce: 0.049 (4.9% strain at F_max)
      - normForceAtToeEnd: 2.0/3.0 (Linear transition occurs at ~67% of F_max)
      - stiffnessAtOneNormForce: 1.375 / strainAtOneNormForce
    
    We limit the fitting range up to this linear transition point (approx 2/3 * F_max) 
    to accurately capture the "toe region" characteristics where the tendon is compliant.
    We specifically use the extracted PASSIVE force data (a=0) to identify this threshold,
    avoiding contamination from active muscle force peaks at shorter lengths.
    """
    l_slack = target_data['l_slack']
    f_max = target_data['f_max']
    
    # Min length: starts at slack length
    min_len = l_slack
    
    # Max length: based on PASSIVE force reaching 2/3 F_max.
    # We use the passive force matrix if available.
    if target_data.get('passive_force_matrix') is not None:
        forces = target_data['passive_force_matrix'][:, 0]
    else:
        # Fallback to total force if passive not loaded (should not happen with updated code)
        forces = target_data['force_matrix'][:, 0]

    mtu_lengths = target_data['mtu_lengths']
    limit_force = (2.0 / 3.0) * f_max
    
    # Since we are using PASSIVE force (specifically extracted where a=0),
    # we don't need heuristics to skip active peaks. The passive curve is monotonic.
    # Just find the first point where force > limit.
    
    over_indices = np.where(forces > limit_force)[0]
    
    if len(over_indices) > 0:
        idx = over_indices[0]
        max_len = mtu_lengths[idx]
    else:
        # If force never reaches the limit (range too short), use full range
        max_len = mtu_lengths[-1]
            
    return min_len, max_len

def plot_fitting_range_on_ax(ax, min_len, max_len, label='Fit Range'):
    """
    Helper to plot fitting range on a matplotlib axis.
    """
    ax.axvspan(min_len, max_len, color='green', alpha=0.1, label=label)
    ax.axvline(min_len, color='g', linestyle='--', alpha=0.5)
    ax.axvline(max_len, color='g', linestyle='--', alpha=0.5)

def objective_function(x, target_data, verbose=1):
    global _obj_iter_count, _obj_start_time, _obj_verbose
    
    iter_start_time = time.time()
    
    # Unpack 9 parameters: F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF
    F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF = x
    
    # Validate parameters: check for NaN, Inf, or invalid values
    params = [F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF]
    param_names = ['F_max', 'l_opt', 'l_slack', 'v_max', 'W', 'C', 'N', 'K', 'E_REF']
    
    for param_val, param_name in zip(params, param_names):
        if np.isnan(param_val) or np.isinf(param_val):
            if verbose >= 1:
                print(f"\n[Objective #{_obj_iter_count}] Invalid parameter: {param_name}={param_val}")
            raise ValueError(f"Invalid parameter: {param_name}={param_val}")

    # Check physical constraints
    if F_max <= 0 or l_opt <= 0 or l_slack <= 0 or v_max <= 0:
        if verbose >= 1:
            print(f"\n[Objective #{_obj_iter_count}] Invalid physical parameters: F_max={F_max}, l_opt={l_opt}, l_slack={l_slack}, v_max={v_max}")
        raise ValueError(f"Invalid physical parameters: F_max={F_max}, l_opt={l_opt}, l_slack={l_slack}, v_max={v_max}")

    ref_f_max = target_data['f_max']
    ref_l_opt = target_data['l_opt']
    ref_l_slack = target_data['l_slack']

    mtu_lengths = target_data['mtu_lengths']  # Actual MTU lengths (m)
    norm_velocities = target_data['norm_velocities']
    force_matrix = target_data['force_matrix']

    # Define fitting range
    min_len, max_len = get_fitting_range(target_data)
    
    # Filter data indices
    range_mask = (mtu_lengths >= min_len) & (mtu_lengths <= max_len)
    if not np.any(range_mask):
         # If no data in range (unlikely), warn and use all? 
         # Or just expand slightly? Let's assume there is data or use all if empty.
         if verbose >= 1:
             print(f"[Warning] No data points in range [{min_len:.4f}, {max_len:.4f}]. Using all data.")
         range_mask = np.ones_like(mtu_lengths, dtype=bool)

    # Apply mask to data
    target_l_phys = mtu_lengths[range_mask]
    target_force_matrix_subset = force_matrix[range_mask, :]
    
    _obj_iter_count += 1
    if _obj_start_time is None:
        _obj_start_time = time.time()
    
    total_elapsed = time.time() - _obj_start_time
    
    if verbose >= 1:
        print(f"\n[Objective #{_obj_iter_count}] Starting evaluation...")
        print(f"  Params: F_max={F_max:.2f}, l_opt={l_opt:.4f}, l_slack={l_slack:.4f}, v_max={v_max:.2f}")
        print(f"          W={W:.3f}, C={C:.3f}, N={N:.3f}, K={K:.3f}, E_REF={E_REF:.4f}")
        print(f"  Total elapsed: {total_elapsed:.1f}s")
    
    if verbose >= 2:
        print(f"  Creating MuJoCo model...")
    
    cp_params = CompliantTendonParams(
        F_max=F_max,
        l_opt=l_opt,
        l_slack=l_slack,
        v_max=v_max,
        W=W, C=C, N=N, K=K, E_REF=E_REF
    )
    
    model = create_model(cp_params)
    data = mujoco.MjData(model) # Create data once per objective call
    
    model_time = time.time() - iter_start_time
    
    if verbose >= 2:
        print(f"  Model created in {model_time:.3f}s")
    
    total_error = 0
    count = 0
    
    # Sampling: Use all length points, fixed v=0
    l_indices = range(len(mtu_lengths))
    
    # Find index of v=0 in norm_velocities for target data extraction
    v0_idx = np.argmin(np.abs(norm_velocities))
    
    total_points = len(l_indices)
    
    if verbose >= 1:
        print(f"  Evaluating {total_points} points (All lengths, fixed v=0)...")
    
    sim_start_time = time.time()
    
    # Target forces for v=0 profile across all lengths
    # force_matrix shape is (n_lengths, n_velocities)
    f_target_profile = target_force_matrix_subset[:, v0_idx]
    
    # Compute simulated forces for v=0 across all target lengths
    v_phy = 0.0 # Fixed v=0
    f_sim_profile = compute_forces_at_velocity(model, data, v_phy, target_l_phys, activation=1.0)
    
    # Calculate residuals and filter out pathological simulation points
    residuals_raw = f_sim_profile - f_target_profile

    # Base validity: finite and not excessively large (>|5 * F_max|)
    valid_mask = np.isfinite(f_sim_profile) & (np.abs(f_sim_profile) <= 5.0 * ref_f_max)

    # Discontinuity filter: mark points following a large jump as invalid
    if f_sim_profile.size > 1:
        jump_thresh = 0.1 * ref_f_max
        jumps = np.abs(np.diff(f_sim_profile)) > jump_thresh
        bad_indices = np.where(jumps)[0] + 1  # mark the point after the jump
        valid_mask[bad_indices] = False

    # Ensure fixed-length residual vector for least_squares:
    # keep original length, but penalize invalid points heavily.
    if not np.any(valid_mask):
        raise ValueError("All simulated points were filtered out (non-finite, over-limit, or discontinuous).")

    penalty = 5.0 * ref_f_max
    residuals_penalized = residuals_raw.copy()
    residuals_penalized[~valid_mask] = np.sign(residuals_penalized[~valid_mask]) * penalty

    all_residuals = residuals_penalized  # fixed length
    mse = np.mean(all_residuals**2)
    
    sim_time = time.time() - sim_start_time
    
    sim_time = time.time() - sim_start_time
    iter_time = time.time() - iter_start_time
    
    if verbose >= 1:
        print(f"  Simulation completed: {sim_time:.2f}s")
        print(f"  Residuals: {all_residuals}")

        print(f"  MSE: {mse:.6f}")
        print(f"  Total iteration time: {iter_time:.2f}s")
        if _obj_iter_count > 1:
            avg_iter_time = total_elapsed / _obj_iter_count
            print(f"  Average iteration time: {avg_iter_time:.2f}s")
    
    return all_residuals


# %%
# ==========================================
# 6. Plotting Function
# ==========================================
def plot_results(best_params, target_data, muscle_name, initial_params=None):
    print("\n[Plotting] Generating length-only plot (v=0)...")
    F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF = best_params
    cp_params = CompliantTendonParams(
        F_max=F_max, l_opt=l_opt, l_slack=l_slack, v_max=v_max,
        W=W, C=C, N=N, K=K, E_REF=E_REF
    )
    model = create_model(cp_params)
    data = mujoco.MjData(model)

    mtu_lengths = target_data['mtu_lengths']
    dense_L_phy = np.linspace(mtu_lengths.min(), mtu_lengths.max(), 60)

    # Only v=0, activation=1.0
    f_sim = compute_forces_at_velocity(model, data, 0.0, dense_L_phy, activation=1.0)
    f_data = target_data['force_matrix'][:, 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot fitting range
    min_len, max_len = get_fitting_range(target_data)
    
    # Highlight the fitting range
    plot_fitting_range_on_ax(ax, min_len, max_len, label='Fit Range')

    ax.plot(mtu_lengths, f_data, "k.", label="data v=0")
    ax.plot(dense_L_phy, f_sim, "r-", label="fit v=0")
    ax.set_title(f"{muscle_name} length-force (v=0)")
    ax.set_xlabel("MTU length (m)")
    ax.set_ylabel("Force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small")
    
    # Add fitted parameters as text at the bottom
    param_text = f"Fitted: F_max={F_max:.2f}, l_opt={l_opt:.2f}, l_slack={l_slack:.2f}, v_max={v_max:.2f}\n"
    param_text += f"W={W:.2f}, C={C:.2f}, N={N:.2f}, K={K:.2f}, E_REF={E_REF:.2f}"
    
    if initial_params is not None:
        F0, l0, ls0, v0, W0, C0, N0, K0, E0 = initial_params
        param_text += f"\nInitial: F_max={F0:.2f}, l_opt={l0:.2f}, l_slack={ls0:.2f}, v_max={v0:.2f}\n"
        param_text += f"W={W0:.2f}, C={C0:.2f}, N={N0:.2f}, K={K0:.2f}, E_REF={E0:.2f}"
    
    ax.text(0.5, -0.25, param_text, transform=ax.transAxes, 
            fontsize=7, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    os.makedirs("mujoco_muscle_data", exist_ok=True)
    out_path = os.path.join("mujoco_muscle_data", f"{muscle_name}_fit_v0.png")
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])  # Leave more space at bottom
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Plotting] Saved: {out_path}")


# %%
# ==========================================
# 7. Main Execution
# ==========================================
def callback_function(xk, state=None):
    """Callback function to show optimization progress"""
    global _obj_iter_count
    if _obj_iter_count > 0:
        print(f"\n[Callback] Optimization step completed")
        print(f"  Current params:")
        print(f"    F_max={xk[0]:.2f}, l_opt={xk[1]:.4f}, l_slack={xk[2]:.4f}, v_max={xk[3]:.2f}")
        print(f"    W={xk[4]:.4f}, C={xk[5]:.4f}, N={xk[6]:.4f}, K={xk[7]:.4f}, E_REF={xk[8]:.4f}")
    return False

def fit_muscle(muscle_name, data_dir="osim_muscle_data", params_csv="osim_muscle_data/all_muscle_parameters.csv", verbose=1):
    """
    Fit muscle parameters
    
    Args:
        muscle_name: Name of the muscle to fit
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)
    """
    global _obj_iter_count, _obj_start_time, _obj_verbose
    
    _obj_verbose = verbose
    _obj_iter_count = 0
    _obj_start_time = None
    
    print(f"=" * 60)
    print(f"Starting fit for {muscle_name}")
    print(f"=" * 60)
    
    param_csv = params_csv
    data_dir = data_dir
    
    try:
        print(f"[Loading] Reading data from {data_dir}...")
        target_data = load_length_force_sim(muscle_name, param_csv, data_dir)
        print(f"[Loading] Data loaded successfully!")
        print(f"  - F_max: {target_data['f_max']:.2f}")
        print(f"  - l_opt: {target_data['l_opt']:.4f}")
        print(f"  - l_slack: {target_data['l_slack']:.4f}")
        print(f"  - v_max: {target_data['v_max']:.2f}")
        print(f"  - Data shape: {target_data['force_matrix'].shape}")
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] Value error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Initial Guess: [F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF]
    # v_max is kept fixed (not optimized) for length-only fitting
    base_F = target_data['f_max']
    base_L_opt = target_data['l_opt']
    base_L_slack = target_data['l_slack']
    base_V_max = target_data['v_max']
    
    x0 = [
        base_F, 
        base_L_opt * 0.95, 
        base_L_slack* 1.0, 
        base_V_max,  # fixed
        0.56, -2.995732274, 1.5, 5.0, 0.04
    ]
    
    # Bounds: v_max fixed; others allowed to vary
    bounds = [
        # (base_F, base_F),          # F_max
        # (base_L_opt, base_L_opt),  # l_opt
        # (base_L_slack, base_L_slack), # l_slack
        # (base_V_max, base_V_max),              # v_max fixed

        (base_F * 0.5, base_F * 1.5),          # F_max
        (base_L_opt * 0.85, base_L_opt * 1.05),  # l_opt
        (base_L_slack * 0.95, base_L_slack * 1.05), # l_slack
        (base_V_max, base_V_max + 1e-7),              # v_max fixed

        
        # (0.56, 0.56 + 1e-7),                            # W = 0.56
        # (-2.995732274, -2.995732274 + 1e-7),                          # C = -2.995732274
        # (1.5, 1.5 + 1e-7),                            # N = 1.5
        # (5.0, 5.0 + 1e-7),                            # K = 5.0
        # (0.04, 0.04 + 1e-7)                           # E_REF = 0.04

        (0.01, 10),                            # W = 0.56
        (-100, -0.01),                          # C = -2.995732274
        (1.5, 1.5 + 1e-7),                            # N = 1.5
        (5.0, 5.0 + 1e-7),                            # K = 5.0
        (0.01, 1.0)                           # E_REF = 0.04
    ]
    
    # print(f"\n[Optimization] Starting optimization...")
    # print(f"  - Optimizing 9 parameters: F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF")
    # print(f"  - Initial guess: {x0}")
    # print(f"  - Max iterations: 50")
    # print(f"  - Verbose level: {verbose}")
    
    _obj_iter_count = 0
    _obj_start_time = time.time()
    opt_start_time = time.time()
    
    # Wrapper function for objective with progress
    def obj_wrapper(x):
        return objective_function(x, target_data, verbose=verbose)
    
    # Convert bounds for least_squares: ([min_vals], [max_vals])
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    ls_bounds = (lower_bounds, upper_bounds)

    res = least_squares(
        obj_wrapper,
        x0,
        bounds=ls_bounds,
        max_nfev=10000,  # Increased max evaluations
        verbose=2 if verbose >= 1 else 0, # Increased verbosity
        jac='3-point',  # More accurate Jacobian approximation (slower but more stable)
        # loss='soft_l1', # Robust loss function to handle outliers/noise better than linear (least squares)
        ftol=1e-6,     # Tighter tolerance for function value change
        xtol=1e-6,     # Tighter tolerance for independent variable change
        gtol=1e-6      # Tighter tolerance for gradient norm
    )
    
    opt_time = time.time() - opt_start_time
    print(f"\n[Optimization] Finished in {opt_time:.2f}s")
    print(f"  - Status: {res.message}")
    print(f"  - Success: {res.success}")
    # res.cost is 0.5 * sum(residuals**2)
    # We want MSE = mean(residuals**2)
    final_mse = np.mean(res.fun**2)
    print(f"  - Final MSE: {final_mse:.6f}")
    if hasattr(res, 'nfev'):
        print(f"  - Iterations (nfev): {res.nfev}")
    else:
        print(f"  - Iterations: N/A")
    print(f"\n[Optimization] Fitted Parameters:")
    print(f"  - F_max: {res.x[0]:.2f} (initial: {x0[0]:.2f})")
    print(f"  - l_opt: {res.x[1]:.4f} (initial: {x0[1]:.4f})")
    print(f"  - l_slack: {res.x[2]:.4f} (initial: {x0[2]:.4f})")
    print(f"  - v_max: {res.x[3]:.2f} (initial: {x0[3]:.2f})")
    print(f"  - W: {res.x[4]:.4f}")
    print(f"  - C: {res.x[5]:.4f}")
    print(f"  - N: {res.x[6]:.4f}")
    print(f"  - K: {res.x[7]:.4f}")
    print(f"  - E_REF: {res.x[8]:.4f}")
    
    if not res.success:
        raise ValueError(f"Optimization did not converge successfully! {res.message}")
    
    print(f"\n[Plotting] Starting to generate plots...")
    sys.stdout.flush()

    print(f"[Plotting] Calling plot_results with fitted parameters...")
    sys.stdout.flush()
    plot_results(res.x, target_data, muscle_name, initial_params=x0)
    print(f"[Plotting] Plot saved (no plt.show).")
    sys.stdout.flush()
    
    print(f"\n[Complete] Fitting finished successfully!")
    return res.x


# %%
def fit_all_muscles_length_only(data_dir="osim_muscle_data",
                                params_csv="osim_muscle_data/all_muscle_parameters.csv",
                                verbose=0,
                                out_param_csv="mujoco_muscle_data/fitted_params_length_only.csv",
                                plot_path="mujoco_muscle_data/fitted_length_force_all.png"):
    files = [f for f in os.listdir(data_dir) if f.endswith("_sim_total.csv")]
    muscles = [f.replace("_sim_total.csv", "") for f in files]
    muscles.sort() # Ensure consistent alphabetical order
    fitted_rows = []

    # figure grid
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        plt = None
        print(f"Plot import failed: {e}")

    n = len(muscles)
    ncols = 8
    nrows = int(np.ceil(n / ncols)) if n > 0 else 0
    if plt is not None and n > 0:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for idx, mname in enumerate(muscles):
        print(f"\n=== Fitting {mname} ===")
        res = fit_muscle(mname, data_dir=data_dir, params_csv=params_csv, verbose=verbose)
        if res is None:
            continue
        fitted_rows.append([mname] + list(res))

        if plt is not None:
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]
            # load data and simulate fitted curve at v=0
            target = load_length_force_sim(mname, params_csv, data_dir)
            cp_params = CompliantTendonParams(*res)
            model = create_model(cp_params)
            data = mujoco.MjData(model)
            
            # Use dense spacing for smooth curve, similar to individual plots
            mtu_lengths = target["mtu_lengths"]
            dense_L_phy = np.linspace(mtu_lengths.min(), mtu_lengths.max(), 100)
            
            v_phy = 0.0
            f_sim = compute_forces_at_velocity(model, data, v_phy, dense_L_phy, activation=1.0)
            
            # Plot fitting range
            min_len, max_len = get_fitting_range(target)

            plot_fitting_range_on_ax(ax, min_len, max_len, label='Range')

            ax.plot(mtu_lengths, target["force_matrix"][:,0], "k.", label="data", markersize=3)
            ax.plot(dense_L_phy, f_sim, "r-", label="fit", linewidth=1)
            ax.set_title(mname)
            ax.set_xlabel("MTU length (m)")
            ax.set_ylabel("Force (N)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="x-small")
            
            # Calculate initial guess (same as in fit_muscle)
            base_F = target['f_max']
            base_L_opt = target['l_opt']
            base_L_slack = target['l_slack']
            base_V_max = target['v_max']
            x0 = [
                base_F, 
                base_L_opt, 
                base_L_slack, 
                base_V_max,
                0.56, -2.995732274, 1.5, 5.0, 0.04
            ]
            
            # Add fitted parameters as text at the bottom
            F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF = res
            F0, l0, ls0, v0, W0, C0, N0, K0, E0 = x0
            param_text = f"Fit: F={F_max:.2f}, l_o={l_opt:.2f}, l_s={l_slack:.2f}\n"
            param_text += f"W={W:.2f}, C={C:.2f}, N={N:.2f}, K={K:.2f}, E={E_REF:.2f}\n"
            param_text += f"Init: F={F0:.2f}, l_o={l0:.2f}, l_s={ls0:.2f}\n"
            param_text += f"W={W0:.2f}, C={C0:.2f}, N={N0:.2f}, K={K0:.2f}, E={E0:.2f}"
            ax.text(0.5, -0.25, param_text, transform=ax.transAxes, 
                    fontsize=5, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # write params CSV
    if fitted_rows:
        out_dir_params = os.path.dirname(out_param_csv)
        if out_dir_params:
            os.makedirs(out_dir_params, exist_ok=True)
        header = ["muscle","F_max","l_opt","l_slack","v_max","W","C","N","K","E_REF"]
        with open(out_param_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(fitted_rows)
        print(f"Saved fitted parameters: {out_param_csv}")

    if plt is not None and n > 0:
        # hide empty subplots
        for k in range(n, nrows*ncols):
            row = k // ncols
            col = k % ncols
            fig.delaxes(axes[row][col])
        # Leave more space at bottom for parameter text and increase spacing between subplots
        fig.subplots_adjust(bottom=0.20, hspace=0.6, wspace=0.4)
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_path, dpi=200)
        print(f"Saved comparison plot: {plot_path}")
        # Show only the final combined figure (per request)
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)


# Run fitting for all muscles with v=0 data
if __name__ == "__main__":
    fit_all_muscles_length_only(verbose=0)

    # Post-processing: Apply params to compliant XML
    import subprocess
    import shutil
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_03 = os.path.join(repo_root, "scripts", "03_apply_fitted_params.py")
    
    src_xml = os.path.join(repo_root, "myosim_convert", "myo_sim", "leg", "assets", "myolegs_muscle_rigid.xml")
    target_xml = os.path.join(repo_root, "myosim_convert", "myo_sim", "leg", "assets", "myolegs_muscle_compliant.xml")
    csv_path = os.path.join(repo_root, "mujoco_muscle_data", "fitted_params_length_only.csv")
    
    # 1. Ensure compliant XML exists (copy from base to ensure fresh structure)
    print(f"\n[Apply] Creating/Overwriting {os.path.basename(target_xml)} from base...")
    shutil.copy2(src_xml, target_xml)
    
    # 2. Run 03 script to update it
    print(f"[Apply] Running 03_apply_fitted_params.py to inject parameters...")
    # Args: [xml_path] [csv_path] [out_path]
    cmd = [sys.executable, script_03, target_xml, csv_path, target_xml]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"[Apply] Successfully updated {target_xml}")
    else:
        print("Error running script 03:")
        print(result.stderr)
