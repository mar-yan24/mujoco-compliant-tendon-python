# MuJoCo Compliant Tendon - Python

This project converts muscle parameters from OpenSim to MuJoCo's compliant tendon model using least squares fitting.

## Project Structure

```
├── scripts/                           # Executable scripts (ordered by workflow)
│   ├── 01_extract_opensim_data.py    # Extract muscle force data from OpenSim
│   ├── 02_fit_mujoco_params.py       # Fit MuJoCo parameters to OpenSim data
│   ├── 02a_manual_fitting_gui.py      # (optional) Interactive manual parameter fitting GUI
│   └── 03_apply_fitted_params.py     # Apply fitted parameters to MuJoCo XML
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── compliant_graph.ipynb
│   ├── compliant_tendon.ipynb
│   ├── myoleg_fitting.ipynb
│   ├── myoleg_muscle_param.ipynb
│   └── opensim_muscle_param.ipynb
│
├── osim_muscle_data/                  # OpenSim extracted data (CSVs & plots)
├── mujoco_muscle_data/                # MuJoCo fitted parameters (CSVs & plots)
├── opensim_models/                    # OpenSim model files
├── myosim_convert/                    # Model conversion files
│
└── outputs/                           # Media outputs
    ├── videos/                        # Simulation videos
    └── images/                        # Output images
```

## Workflow

### 1. Extract OpenSim Muscle Data

Extract muscle force-length data from OpenSim models:

```bash
python scripts/01_extract_opensim_data.py
```

This script:
- Loads OpenSim muscle models from `opensim_models/`
- Extracts muscle parameters (F_max, l_opt, l_slack, v_max)
- Simulates length-force curves at v=0
- Saves data to `osim_muscle_data/`:
  - `all_muscle_parameters.csv` - Base muscle parameters
  - `{muscle_name}_sim_*.csv` - Force-length data
  - `{muscle_name}_lf_v0.png` - Individual muscle plots
  - Combined visualization plots

### 2. Fit MuJoCo Parameters

Fit MuJoCo compliant tendon parameters to match OpenSim data:

```bash
python scripts/02_fit_mujoco_params.py
```

This script:
- Reads OpenSim data from `osim_muscle_data/`
- Uses least squares optimization to fit 9 MuJoCo parameters:
  - `F_max` - Maximum isometric force
  - `l_opt` - Optimal fiber length
  - `l_slack` - Tendon slack length
  - `v_max` - Maximum contraction velocity
  - `W, C, N, K, E_REF` - Compliant tendon model parameters
- Saves results to `mujoco_muscle_data/`:
  - `fitted_params_length_only.csv` - Fitted parameters for all muscles
  - `{muscle_name}_fit_v0.png` - Individual fit plots
  - `fitted_length_force_all.png` - Combined comparison plot

### 2a. Manual Parameter Tuning (Optional)

Interactively adjust parameters using sliders:

```bash
python scripts/02a_manual_fitting_gui.py --muscle glmax1_r \
    --data_dir osim_muscle_data \
    --params_csv osim_muscle_data/all_muscle_parameters.csv
```

Features:
- Real-time visualization of parameter changes
- 9 adjustable sliders for all compliant tendon parameters
- Print current parameters to console

### 3. Apply Parameters to MuJoCo Model

Apply the fitted parameters to your MuJoCo XML file:

```bash
python scripts/04_apply_fitted_params.py [xml_path] [csv_path] [out_path]
```

Default paths (when run from repo root):
- `xml_path`: `myosim_convert/myo_sim/leg/assets/myolegs_muscle.xml`
- `csv_path`: `mujoco_muscle_data/fitted_params_length_only.csv`
- `out_path`: Overwrites xml_path unless specified

The script:
- Updates `gainprm` with fitted parameters
- Sets `gaintype` to "compliant_mtu"
- Mirrors left/right muscle pairs automatically
- Preserves other attributes (e.g., lengthrange)

## Requirements

```bash
pip install -r requirements.txt
```

For OpenSim-specific functionality:
```bash
pip install -r requirements_opensim.txt
```

## Key Data Files

**Most Important:**
- `osim_muscle_data/all_muscle_parameters.csv` - OpenSim muscle parameters
- `osim_muscle_data/{muscle}_sim_total.csv` - OpenSim force-length data
- `mujoco_muscle_data/fitted_params_length_only.csv` - **Final fitted MuJoCo parameters**

## Notes

- Scripts should be run from the repository root directory
- The fitting process focuses on length-force curves at v=0 (isometric)
- All right-side muscles (_r) are fitted; left-side muscles are mirrored
- Output folders (`osim_muscle_data/`, `mujoco_muscle_data/`) are created automatically
