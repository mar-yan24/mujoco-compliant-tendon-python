# TODO
<!-- 1. extract muscle parameter from mujoco
2. draw flv curve from mujoco
3. least square fitting -->
1. convert muscle param to mujoco compliant tendon from opensim using ls fitting
2. apply to mujoco xml using mjspec



### manual parameter fitting

```bash
python manual_fit_gui.py --muscle glmax1_r --data_dir osim_muscle_data --params_csv osim_muscle_data/all_muscle_parameters.csv
```