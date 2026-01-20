# MACEMP Simulation on TSUBAME

This directory contains the necessary scripts to run machine learning force field simulations on the TSUBAME supercomputer without using Docker.

## 1. Environment Setup

The simulation environment is automatically created and configured when you submit the job script (`job_template.sh`) for the first time.

The script checks for the virtual environment in `~/work/python-venvs/macemp_env`. If it is missing, the job will:
1.  Create the virtual environment on the large storage (`~/work`).
2.  Install PyTorch, DGL (CUDA version), and other dependencies.
3.  Download the required `CHGNet_r2SCAN` model.

**Note:** The first run will take longer (a few minutes) due to these installation steps.

## 2. Running a Simulation

Simulations are submitted as jobs to the TSUBAME scheduler.

1.  **Customize the job script:**
    Open `job_template.sh` and edit the settings:
    *   **TSUBAME settings:** Modify the `#$` lines (e.g., `-l f_node`, `-l h_rt`).
    *   **Simulation parameters:** Change the arguments for the `python run_simulation_cli.py` command.

2.  **Submit the job:**
    Use the `qsub` command to submit your job script:
    ```bash
    qsub job_template.sh
    ```

    The results will be saved in the directory you specified with the `--output-dir` argument.

## Example

The `job_template.sh` is pre-configured to run a simulation on `BiCoO3_tetra_1244_op_25.cif` using the `CHGNet_r2SCAN` model.

```bash
# Submit the example job (Environment will be built automatically if needed)
qsub job_template.sh
```

## Note on Models

*   The scripts can use `CHGNet`, `CHGNet_r2SCAN`, `MatterSim`, `Orb`, `NequipOLM`, and `MatRIS` models.
*   If you use the `NequipOLM` model, you must place the compiled model file (`nequip-oam-l.nequip.pt2`) in a directory named `NequipOLM_model` within this (`TSUBAME`) directory. The expected path is `./NequipOLM_model/nequip-oam-l.nequip.pt2`.
*   **CHGNet_r2SCAN Model:** The `setup_env.sh` script automatically downloads this model to `./models/CHGNet_r2SCAN`. If the download fails, you can manually place the model folder there.
