# MACEMP Simulation on TSUBAME

This directory contains the necessary scripts to run machine learning force field simulations on the TSUBAME supercomputer without using Docker.

## 1. Environment Setup

First, you need to create a dedicated Python virtual environment. The provided script will install all necessary packages.

1.  **Run the setup script:**
    ```bash
    bash setup_env.sh
    ```
    This will create a virtual environment in `~/python-venvs/macemp_env` and install the required libraries. You only need to do this once.

2.  **Activate the environment:**
    Before running any simulation, you must activate the environment in your shell:
    ```bash
    source ~/python-venvs/macemp_env/bin/activate
    ```

## 2. Running a Simulation

Simulations are submitted as jobs to the TSUBAME scheduler. A template job script `job_template.sh` is provided.

1.  **Customize the job script:**
    Open `job_template.sh` and edit the settings:
    *   **TSUBAME settings:** Modify the `#$` lines (e.g., `-l f_node`, `-l h_rt`) according to your needs.
    *   **Simulation parameters:** Change the arguments for the `python run_simulation_cli.py` command. You can specify the input CIF file, output directory, model, temperature, and other parameters.

2.  **Submit the job:**
    Use the `qsub` command to submit your job script:
    ```bash
    qsub job_template.sh
    ```

    The results will be saved in the directory you specified with the `--output-dir` argument.

## Example

The `job_template.sh` is pre-configured to run a simulation on `BiCoO3_tetra_1244_op_25.cif` using the `CHGNet` model. You can run it directly after setting up the environment.

```bash
# 1. Set up the environment (only once)
bash setup_env.sh

# 2. Submit the example job
qsub job_template.sh
```

## Note on Models

*   The scripts can use `CHGNet`, `MatterSim`, `Orb`, and `NequipOLM` models.
*   If you use the `NequipOLM` model, you must place the compiled model file (`nequip-oam-l.nequip.pt2`) in a directory named `NequipOLM_model` within this (`TSUBAME`) directory. The expected path is `./NequipOLM_model/nequip-oam-l.nequip.pt2`.
