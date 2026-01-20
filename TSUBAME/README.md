# MACEMP Simulation on TSUBAME

This directory contains the necessary scripts to run machine learning force field simulations on the TSUBAME supercomputer without using Docker.

## 1. Environment Setup

First, you need to create a dedicated Python virtual environment and download the required model repositories.

1.  **Set up the Python environment:**
    The provided script will install all necessary pip packages.
    ```bash
    bash setup_env.sh
    ```

2.  **Clone the MatRIS repository (if using the MatRIS model):**
    ```bash
    git clone https://github.com/HPC-AI-Team/MatRIS.git
    ```

3.  **Activate the environment:**
    Before running any simulation, you must activate the environment in your shell:
    ```bash
    source ~/work/python-venvs/macemp_env/bin/activate
    ```

## 2. Running a Simulation

Simulations are submitted as jobs to the TSUBAME scheduler. A template job script `job_template.sh` is provided.

1.  **Customize the job script:**
    Open `job_template.sh` and edit the settings:
    *   **TSUBAME settings:** Modify the `#$` lines (e.g., `-l f_node`, `-l h_rt`) according to your needs.
    *   **Simulation parameters:** Change the arguments for the `python run_simulation_cli.py` command. You can specify the input CIF file, output directory, model, temperature, and other parameters. For example, you can add `--skip-optimization` to bypass the initial structure relaxation before the NPT simulation, or `--fmax 0.005` to set a custom convergence criterion for optimization.

2.  **Submit the job:**
    Use the `qsub` command to submit your job script:
    ```bash
    qsub job_template.sh
    ```

    The results will be saved in the directory you specified with the `--output-dir` argument.

## Example

The `job_template.sh` is pre-configured to run a simulation on `BiCoO3_tetra_1244_op_25.cif` using the `CHGNet_r2SCAN` model. You can run it directly after setting up the environment.

```bash
# 1. Set up the environment (only once)
bash setup_env.sh

# 2. Submit the example job
qsub job_template.sh
```

## Note on Models

*   The scripts can use `CHGNet`, `CHGNet_r2SCAN`, `MatterSim`, `Orb`, `NequipOLM`, and `MatRIS` models.
*   If you use the `NequipOLM` model, you must place the compiled model file (`nequip-oam-l.nequip.pt2`) in a directory named `NequipOLM_model` within this (`TSUBAME`) directory. The expected path is `./NequipOLM_model/nequip-oam-l.nequip.pt2`.
*   **CHGNet_r2SCAN Model:** The `setup_env.sh` script automatically downloads this model to `./models/CHGNet_r2SCAN`. If the download fails, you can manually place the model folder there.
