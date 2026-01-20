# VASP and MLFF Energy/Force Comparison Tool

This tool calculates and compares the total energy and atomic forces of a given crystal structure using both a selected Machine Learning Force Field (MLFF) and VASP.

## Features

-   Performs single-point (no relaxation) calculations.
-   Compares results from a chosen MLFF (`CHGNet`, `MatRIS`, etc.) against VASP.
-   Automates the VASP calculation process:
    -   Generates VASP input files (`INCAR`, `POSCAR`, `KPOINTS`, `POTCAR`) using `pymatgen`'s `MPStaticSet`.
    -   Generates a PBS job script for a CPU-only VASP calculation.
    -   Submits the job via `qsub` and waits for it to complete by monitoring `qstat`.
    -   Automatically parses the energy and forces from the VASP output.
-   Accepts single `.cif` files or multi-frame `.xyz` trajectory files as input.
-   Outputs results into separate, clean CSV files for energies and forces.

## 1. Setup

This tool is designed to be run on a VASP calculation server with a PBS job scheduler and a Conda installation (like Miniconda or Anaconda).

1.  **Create and set up the Conda environment:**
    Run the provided setup script. This will create a dedicated Conda environment (`vasp_compare_env`) with Python 3.10 and install all necessary packages.
    ```bash
    bash setup_vasp_env.sh
    ```

2.  **Activate the environment:**
    Before running the main script, you must activate the Conda environment:
    ```bash
    conda activate vasp_compare_env
    ```

## 2. Usage

The main script is `compare_energy_forces.py`.

### Command

```bash
python compare_energy_forces.py --input-path <path_to_file> --mlff-model <model_name> --output-dir <path_to_dir> [OPTIONS]
```

### Arguments

-   `--input-path`: **(Required)** Path to the input `.cif` or `.xyz` file.
-   `--mlff-model`: **(Required)** The MLFF model to use for comparison (e.g., `CHGNet`, `MatRIS`, `NequipOLM`).
-   `--output-dir`: **(Required)** The directory where the output CSV files will be saved.
-   `--num-snapshots`: (Optional) If the input is an `.xyz` file, this specifies how many random frames to select for calculation. Defaults to `1`.

### Example

```bash
# Compare CHGNet and VASP for a single structure
python compare_energy_forces.py \
    --input-path ./my_structure.cif \
    --mlff-model CHGNet \
    --output-dir ./comparison_results

# Compare MatRIS and VASP for 5 random structures from a trajectory
python compare_energy_forces.py \
    --input-path ./my_trajectory.xyz \
    --mlff-model MatRIS \
    --num-snapshots 5 \
    --output-dir ./comparison_results
```

## 3. Output Files

The script will generate the following files in your specified `--output-dir`:

-   `energies.csv`: A summary file containing the calculated energies for each structure.
    -   Columns: `structure_name`, `<MLFF_MODEL>_energy`, `VASP_energy`
-   `<structure_name>_forces.csv`: A separate file for each calculated structure, detailing the forces on each atom.
    -   Columns: `atom_index`, `symbol`, `MLFF_Fx`, `MLFF_Fy`, `MLFF_Fz`, `VASP_Fx`, `VASP_Fy`, `VASP_Fz`
