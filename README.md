# CLI Molecular Dynamics Simulator

## Overview

This tool runs molecular dynamics (MD) simulations using Machine Learning Force Fields (MLFF) in a self-contained, portable Docker environment. It is designed to be run from the command line, taking a CIF file as input and producing a variety of analysis results, including trajectory data, plots, and statistical summaries.

The entire environment is containerized, ensuring that the exact same calculation results can be obtained on any machine with Docker and an NVIDIA GPU, from a local PC to a supercomputer.

## Prerequisites

- **Docker:** The tool is built on Docker. You must have Docker installed.
- **NVIDIA GPU & Drivers:** The simulations are hardware-accelerated. An NVIDIA GPU with appropriate drivers and CUDA support accessible to Docker is required.

## Usage

All operations, from building the Docker image to running the simulation, are handled by the `run_cli.sh` script.

### Basic Command

The script is executed with the following format:
```bash
bash run_cli.sh <path_to_input_cif> <path_to_output_dir> [OPTIONS]
```
- `<path_to_input_cif>`: **(Required)** The relative or absolute path to the input structure file in CIF format.
- `<path_to_output_dir>`: **(Required)** The relative or absolute path to the directory where all results will be saved. The script will create this directory if it does not exist.
- `[OPTIONS]`: Optional arguments to customize the simulation. See the list below for details.

### Example

To run a full simulation on `my_structure.cif` and save the results in a folder named `my_results`, with the simulation ending at 500K:

```bash
bash run_cli.sh ./my_structure.cif ./my_results --temp-end 500
```

**Note:** The first time you run the script, it will build the Docker image, which may take several minutes. Subsequent runs will be much faster as they will use the existing image.

**Note on Models:** Some models, like NequipOLM, require a one-time compilation step. This will happen automatically the first time you run a simulation with that model. The compiled model will be saved in a new `models/` directory in your project folder to avoid recompilation on future runs.

## Command-Line Arguments (`OPTIONS`)

The following optional arguments can be passed to the script to control the simulation details.

| Argument | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--job-type` | string | `full_simulation` | `full_simulation`, `optimize_only` | Specifies the type of job. `optimize_only` will only perform structure optimization. |
| `--model` | string | `CHGNet` | `CHGNet`, `MatterSim`, `Orb`, `NequipOLM`, `MatRIS` | The Machine Learning Force Field model to use for the calculation. (Note: MatRIS is included by cloning its GitHub repository into the image). |
| `--sim-mode` | string | `Realistic (ISIF=3)` | `Realistic (ISIF=3)`, `Legacy (Orthorombic)` | The simulation mode for cell relaxation. |
| `--skip-optimization`| flag | `False` | | If specified, the initial structure optimization step will be skipped. |
| `--fmax` | float | `0.001` | | Maximum force (eV/Å) for structure optimization convergence. |
| `--magmom-specie`| string | `Co` | | The chemical symbol of the species for which to track magnetic moments (e.g., 'Co', 'Fe'). |
| `--temp-start` | integer | `1` | | The starting temperature for the NPT simulation in Kelvin. |
| `--temp-end` | integer | `1000` | | The ending temperature for the NPT simulation in Kelvin. |
| `--temp-step` | integer | `5` | | The temperature step size for the NPT simulation in Kelvin. |
| `--eq-steps` | integer | `2000` | | The number of equilibration steps to run at each temperature point. |
| `--n-gpu-jobs` | integer | `3` | | The number of parallel jobs to run during the NPT simulation. Adjust based on GPU memory. |
| `--enable-cooling`| flag | `False` | | If specified, a cooling phase will be run after the heating phase, returning to the start temperature. |

## Output Files

The simulation will produce the following files in the specified output directory:

- `optimized_structure.cif`: The initial structure after geometry optimization.
- `trajectory.xyz`: The full trajectory of the NPT simulation in XYZ format.
- `npt_vs_temp.png`: A plot of lattice parameters and volume against temperature.
- `npt_summary_full.csv`: A CSV file with detailed data for every step of the simulation.
- `npt_last_steps.csv`: A CSV file containing only the last step data for each temperature point.
- `npt_summary_stats.csv`: A CSV file with the mean and standard deviation of properties at each temperature.
- `magmoms_per_atom.csv`: (If `magmom_specie` is used) A CSV file tracking the magnetic moment of each specified atom over time.
