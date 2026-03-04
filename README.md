# CLI Molecular Dynamics Simulator

## Overview

This tool runs molecular dynamics (MD) simulations using Machine Learning Force Fields (MLFF) in a self-contained, portable Docker environment. It is designed to be run from the command line, taking a CIF file as input and producing a variety of analysis results, including trajectory data, plots, and statistical summaries.

The entire environment is containerized, ensuring that the exact same calculation results can be obtained on any machine with Docker and an NVIDIA GPU, from a local PC to a supercomputer.

## Prerequisites

- **Docker:** The tool is built on Docker. You must have Docker installed.
- **NVIDIA GPU & Drivers:** The simulations are hardware-accelerated. An NVIDIA GPU with appropriate drivers and CUDA support accessible to Docker is required.

## Quick Start (Automatic Test)

To verify the installation and see the tool in action, simply run the helper script without any arguments:

```bash
bash run_cli.sh
```

This will automatically:
1.  Build the Docker image (if not already built).
2.  Start a container.
3.  Perform a **single-point energy calculation** on a default structure (`BiCoO3_tetra_222.cif`).
4.  **Monitor and report CPU/GPU usage** during the calculation.

## Usage

All operations are handled by the `run_cli.sh` script.

### Basic Command

The script is executed with the following format:
```bash
bash run_cli.sh <path_to_input_cif> <path_to_output_dir> [OPTIONS]
```
- `<path_to_input_cif>`: **(Required)** The relative or absolute path to the input structure file in CIF format.
- `<path_to_output_dir>`: **(Required)** The relative or absolute path to the directory where all results will be saved.
- `[OPTIONS]`: Optional arguments to customize the simulation.

### Example

To run a full simulation on `my_structure.cif` and save the results in `my_results`:

```bash
bash run_cli.sh ./my_structure.cif ./my_results --temp-end 500 --model CHGNet
```

## Resource Monitoring

Every run automatically monitors and outputs a resource usage summary at the end:

```text
========================================
      RESOURCE USAGE SUMMARY
========================================
CPU Usage:    Avg: 12.5%, Max: 45.0%
RAM Usage:    Avg: 8.2%, Max: 10.5%
GPU Usage:    Avg: 65.2%, Max: 98.0%
GPU VRAM:     Avg: 15.4%, Max: 20.1%
========================================
```

## Advanced: Running with Docker directly

If you prefer to run the container without the helper script, use the following command:

```bash
docker run --rm --gpus all \
    -v $(pwd)/inputs:/inputs \
    -v $(pwd)/outputs:/outputs \
    md-cli-runner --cif-path /inputs/your_file.cif --output-dir /outputs --model MatterSim
```

## Command-Line Arguments (`OPTIONS`)

| Argument | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--task` | string | `full_simulation` | `full_simulation`, `optimize_only`, `single_point` | Specifies the type of task. |
| `--model` | string | `CHGNet` | `CHGNet`, `MatterSim`, `Orb`, `NequipOLM`, `MatRIS` | The MLFF model to use. |
| `--temp-start` | integer | `1` | | Starting temperature (K). |
| `--temp-end` | integer | `1000` | | Ending temperature (K). |
| `--eq-steps` | integer | `2000` | | Equilibration steps per temperature point. |
| `--n-gpu-jobs` | integer | `3` | | Number of parallel GPU jobs. |

## Output Files

- `optimized_structure.cif`: Initial structure after optimization.
- `trajectory.xyz`: Full trajectory of the simulation.
- `npt_vs_temp.png`: Plot of lattice parameters vs temperature.
- `npt_summary_full.csv`: Detailed data for every step.
- `npt_summary_stats.csv`: Mean and standard deviation of properties.
