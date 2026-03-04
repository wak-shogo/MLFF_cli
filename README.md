# CLI Molecular Dynamics Simulator

## Overview

This tool runs molecular dynamics (MD) simulations using Machine Learning Force Fields (MLFF) in a self-contained, portable Docker environment.

## Prerequisites

- **Docker:** Must be installed and running.
- **NVIDIA GPU & Drivers:** Required for hardware acceleration.

## Build Instructions

You can build the Docker images manually before running simulations.

### Full Image (All Models)
Includes CHGNet, MatterSim, Orb, Nequip, and MatRIS.
```bash
bash run_cli.sh build
```

### Minimal Image (CHGNet Only)
Includes only CHGNet and CHGNet_r2SCAN. Fast build, smaller disk footprint.
```bash
bash run_cli.sh build-minimal
```

## Quick Start (Automatic Test)

To verify the installation, simply run:
```bash
bash run_cli.sh
```
Or for the minimal environment:
```bash
bash run_cli.sh --minimal
```

## Usage

### Basic Command
```bash
bash run_cli.sh <path_to_input_cif> <path_to_output_dir> [OPTIONS]
```

### Options
- `--minimal`: Use the minimal Docker image (must build it first or it will auto-build).
- `--model`: `CHGNet`, `CHGNet_r2SCAN`, `MatterSim`, `Orb`, `NequipOLM`, `MatRIS`
- `--job-type`: `full_simulation`, `optimize_only`

### Example (Minimal Run)
```bash
bash run_cli.sh ./my_structure.cif ./my_results --minimal --model CHGNet
```

## Resource Monitoring

Every run automatically reports CPU, RAM, and GPU usage at the end of the simulation.

```text
========================================
      RESOURCE USAGE SUMMARY
========================================
CPU Usage:    Avg: 12.5%, Max: 45.0%
...
========================================
```

## Output Files

- `optimized_structure.cif`: Initial structure after optimization.
- `trajectory.xyz`: Full trajectory of the simulation.
- `npt_vs_temp.png`: Plot of lattice parameters vs temperature.
- `npt_summary_stats.csv`: Mean and standard deviation of properties.
