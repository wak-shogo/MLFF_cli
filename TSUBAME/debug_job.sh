#!/bin/bash
# debug_job.sh
# This is a debugging job script for TSUBAME.
# It includes extra commands to trace execution and check the environment.

# --- TSUBAME Job Settings ---
#$ -cwd
#$ -N macemp_debug
#$ -l node_q=1
#$ -l h_rt=00:10:00
#$ -o simulation.out
#$ -e simulation.err

# --- Execution Tracing ---
# Print every command as it is executed to stderr
set -x

# --- Environment Setup ---
echo "--- Setting up environment ---"
. /etc/profile.d/modules.sh

echo "Loading CUDA module..."
module load cuda/12.3.2

echo "Checking GPU status..."
nvidia-smi

echo "Activating Python virtual environment..."
VENV_DIR="$HOME/python-venvs/macemp_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Python environment not found at $VENV_DIR" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "Checking Python executable path..."
which python

echo "Listing installed packages..."
pip list

# --- Simulation Execution ---

echo "--- Starting simulation script ---"

python run_simulation_cli.py \
    --cif-path ./BiCoO3_tetra_1244_op_25.cif \
    --output-dir ./results/BiCoO3_CHGNet_debug \
    --model CHGNet \
    --temp-end 100 \
    --n-gpu-jobs 1

echo "--- Simulation script finished ---"

# --- Deactivate Environment ---
deactivate
