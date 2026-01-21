#!/bin/bash
# job_template.sh
# This is a template for a TSUBAME job script.
# You need to fill in the job-specific details and arguments.

# --- TSUBAME Job Settings ---
#$ -cwd
#$ -N macemp_simulation
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -o simulation.out
#$ -e simulation.err

# --- Environment Setup ---
echo "--- Setting up environment ---"
. /etc/profile.d/modules.sh

echo "Loading CUDA module..."
module load cuda/12.3.2

echo "Activating Python virtual environment..."
VENV_DIR="$HOME/python-venvs/macemp_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Python environment not found at $VENV_DIR" >&2
    echo "Please run the setup_env.sh script first." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Add the MatRIS repository to the Python Path
export PYTHONPATH="$SGE_O_WORKDIR/MatRIS:$PYTHONPATH"

# --- Simulation Execution ---
echo "--- Starting simulation script ---"

# Run the simulation script
# --cif-path: Path to the input structure file.
# --output-dir: Directory to save results.
# --model: The ML force field to use.
# --temp-end: The maximum temperature for the simulation.
# --n-gpu-jobs: Number of parallel jobs (should match GPU resources)
python run_simulation_cli.py \
    --cif-path ./BiCoO3_tetra_1244_op_25.cif \
    --output-dir ./results/BiCoO3_CHGNet \
    --model CHGNet \
    --temp-end 1200 \
    --n-gpu-jobs 1

echo "--- Simulation script finished ---"

# --- Deactivate Environment ---
deactivate
