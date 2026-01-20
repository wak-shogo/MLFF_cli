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
# Load the necessary modules.
. /etc/profile.d/modules.sh
# Ensure CUDA module matches the version we are installing for (12.1 compatible)
module load cuda/12.3.2

# Define Virtual Environment Path (Large Storage)
VENV_DIR="$HOME/work/python-venvs/macemp_env"

# Check if environment exists; if not, create it (on the GPU node)
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Virtual environment not found. Building it now... ---" 
    
    # Create directory
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
    
    # Activate
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools
    
    echo "Installing required Python packages..."
    # 1. Install PyTorch (CUDA 12.1 version)
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

    # 2. Install DGL (CUDA 12.1 version compatible with Torch 2.4)
    # Force reinstall to ensure no CPU cache is used
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html --force-reinstall

    # 3. Install other dependencies
    pip install pandas ase matplotlib joblib chgnet mattersim orb-models nequip numpy nglview pynanoflann matgl tqdm

    echo "--- Environment build complete ---"
else
    echo "--- Virtual environment found. Activating... ---"
    source "$VENV_DIR/bin/activate"
fi

# --- Model Download Check ---
# Ensure CHGNet_r2SCAN is available
MODEL_DIR="./models/CHGNet_r2SCAN"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading CHGNet_r2SCAN model..."
    mkdir -p models
    git clone https://github.com/materialyzeai/matgl.git matgl_temp
    if [ -d "matgl_temp/pretrained_models/CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES" ]; then
        cp -r matgl_temp/pretrained_models/CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES "$MODEL_DIR"
        echo "✅ CHGNet_r2SCAN downloaded successfully."
    else
        echo "❌ Error: Model folder not found in cloned repo."
    fi
    rm -rf matgl_temp
fi

# Add the MatRIS repository to the Python Path (if manually cloned)
export PYTHONPATH="$SGE_O_WORKDIR/MatRIS:$PYTHONPATH"

# --- Simulation Execution ---
echo "Starting simulation..."

# Run the simulation script
python run_simulation_cli.py \
    --cif-path ./BiCoO3_tetra_1244_op_25.cif \
    --output-dir ./results/BiCoO3_CHGNet \
    --model CHGNet_r2SCAN \
    --temp-end 1200 \
    --n-gpu-jobs 1

echo "Simulation finished."

# --- Deactivate Environment ---
deactivate