#!/bin/bash
# setup_env.sh
# This script sets up a Python virtual environment for the simulation.

# --- Configuration ---
VENV_DIR="$HOME/python-venvs/macemp_env"

# --- Main Script ---
echo "Creating Python virtual environment at: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "Activating the environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading setuptools..."
pip install --upgrade pip setuptools

echo "Installing required Python packages..."

# 1. Install PyTorch (CUDA 12.1 version)
# Using CUDA 12.1 as it is compatible with the TSUBAME environment (Driver 12.9)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install DGL (CUDA 12.1 version compatible with Torch 2.4)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# 3. Install other dependencies
pip install pandas ase matplotlib joblib chgnet mattersim orb-models nequip numpy nglview pynanoflann matgl tqdm

echo "---"
echo "Downloading CHGNet_r2SCAN model..."
mkdir -p models
if [ ! -d "models/CHGNet_r2SCAN" ]; then
    git clone https://github.com/materialyzeai/matgl.git matgl_temp
    if [ -d "matgl_temp/pretrained_models/CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES" ]; then
        cp -r matgl_temp/pretrained_models/CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES models/CHGNet_r2SCAN
        echo "✅ CHGNet_r2SCAN downloaded successfully to ./models/CHGNet_r2SCAN"
    else
        echo "❌ Error: Model folder not found in cloned repo. Please check the repository structure."
    fi
    rm -rf matgl_temp
else
    echo "CHGNet_r2SCAN model already exists in ./models/CHGNet_r2SCAN. Skipping download."
fi

echo "---"
echo "IMPORTANT: To use the MatRIS model, you must manually clone the repository into this directory:"
echo "git clone https://github.com/HPC-AI-Team/MatRIS.git"
echo "---"


echo "Deactivating environment."
deactivate

echo "---"
echo "Setup complete."
echo "To activate the environment, run:"
echo "source $VENV_DIR/bin/activate"
echo "---"
