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
pip install pandas ase matplotlib torch joblib chgnet mattersim orb-models nequip numpy nglview pynanoflann matgl dgl tqdm

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
