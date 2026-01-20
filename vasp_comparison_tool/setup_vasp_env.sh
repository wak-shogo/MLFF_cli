#!/bin/bash
# setup_vasp_env.sh
# This script sets up a Conda environment for the VASP/MLFF comparison tool.

# --- Configuration ---
ENV_NAME="vasp_compare_env"
PYTHON_VERSION="3.10"

# --- Main Script ---
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."

# Check if the environment already exists
if conda info --envs | grep -q "^\s*$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists. For a clean install, remove it first with 'conda env remove -n $ENV_NAME'"
else
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Conda environment. Please check your Conda installation."
        exit 1
    fi
fi

echo "Activating the environment to install packages..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Step 1: Installing a modern compiler toolchain from conda-forge..."
# This is the key step: install modern compilers into the environment.
# pip will automatically find and use these instead of the old system compilers.
conda install -c conda-forge gcc_linux-64 gxx_linux-64 go rust hdf5 -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to install compilers from conda-forge."
    exit 1
fi

echo "Step 2: Installing PyTorch from the official pytorch channel..."
conda install pytorch cpuonly -c pytorch -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to install PyTorch."
    exit 1
fi

echo "Step 3: Installing all remaining packages with pip..."
# Create a temporary directory within the project for pip builds.
# This avoids running out of space or inodes on the system's /tmp partition.
BUILD_DIR=".pip_build_tmp"
mkdir -p "$BUILD_DIR"

# With the new compilers in place, pip can now build packages from source without errors.
# We direct its temporary files to our local build directory.
TMPDIR=$(pwd)/"$BUILD_DIR" pip install pandas ase matplotlib pymatgen chgnet mattersim orb-models nequip-allegro

# Clean up the temporary build directory
rm -rf "$BUILD_DIR"

# MatRIS (clone)
echo "Cloning MatRIS from GitHub..."
if [ -d "MatRIS" ]; then
    echo "MatRIS directory already exists, skipping clone."
else
    git clone https://github.com/HPC-AI-Team/MatRIS.git
fi

echo "Deactivating environment."
conda deactivate

echo "---"
echo "Setup complete."
echo "To activate the environment, run:"
echo "conda activate $ENV_NAME"
echo "---"
