#!/bin/bash
# /workspace/entrypoint.sh

# This script is the entrypoint for the Docker container.
# It handles model compilation (if needed) and resource monitoring.

set -e

# --- Compilation for Nequip ---
MODEL_PATH="/workspace/NequipOLM_model/nequip-oam-l.nequip.pt2"
if [ ! -f "$MODEL_PATH" ]; then
    echo "--- Nequip model not found. Compiling... (This will only happen once) ---"
    mkdir -p "$(dirname "$MODEL_PATH")"
    nequip-compile \
        nequip.net:mir-group/NequIP-OAM-L:0.1 \
        "$MODEL_PATH" \
        --mode aotinductor \
        --device cuda \
        --target ase || echo "Warning: Nequip compilation failed. Some features may be unavailable."
    echo "--- Nequip model compilation complete. ---"
fi

# --- Main Execution ---
if [ $# -eq 0 ]; then
    # Default behavior: Run a test single-point calculation
    echo "--- No arguments provided. Running automatic test calculation... ---"
    CIF_FILE="BiCoO3_tetra_222.cif"
    MODEL="chgnet"
    
    # Check if the CIF file exists, if not use any available .cif
    if [ ! -f "$CIF_FILE" ]; then
        CIF_FILE=$(ls *.cif | head -n 1)
    fi

    echo "Running test: model=$MODEL, file=$CIF_FILE"
    python3 monitor_resources.py python3 run_simulation_cli.py --cif "$CIF_FILE" --model "$MODEL" --task single_point
else
    # Execute with provided arguments
    echo "--- Executing simulation with arguments: $@ ---"
    python3 monitor_resources.py python3 run_simulation_cli.py "$@"
fi
