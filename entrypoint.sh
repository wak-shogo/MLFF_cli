#!/bin/bash
# /cli_runner/entrypoint.sh

# This script is the entrypoint for the Docker container.
# It first checks if the Nequip model has been compiled, compiles it if necessary,
# and then executes the main simulation CLI script.

set -e

MODEL_PATH="/workspace/NequipOLM_model/nequip-oam-l.nequip.pt2"

# Check if the model file exists. If not, compile it.
# This runs inside the container at runtime, so it has GPU access.
if [ ! -f "$MODEL_PATH" ]; then
    echo "--- Nequip model not found. Compiling... (This will only happen once) ---"
    # Create the directory just in case
    mkdir -p "$(dirname "$MODEL_PATH")"
    # Run the compilation
    nequip-compile \
        nequip.net:mir-group/NequIP-OAM-L:0.1 \
        "$MODEL_PATH" \
        --mode aotinductor \
        --device cuda \
        --target ase
    echo "--- Nequip model compilation complete. ---"
else
    echo "--- Nequip model found. Skipping compilation. ---"
fi

# Execute the main Python script, passing along all arguments given to the container
echo "--- Starting main simulation script ---"
exec python3 run_simulation_cli.py "$@"
