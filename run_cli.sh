#!/bin/bash
# /cli_runner/run_cli.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
IMAGE_NAME="md-cli-runner"

# --- Argument Validation ---
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_input_cif> <path_to_output_dir> [additional_args_for_cli...]"
    echo "Example: $0 ./my_structure.cif ./results --model CHGNet --temp-end 500"
    exit 1
fi

INPUT_CIF_PATH="$1"
OUTPUT_DIR_PATH="$2"
# Get all arguments from the 3rd one onwards
ADDITIONAL_ARGS="${@:3}"

if [ ! -f "$INPUT_CIF_PATH" ]; then
    echo "Error: Input CIF file not found at '$INPUT_CIF_PATH'"
    exit 1
fi

# --- Path Conversion & Setup ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Convert user-provided paths to absolute paths to ensure they mount correctly
INPUT_CIF_ABS=$(realpath "$INPUT_CIF_PATH")
# Create the output directory on the host if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"
OUTPUT_DIR_ABS=$(realpath "$OUTPUT_DIR_PATH")

# Define paths as they will be seen inside the container
CONTAINER_INPUT_DIR="/inputs"
CONTAINER_OUTPUT_DIR="/outputs"
CONTAINER_CIF_PATH="${CONTAINER_INPUT_DIR}/$(basename "$INPUT_CIF_ABS")"

# --- Docker Build ---
# Check if the image already exists. If not, build it.
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "--- Docker image '$IMAGE_NAME' not found. Building... ---"
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile.cli" "$SCRIPT_DIR"
    echo "--- Build complete ---"
else
    echo "--- Using existing Docker image '$IMAGE_NAME' ---"
fi


# --- Docker Run ---
echo "--- Starting simulation container ---"
echo "Host input CIF: $INPUT_CIF_ABS"
echo "Host output dir:  $OUTPUT_DIR_ABS"
echo "Container CIF path: $CONTAINER_CIF_PATH"
echo "-------------------------------------"

# Execute the docker container
# --rm: Automatically remove the container when it exits
# --gpus all: Provide access to all available GPUs
# -v: Mount volumes to get data in and out of the container
#   - Mount the directory containing the input file as /inputs
#   - Mount the host output directory as /outputs
docker run \
    --rm \
    --gpus all \
    -v "$(dirname "$INPUT_CIF_ABS")":"$CONTAINER_INPUT_DIR" \
    -v "$OUTPUT_DIR_ABS":"$CONTAINER_OUTPUT_DIR" \
    "$IMAGE_NAME" \
    --cif-path "$CONTAINER_CIF_PATH" \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    $ADDITIONAL_ARGS

echo "--- Simulation finished successfully! ---"
echo "Results are saved in: $OUTPUT_DIR_ABS"
