#!/bin/bash
# /cli_runner/run_cli.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
IMAGE_NAME_FULL="md-cli-runner"
IMAGE_NAME_MINIMAL="md-cli-runner-minimal"
IMAGE_NAME="$IMAGE_NAME_FULL"
DOCKERFILE="Dockerfile.cli"

# Get the absolute path of the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Helper Functions ---
build_image() {
    local name=$1
    local file=$2
    echo "--- Building Docker image '$name' from '$file'... ---"
    docker build -t "$name" -f "$SCRIPT_DIR/$file" "$SCRIPT_DIR"
    echo "--- Build complete ---"
}

# --- Command Line Arguments ---
if [ "$1" == "build" ]; then
    build_image "$IMAGE_NAME_FULL" "Dockerfile.cli"
    exit 0
elif [ "$1" == "build-minimal" ]; then
    build_image "$IMAGE_NAME_MINIMAL" "Dockerfile.minimal"
    exit 0
fi

# Check for --minimal flag in regular run
if [[ "$*" == *"--minimal"* ]]; then
    IMAGE_NAME="$IMAGE_NAME_MINIMAL"
    DOCKERFILE="Dockerfile.minimal"
    # Remove --minimal from arguments
    set -- "${@/--minimal/}"
fi

# --- Main Logic ---
# Auto-build if image doesn't exist
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    build_image "$IMAGE_NAME" "$DOCKERFILE"
fi

if [ "$#" -eq 0 ]; then
    # No arguments provided: Run automatic test
    echo "--- No arguments provided. Running automatic test inside container... ---"
    docker run --rm --gpus all "$IMAGE_NAME"
    exit 0
fi

# Argument Validation for custom run
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_input_cif> <path_to_output_dir> [additional_args_for_cli...]"
    echo "Special commands:"
    echo "  $0 build          - Build the full Docker image"
    echo "  $0 build-minimal  - Build the minimal Docker image (CHGNet only)"
    exit 1
fi

INPUT_CIF_PATH="$1"
OUTPUT_DIR_PATH="$2"
ADDITIONAL_ARGS="${@:3}"

if [ ! -f "$INPUT_CIF_PATH" ]; then
    echo "Error: Input CIF file not found at '$INPUT_CIF_PATH'"
    exit 1
fi

# --- Path Conversion & Setup ---
INPUT_CIF_ABS=$(realpath "$INPUT_CIF_PATH")
mkdir -p "$OUTPUT_DIR_PATH"
mkdir -p "$SCRIPT_DIR/models/NequipOLM_model"
OUTPUT_DIR_ABS=$(realpath "$OUTPUT_DIR_PATH")
MODELS_DIR_ABS=$(realpath "$SCRIPT_DIR/models")

CONTAINER_INPUT_DIR="/inputs"
CONTAINER_OUTPUT_DIR="/outputs"
CONTAINER_CIF_PATH="${CONTAINER_INPUT_DIR}/$(basename "$INPUT_CIF_ABS")"

# --- Docker Run ---
echo "--- Starting simulation container using image: $IMAGE_NAME ---"
docker run \
    --rm \
    --gpus all \
    -v "$(dirname "$INPUT_CIF_ABS")":"$CONTAINER_INPUT_DIR" \
    -v "$OUTPUT_DIR_ABS":"$CONTAINER_OUTPUT_DIR" \
    -v "$MODELS_DIR_ABS/NequipOLM_model":"/workspace/NequipOLM_model" \
    "$IMAGE_NAME" \
    --cif-path "$CONTAINER_CIF_PATH" \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    $ADDITIONAL_ARGS

echo "--- Simulation finished successfully! ---"
echo "Results are saved in: $OUTPUT_DIR_ABS"
