#!/bin/bash

# example_run.sh - Example script showing how to run the test infrastructure
# Copy this file and modify the configuration variables for your setup

set -e

#############################################################################
# CONFIGURATION - Modify these variables for your environment
#############################################################################

# Docker container name
CONTAINER_NAME="your_container_name"

# Path to hipblaslt inside the Docker container
DOCKER_HIPBLASLT_PATH="/workspace/rocm-libraries_p/projects/hipblaslt"

# Path to hipblaslt on the host system
HOST_HIPBLASLT_PATH="/home/${USER}/rocm-libraries_p/projects/hipblaslt"

# Path to stinkytofu test tools (optional)
# Leave empty to use the default location (same directory as this script)
# Set this if you want to decouple the test tools from the hipblaslt repository
HOST_STINKYTOFU_TOOL_PATH=""

# ROCm cmake build system: 0=old (tensile_client), 1=new (tensilelite-client)
# The old system requires building tensilelite-client separately
# The new system builds everything in one unified cmake build
NEW_ROCM_CMAKE_BUILD_SYSTEM="0"

# Output directory for test results
# Results will be organized as:
#   ${OUTPUT_BASE_DIR}/${gpu_arch}/database/     - JSON and reports
#   ${OUTPUT_BASE_DIR}/${gpu_arch}/logs/         - Detailed logs by session
OUTPUT_BASE_DIR="./regression"

# Test type: exe_time, codegen_time, dbg_verify, or all
TEST_TYPE="all"

# Test pattern (regex) - use ".*" for all tests
TEST_PATTERN=".*"

# GPU architecture (leave empty for auto-detection)
GPU_ARCH=""

# Base log directory for comparison against current run
BASE_DIR_FOR_COMPARISON=""

# Preserve build directory: 0=disabled (clean up), 1=enabled (keep ${OUT_DIR})
# Enable this to reuse build artifacts for faster re-runs or debugging
PRESERVE_BUILD=0

# Verbose mode: 0=disabled (concise output), 1=enabled (detailed output)
VERBOSE=0

#############################################################################
# DO NOT MODIFY BELOW THIS LINE
#############################################################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If custom tool path is provided, use that for scripts and utilities
if [[ -n "$HOST_STINKYTOFU_TOOL_PATH" ]]; then
    SCRIPT_DIR="${HOST_STINKYTOFU_TOOL_PATH}/scripts"
fi

# Use OUTPUT_BASE_DIR directly (timestamp will be added automatically in logs subfolder)
OUTPUT_DIR="${OUTPUT_BASE_DIR}"

# Use Python utilities to get git info, hostname, and GPU arch
GIT_INFO=$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from lib.utils import get_git_info
print(get_git_info('${HOST_HIPBLASLT_PATH}'))
")

HOSTNAME=$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from lib.utils import get_hostname
print(get_hostname())
")

# Get GPU arch - use auto-detection if not specified
if [[ -z "$GPU_ARCH" ]]; then
    GPU_ARCH=$(/opt/rocm/bin/rocm_agent_enumerator | grep -v gfx000 | head -n1)
fi

echo "=============================================="
echo "stinkytofu Test Infrastructure"
echo "=============================================="
echo "Container: $CONTAINER_NAME"
echo "Test type: $TEST_TYPE"
echo "Output dir: $OUTPUT_DIR"
if [[ -n "$HOST_STINKYTOFU_TOOL_PATH" ]]; then
    echo "Tool path: $HOST_STINKYTOFU_TOOL_PATH"
fi
echo "CMake system: $([ "$NEW_ROCM_CMAKE_BUILD_SYSTEM" == "1" ] && echo 'new' || echo 'old')"
echo "Preserve build: $([ "$PRESERVE_BUILD" == "1" ] && echo 'enabled' || echo 'disabled')"
echo "=============================================="
echo ""

if [[ -n "$BASE_DIR_FOR_COMPARISON" ]]; then
    echo "Base directory for comparison: $BASE_DIR_FOR_COMPARISON"
else
    echo "Warning: No base directory for comparison provided"
fi

# Check if configuration has been modified
if [[ "$CONTAINER_NAME" == "your_container_name" ]]; then
    echo "ERROR: Please edit this script and set the configuration variables!"
    echo "Edit the following variables in this script:"
    echo "  - CONTAINER_NAME"
    echo "  - DOCKER_HIPBLASLT_PATH"
    echo "  - HOST_HIPBLASLT_PATH"
    exit 1
fi

# Build the command
CMD="python3 ${SCRIPT_DIR}/test_runner.py"
CMD="$CMD --container-name \"$CONTAINER_NAME\""
CMD="$CMD --docker-path \"$DOCKER_HIPBLASLT_PATH\""
CMD="$CMD --host-path \"$HOST_HIPBLASLT_PATH\""
CMD="$CMD --output-dir \"$OUTPUT_DIR\""
CMD="$CMD --test \"$TEST_TYPE\""
CMD="$CMD --pattern \"$TEST_PATTERN\""

if [[ -n "$HOST_STINKYTOFU_TOOL_PATH" ]]; then
    CMD="$CMD --tool-path \"$HOST_STINKYTOFU_TOOL_PATH\""
fi

if [[ -n "$NEW_ROCM_CMAKE_BUILD_SYSTEM" ]]; then
    CMD="$CMD --new-cmake \"$NEW_ROCM_CMAKE_BUILD_SYSTEM\""
fi

if [[ -n "$GPU_ARCH" ]]; then
    CMD="$CMD --gpu-arch \"$GPU_ARCH\""
fi

if [[ $PRESERVE_BUILD -eq 1 ]]; then
    CMD="$CMD --preserve-build"
fi

if [[ $VERBOSE -eq 1 ]]; then
    CMD="$CMD --verbose"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the tests
eval "$CMD"

python3 "${SCRIPT_DIR}/export_database_csv.py" \
 --database-dir "${OUTPUT_DIR}/${GPU_ARCH}/database/" \
 --hostname ${HOSTNAME} \
 --gpu-arch ${GPU_ARCH}

if [[ -n "$BASE_DIR_FOR_COMPARISON" ]]; then
    echo "Comparing logs..."
    python3 "${SCRIPT_DIR}/compare_logs.py" \
    --base-folder "${OUTPUT_DIR}/${GPU_ARCH}/logs/${BASE_DIR_FOR_COMPARISON}/" \
    --compared-folder "${OUTPUT_DIR}/${GPU_ARCH}/logs/${HOSTNAME}-${GIT_INFO}/" \
    --output-prefix my_comparison
    echo ""
    cat my_comparison_report.txt
    echo ""
fi
