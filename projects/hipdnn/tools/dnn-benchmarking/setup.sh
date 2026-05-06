#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$HIPDNN_ROOT/../.." && pwd)"
INSTALL_DIR="/opt/rocm"
VENV_DIR="/workspace/.venv"

FORCE_BUILD=0
usage() {
    echo "Usage: $0 [--force-build] [--install-dir <path>]"
    echo ""
    echo "  --force-build        Force rebuild of hipDNN and the MIOpen provider,"
    echo "                           overwriting existing artifacts."
    echo "  --install-dir <path> Install prefix for hipDNN and the MIOpen provider."
    echo "                           Default: $INSTALL_DIR"
    echo ""
    echo "  The installed plugin will be at:"
    echo "    <install-dir>/lib/hipdnn_plugins/engines/"
    echo "  Pass that path to --plugin-path when benchmarking."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-build) FORCE_BUILD=1 ;;
        --install-dir) shift; INSTALL_DIR="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

# 1. Create or activate venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment at $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Redirect Python's bytecode cache away from the network home directory.
# The source tree lives on a network filesystem; without this, every import
# writes/reads .pyc files over the network. Must be injected into the venv
# activate script so it's set before the interpreter starts (setting it in
# Python code is too late for that process's own imports).
ACTIVATE_LOCAL="$VENV_DIR/bin/activate.local"
if [ ! -f "$ACTIVATE_LOCAL" ] || ! grep -q PYTHONPYCACHEPREFIX "$ACTIVATE_LOCAL"; then
    echo 'export PYTHONPYCACHEPREFIX=/workspace/pycache' >> "$ACTIVATE_LOCAL"
fi
if ! grep -q "activate.local" "$VENV_DIR/bin/activate"; then
    # shellcheck disable=SC2016
    echo 'source "$(dirname "${BASH_SOURCE[0]}")/activate.local" 2>/dev/null || true' \
        >> "$VENV_DIR/bin/activate"
fi
export PYTHONPYCACHEPREFIX=/workspace/pycache

# 2. Detect GPU architecture and install ROCm PyTorch from the matching nightly index.
detect_gpu_arch() {
    local arch
    if command -v rocm_agent_enumerator &>/dev/null; then
        arch=$(rocm_agent_enumerator | grep -m1 'gfx9')
    elif command -v rocminfo &>/dev/null; then
        arch=$(rocminfo | grep -oP 'gfx\d+' | head -1)
    fi
    echo "${arch:-}"
}

GPU_ARCH=$(detect_gpu_arch)
case "$GPU_ARCH" in
    gfx90*) INDEX_ARCH="gfx90X" ;;
    gfx94*) INDEX_ARCH="gfx94X" ;;
    *)
        echo "ERROR: Unsupported GPU architecture '${GPU_ARCH:-none}'."
        echo "Supported: gfx90a (MI200/MI210/MI250), gfx942 (MI300X/MI300A)"
        exit 1 ;;
esac

INDEX_URL="https://rocm.nightlies.amd.com/v2-staging/${INDEX_ARCH}-dcgpu/"
echo "Detected GPU: $GPU_ARCH → installing PyTorch from $INDEX_URL"

# Install ROCm torch first from its dedicated index. Then editable-install the
# package; pyproject.toml omits torch (so pip won't touch the already-installed
# ROCm build) and lists the rest (numpy, pytest, pytest-cov) which resolve
# cleanly from PyPI.
pip install --pre torch --index-url "$INDEX_URL"
pip install -e "$SCRIPT_DIR"

# 3. Build and install hipDNN + providers
# The installed cmake configs use install-tree paths; pointing CMAKE_PREFIX_PATH at
# the raw build dir causes "non-existent path" errors in hipdnn_data_sdkConfig.cmake.
HIPDNN_CONFIG="$INSTALL_DIR/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
if [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$HIPDNN_CONFIG" ]; then
    echo "Building and installing hipDNN and providers..."
    cd "$WORKSPACE_ROOT"
    cmake --preset hipdnn-providers
    cmake --install build
    cd "$SCRIPT_DIR"
fi

# 5. Install hipdnn Python bindings
# Build in a container-local directory so concurrent containers don't race on
# the shared-mount build/ directory (CMakeCache, object files, .so).
CMAKE_PREFIX_PATH="$INSTALL_DIR" \
    SKBUILD_BUILD_DIR="/workspace/hipdnn-python-build" \
    pip install -e "$HIPDNN_ROOT/python"

echo ""
echo "Setup complete. Activate the virtual environment with:"
echo "  source $VENV_DIR/bin/activate"
if [ "$FORCE_BUILD" -eq 1 ]; then
    echo ""
    echo "Plugins installed to: $INSTALL_DIR/lib/hipdnn_plugins/engines/"
    echo "Run benchmarks with:"
    echo "  python -m dnn_benchmarking --graph <graph.json> \\"
    echo "    --plugin-path $INSTALL_DIR/lib/hipdnn_plugins/engines"
fi
